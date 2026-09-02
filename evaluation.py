"""Unified full-event closed-loop evaluation protocol."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PRIVATE_EXPERTS = [
    "Mamba",
    "Gipps",
    "GM",
    "FVD",
    "Wiedemann",
    "NETSIM",
    "Mod_NETSIM",
    "Fritzsche",
]
ALL_EXPERTS = PRIVATE_EXPERTS + ["Transformer_shared", "IDM_shared"]


@dataclass(frozen=True)
class EvaluationProtocol:
    history_steps: int = 20
    # 0 means: evaluate every event to its actual final frame.
    # A positive value is retained only as an optional debugging cap.
    horizon_steps: int = 0
    dt: float = 0.1
    min_acceleration: float = -4.0
    max_acceleration: float = 3.0
    max_jerk: float = 5.0
    min_speed: float = 0.0
    max_speed: float = 55.0
    collision_gap: float = 0.0
    unsafe_thw_threshold: float = 2.0
    thw_speed_epsilon: float = 1e-3
    normalized_state_limit: float = 25.0


def protocol_from_args(args) -> EvaluationProtocol:
    return EvaluationProtocol(
        history_steps=int(args.get("history_len", 20)),
        horizon_steps=int(args.get("test_horizon_steps", 0)),
        dt=float(args.get("dt", 0.1)),
        min_acceleration=float(args.get("min_acceleration", -4.0)),
        max_acceleration=float(args.get("max_acceleration", 3.0)),
        max_jerk=float(args.get("max_jerk", 5.0)),
        min_speed=float(args.get("min_speed", 0.0)),
        max_speed=float(args.get("max_speed", 55.0)),
        collision_gap=float(args.get("collision_gap", 0.0)),
        unsafe_thw_threshold=float(args.get("unsafe_thw_threshold", 2.0)),
        thw_speed_epsilon=float(args.get("thw_speed_epsilon", 1e-3)),
    )


def _predict(model, history):
    try:
        output = model(history, return_diagnostics=True)
    except TypeError:
        output = model(history)

    if not isinstance(output, tuple):
        return output, None, None, None

    prediction = output[0]
    second = output[1] if len(output) > 1 else None
    if isinstance(second, dict):
        return (
            prediction,
            second.get("private_router_weights"),
            second.get("contribution_share"),
            second.get("scene_probabilities"),
        )
    if torch.is_tensor(second):
        return prediction, second[:, :8], second, None
    return prediction, None, None, None


def rollout_event(
    model,
    history_normalized,
    future_physical,
    mean,
    std,
    protocol,
    dataset_name,
    model_name,
    seed,
    event_id,
):
    device = next(model.parameters()).device
    history = history_normalized.squeeze(0).to(device).float()
    future = future_physical.squeeze(0).to(device).float()
    if future.shape[0] == 0:
        return None
    actual_future_steps = int(future.shape[0])
    horizon_steps = (
        actual_future_steps
        if protocol.horizon_steps <= 0
        else min(actual_future_steps, protocol.horizon_steps)
    )
    future = future[:horizon_steps]

    mean_t = torch.as_tensor(mean, device=device, dtype=torch.float32)
    std_t = torch.as_tensor(std, device=device, dtype=torch.float32)
    history_physical = history * std_t + mean_t
    current_window = history.unsqueeze(0)
    current_gap = float(history_physical[-1, 0])
    current_speed = float(history_physical[-1, 1])
    current_relative_speed = float(history_physical[-1, 2])
    current_acceleration = (
        float(history_physical[-1, 1]) - float(history_physical[-2, 1])
    ) / protocol.dt

    predicted_gap, predicted_speed, predicted_acceleration = [], [], []
    router_history, contribution_history, scene_history = [], [], []
    collision = False
    unsafe_event = False
    numerical_failure = False
    collision_step = -1
    terminal = False
    unsafe_frames = 0
    thw_moving_frames = 0
    minimum_predicted_thw = float("inf")
    negative_speed_attempts = 0
    overspeed_attempts = 0

    with torch.inference_mode():
        for step in range(horizon_steps):
            observed_lead_speed = float(future[step, 3])
            if not terminal:
                output, router, contribution, scene = _predict(model, current_window)
                raw_acceleration = float(output.reshape(-1)[0].item())
                if not np.isfinite(raw_acceleration):
                    numerical_failure = True
                    terminal = True
                    applied_acceleration = 0.0
                    next_speed = 0.0
                    next_gap = protocol.collision_gap
                    next_relative_speed = observed_lead_speed
                else:
                    target_acceleration = float(
                        np.clip(
                            raw_acceleration,
                            protocol.min_acceleration,
                            protocol.max_acceleration,
                        )
                    )
                    maximum_change = protocol.max_jerk * protocol.dt
                    applied_acceleration = current_acceleration + float(
                        np.clip(
                            target_acceleration - current_acceleration,
                            -maximum_change,
                            maximum_change,
                        )
                    )
                    attempted_speed = (
                        current_speed + applied_acceleration * protocol.dt
                    )
                    negative_speed_attempts += int(
                        attempted_speed < protocol.min_speed
                    )
                    overspeed_attempts += int(
                        attempted_speed > protocol.max_speed
                    )
                    next_speed = float(
                        np.clip(
                            attempted_speed,
                            protocol.min_speed,
                            protocol.max_speed,
                        )
                    )
                    next_relative_speed = observed_lead_speed - next_speed
                    raw_next_gap = current_gap + protocol.dt * (
                        current_relative_speed + next_relative_speed
                    ) / 2.0
                    if raw_next_gap <= protocol.collision_gap:
                        collision = True
                        collision_step = step + 1
                        terminal = True
                        next_gap = protocol.collision_gap
                    else:
                        next_gap = float(raw_next_gap)

                if router is not None:
                    router_history.append(router[0].detach().cpu().numpy())
                if contribution is not None:
                    contribution_history.append(
                        contribution[0].detach().cpu().numpy()
                    )
                if scene is not None:
                    scene_history.append(scene[0].detach().cpu().numpy())
            else:
                # Collision/numerical failure is an absorbing terminal state.
                applied_acceleration = 0.0
                next_speed = 0.0
                next_gap = protocol.collision_gap
                next_relative_speed = observed_lead_speed

            # Spacing is the net bumper-to-bumper gap; vehicle length is not
            # added.  THW is defined only while the following vehicle moves.
            # A stopped following vehicle is assigned +inf THW and therefore
            # does not violate the THW threshold.  Collision is reported by
            # the independent collision metric above.
            if next_speed > protocol.thw_speed_epsilon:
                thw = next_gap / next_speed
                thw_moving_frames += 1
                minimum_predicted_thw = min(minimum_predicted_thw, thw)
                unsafe = thw < protocol.unsafe_thw_threshold
            else:
                thw = float("inf")
                unsafe = False
            unsafe_frames += int(unsafe)
            unsafe_event = unsafe_event or unsafe
            predicted_gap.append(next_gap)
            predicted_speed.append(next_speed)
            predicted_acceleration.append(applied_acceleration)

            if not terminal:
                next_frame = torch.tensor(
                    [
                        next_gap,
                        next_speed,
                        next_relative_speed,
                        observed_lead_speed,
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                next_normalized = torch.clamp(
                    (next_frame - mean_t) / std_t,
                    -protocol.normalized_state_limit,
                    protocol.normalized_state_limit,
                )
                current_window = torch.cat(
                    [current_window[:, 1:, :], next_normalized.view(1, 1, 4)],
                    dim=1,
                )
                current_gap = next_gap
                current_speed = next_speed
                current_relative_speed = next_relative_speed
                current_acceleration = applied_acceleration

    predicted_gap = np.asarray(predicted_gap)
    predicted_speed = np.asarray(predicted_speed)
    predicted_acceleration = np.asarray(predicted_acceleration)
    true_gap = future[:, 0].cpu().numpy()
    true_speed = future[:, 1].cpu().numpy()
    true_acceleration = np.diff(
        np.concatenate([[float(history_physical[-1, 1])], true_speed])
    ) / protocol.dt

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": int(seed),
        "event_id": int(event_id),
        "available_future_steps": actual_future_steps,
        "horizon_steps": horizon_steps,
        "horizon_seconds": horizon_steps * protocol.dt,
        "mae_acc": float(
            np.mean(np.abs(predicted_acceleration - true_acceleration))
        ),
        "rmse_acc": float(
            np.sqrt(np.mean((predicted_acceleration - true_acceleration) ** 2))
        ),
        "mae_spacing": float(np.mean(np.abs(predicted_gap - true_gap))),
        "rmse_spacing": float(
            np.sqrt(np.mean((predicted_gap - true_gap) ** 2))
        ),
        "mae_speed": float(np.mean(np.abs(predicted_speed - true_speed))),
        "collision": int(collision),
        "collision_step": int(collision_step),
        "unsafe_event": int(unsafe_event),
        "unsafe_frames": int(unsafe_frames),
        "unsafe_frame_rate": unsafe_frames / horizon_steps,
        "unsafe_thw_threshold_seconds": protocol.unsafe_thw_threshold,
        "thw_moving_frames": int(thw_moving_frames),
        "minimum_predicted_thw": (
            float(minimum_predicted_thw)
            if np.isfinite(minimum_predicted_thw)
            else float("nan")
        ),
        "negative_speed_attempts": int(negative_speed_attempts),
        "negative_speed_attempt_rate": (
            negative_speed_attempts / horizon_steps
        ),
        "overspeed_attempts": int(overspeed_attempts),
        "overspeed_attempt_rate": overspeed_attempts / horizon_steps,
        "numerical_failure": int(numerical_failure),
        "minimum_predicted_spacing": float(np.min(predicted_gap)),
    }

    if router_history:
        values = np.asarray(router_history)
        maximum = values.max(axis=1, keepdims=True)
        unique_top = np.isclose(values, maximum).sum(axis=1) == 1
        top_ids = np.argmax(values, axis=1)
        for index, expert in enumerate(PRIVATE_EXPERTS):
            result[f"router_mean_{expert}"] = float(values[:, index].mean())
            result[f"router_std_{expert}"] = float(values[:, index].std())
            result[f"router_top1_fraction_{expert}"] = float(
                np.mean(unique_top & (top_ids == index))
            )
    if contribution_history:
        values = np.asarray(contribution_history)
        for index, expert in enumerate(ALL_EXPERTS):
            result[f"contribution_mean_{expert}"] = float(
                values[:, index].mean()
            )
            result[f"contribution_std_{expert}"] = float(
                values[:, index].std()
            )
    if scene_history:
        values = np.asarray(scene_history)
        hard_ids = np.argmax(values, axis=1)
        for index in range(4):
            result[f"scene_soft_mean_{index + 1}"] = float(
                values[:, index].mean()
            )
            result[f"scene_hard_fraction_{index + 1}"] = float(
                np.mean(hard_ids == index)
            )
    return result


def rollout_event_batch(
    model,
    histories_normalized,
    futures_physical,
    mean,
    std,
    protocol,
    dataset_name,
    model_name,
    seed,
    event_ids,
    device_override=None,
):
    """Roll independent events in parallel without parallelizing time.

    Trajectories are padded only for tensor storage. ``valid`` excludes padded
    frames from state updates and every metric denominator. Collision and
    numerical failure remain absorbing states until each event's true final
    frame, exactly as in :func:`rollout_event`.
    """
    device = (
        torch.device(device_override)
        if device_override is not None
        else next(model.parameters()).device
    )
    history = histories_normalized.to(device, non_blocking=True).float()
    batch_size = history.shape[0]
    available_lengths = torch.tensor(
        [int(item.shape[0]) for item in futures_physical],
        device=device,
        dtype=torch.long,
    )
    horizon_lengths = available_lengths.clone()
    if protocol.horizon_steps > 0:
        horizon_lengths.clamp_(max=protocol.horizon_steps)
    if torch.any(horizon_lengths <= 0):
        raise ValueError("Every batched event must contain a future frame.")

    maximum_horizon = int(horizon_lengths.max().item())
    future = torch.zeros(
        batch_size,
        maximum_horizon,
        4,
        device=device,
        dtype=torch.float32,
    )
    for local_index, item in enumerate(futures_physical):
        length = int(horizon_lengths[local_index].item())
        future[local_index, :length] = item[:length].to(
            device, non_blocking=True
        )

    mean_t = torch.as_tensor(mean, device=device, dtype=torch.float32)
    std_t = torch.as_tensor(std, device=device, dtype=torch.float32)
    history_physical = history * std_t + mean_t
    current_window = history.clone()
    current_gap = history_physical[:, -1, 0].clone()
    current_speed = history_physical[:, -1, 1].clone()
    current_relative_speed = history_physical[:, -1, 2].clone()
    current_acceleration = (
        history_physical[:, -1, 1] - history_physical[:, -2, 1]
    ) / protocol.dt

    previous_true_speed = torch.cat(
        [history_physical[:, -1:, 1], future[:, :-1, 1]], dim=1
    )
    true_acceleration = (future[:, :, 1] - previous_true_speed) / protocol.dt

    collision = torch.zeros(batch_size, dtype=torch.bool, device=device)
    unsafe_event = torch.zeros(batch_size, dtype=torch.bool, device=device)
    numerical_failure = torch.zeros(batch_size, dtype=torch.bool, device=device)
    terminal = torch.zeros(batch_size, dtype=torch.bool, device=device)
    collision_step = torch.full(
        (batch_size,), -1, dtype=torch.long, device=device
    )
    unsafe_frames = torch.zeros(batch_size, dtype=torch.long, device=device)
    thw_moving_frames = torch.zeros(
        batch_size, dtype=torch.long, device=device
    )
    minimum_predicted_thw = torch.full(
        (batch_size,), float("inf"), device=device
    )
    negative_speed_attempts = torch.zeros(
        batch_size, dtype=torch.long, device=device
    )
    overspeed_attempts = torch.zeros(
        batch_size, dtype=torch.long, device=device
    )

    mae_acc_sum = torch.zeros(batch_size, device=device)
    squared_acc_sum = torch.zeros(batch_size, device=device)
    mae_spacing_sum = torch.zeros(batch_size, device=device)
    squared_spacing_sum = torch.zeros(batch_size, device=device)
    mae_speed_sum = torch.zeros(batch_size, device=device)
    minimum_spacing = torch.full(
        (batch_size,), float("inf"), device=device
    )

    router_sum = torch.zeros(batch_size, len(PRIVATE_EXPERTS), device=device)
    router_square_sum = torch.zeros_like(router_sum)
    router_top_counts = torch.zeros_like(router_sum)
    router_count = torch.zeros(batch_size, device=device)
    contribution_sum = torch.zeros(batch_size, len(ALL_EXPERTS), device=device)
    contribution_square_sum = torch.zeros_like(contribution_sum)
    contribution_count = torch.zeros(batch_size, device=device)
    scene_sum = torch.zeros(batch_size, 4, device=device)
    scene_hard_counts = torch.zeros_like(scene_sum)
    scene_count = torch.zeros(batch_size, device=device)

    with torch.inference_mode():
        for step in range(maximum_horizon):
            valid = step < horizon_lengths
            observed_lead_speed = future[:, step, 3]
            running = valid & ~terminal
            running_indices = torch.nonzero(
                running, as_tuple=False
            ).squeeze(1)

            applied_acceleration = torch.zeros(batch_size, device=device)
            next_speed = torch.zeros(batch_size, device=device)
            next_gap = torch.full(
                (batch_size,), protocol.collision_gap, device=device
            )
            next_relative_speed = observed_lead_speed.clone()

            if running_indices.numel() > 0:
                output, router, contribution, scene = _predict(
                    model, current_window[running_indices]
                )
                raw_acceleration = output.reshape(-1).float()

                if router is not None:
                    values = router.float()
                    router_sum[running_indices] += values
                    router_square_sum[running_indices] += values.square()
                    top_ids = torch.argmax(values, dim=1)
                    maximum = values.max(dim=1, keepdim=True).values
                    unique_top = torch.isclose(values, maximum).sum(dim=1) == 1
                    router_top_counts[running_indices] += torch.nn.functional.one_hot(
                        top_ids, num_classes=len(PRIVATE_EXPERTS)
                    ).float() * unique_top.unsqueeze(1).float()
                    router_count[running_indices] += 1.0
                if contribution is not None:
                    values = contribution.float()
                    contribution_sum[running_indices] += values
                    contribution_square_sum[running_indices] += values.square()
                    contribution_count[running_indices] += 1.0
                if scene is not None:
                    values = scene.float()
                    scene_sum[running_indices] += values
                    hard_ids = torch.argmax(values, dim=1)
                    scene_hard_counts[running_indices] += torch.nn.functional.one_hot(
                        hard_ids, num_classes=4
                    ).float()
                    scene_count[running_indices] += 1.0

                finite_positions = torch.isfinite(raw_acceleration)
                finite_indices = running_indices[finite_positions]
                failed_indices = running_indices[~finite_positions]
                if failed_indices.numel() > 0:
                    numerical_failure[failed_indices] = True
                    terminal[failed_indices] = True

                if finite_indices.numel() > 0:
                    target_acceleration = torch.clamp(
                        raw_acceleration[finite_positions],
                        protocol.min_acceleration,
                        protocol.max_acceleration,
                    )
                    maximum_change = protocol.max_jerk * protocol.dt
                    finite_acceleration = current_acceleration[finite_indices] + torch.clamp(
                        target_acceleration - current_acceleration[finite_indices],
                        -maximum_change,
                        maximum_change,
                    )
                    attempted_speed = current_speed[finite_indices] + (
                        finite_acceleration * protocol.dt
                    )
                    negative_speed_attempts[finite_indices] += (
                        attempted_speed < protocol.min_speed
                    ).long()
                    overspeed_attempts[finite_indices] += (
                        attempted_speed > protocol.max_speed
                    ).long()
                    finite_speed = torch.clamp(
                        attempted_speed,
                        protocol.min_speed,
                        protocol.max_speed,
                    )
                    finite_relative_speed = (
                        observed_lead_speed[finite_indices] - finite_speed
                    )
                    raw_next_gap = current_gap[finite_indices] + protocol.dt * (
                        current_relative_speed[finite_indices]
                        + finite_relative_speed
                    ) / 2.0
                    new_collision = raw_next_gap <= protocol.collision_gap

                    applied_acceleration[finite_indices] = finite_acceleration
                    next_speed[finite_indices] = finite_speed
                    next_relative_speed[finite_indices] = finite_relative_speed
                    next_gap[finite_indices] = torch.where(
                        new_collision,
                        torch.full_like(raw_next_gap, protocol.collision_gap),
                        raw_next_gap,
                    )

                    collision_indices = finite_indices[new_collision]
                    if collision_indices.numel() > 0:
                        collision[collision_indices] = True
                        collision_step[collision_indices] = step + 1
                        terminal[collision_indices] = True

                    survivor_indices = finite_indices[~new_collision]
                    if survivor_indices.numel() > 0:
                        next_frame = torch.stack(
                            [
                                next_gap[survivor_indices],
                                next_speed[survivor_indices],
                                next_relative_speed[survivor_indices],
                                observed_lead_speed[survivor_indices],
                            ],
                            dim=1,
                        )
                        next_normalized = torch.clamp(
                            (next_frame - mean_t) / std_t,
                            -protocol.normalized_state_limit,
                            protocol.normalized_state_limit,
                        )
                        current_window[survivor_indices] = torch.cat(
                            [
                                current_window[survivor_indices, 1:, :],
                                next_normalized.unsqueeze(1),
                            ],
                            dim=1,
                        )
                        current_gap[survivor_indices] = next_gap[
                            survivor_indices
                        ]
                        current_speed[survivor_indices] = next_speed[
                            survivor_indices
                        ]
                        current_relative_speed[survivor_indices] = (
                            next_relative_speed[survivor_indices]
                        )
                        current_acceleration[survivor_indices] = (
                            applied_acceleration[survivor_indices]
                        )

            thw_moving = valid & (next_speed > protocol.thw_speed_epsilon)
            thw = next_gap / torch.clamp(
                next_speed, min=protocol.thw_speed_epsilon
            )
            unsafe = thw_moving & (thw < protocol.unsafe_thw_threshold)
            unsafe_frames += unsafe.long()
            thw_moving_frames += thw_moving.long()
            minimum_predicted_thw = torch.where(
                thw_moving,
                torch.minimum(minimum_predicted_thw, thw),
                minimum_predicted_thw,
            )
            unsafe_event |= unsafe
            valid_float = valid.float()
            gap_error = next_gap - future[:, step, 0]
            speed_error = next_speed - future[:, step, 1]
            acceleration_error = (
                applied_acceleration - true_acceleration[:, step]
            )
            mae_acc_sum += acceleration_error.abs() * valid_float
            squared_acc_sum += acceleration_error.square() * valid_float
            mae_spacing_sum += gap_error.abs() * valid_float
            squared_spacing_sum += gap_error.square() * valid_float
            mae_speed_sum += speed_error.abs() * valid_float
            minimum_spacing = torch.where(
                valid,
                torch.minimum(minimum_spacing, next_gap),
                minimum_spacing,
            )

    lengths_float = horizon_lengths.float()
    router_denominator = torch.clamp(router_count, min=1.0).unsqueeze(1)
    router_mean = router_sum / router_denominator
    router_std = torch.sqrt(
        torch.clamp(
            router_square_sum / router_denominator - router_mean.square(),
            min=0.0,
        )
    )
    router_top_fraction = router_top_counts / router_denominator
    contribution_denominator = torch.clamp(
        contribution_count, min=1.0
    ).unsqueeze(1)
    contribution_mean = contribution_sum / contribution_denominator
    contribution_std = torch.sqrt(
        torch.clamp(
            contribution_square_sum / contribution_denominator
            - contribution_mean.square(),
            min=0.0,
        )
    )
    scene_denominator = torch.clamp(scene_count, min=1.0).unsqueeze(1)
    scene_mean = scene_sum / scene_denominator
    scene_hard_fraction = scene_hard_counts / scene_denominator

    def cpu_list(tensor):
        return tensor.detach().cpu().tolist()

    available_values = cpu_list(available_lengths)
    horizon_values = cpu_list(horizon_lengths)
    collision_values = cpu_list(collision.long())
    collision_step_values = cpu_list(collision_step)
    unsafe_event_values = cpu_list(unsafe_event.long())
    unsafe_frame_values = cpu_list(unsafe_frames)
    thw_moving_frame_values = cpu_list(thw_moving_frames)
    minimum_thw_values = cpu_list(minimum_predicted_thw)
    negative_values = cpu_list(negative_speed_attempts)
    overspeed_values = cpu_list(overspeed_attempts)
    numerical_values = cpu_list(numerical_failure.long())
    minimum_values = cpu_list(minimum_spacing)
    mae_acc_values = cpu_list(mae_acc_sum / lengths_float)
    rmse_acc_values = cpu_list(torch.sqrt(squared_acc_sum / lengths_float))
    mae_spacing_values = cpu_list(mae_spacing_sum / lengths_float)
    rmse_spacing_values = cpu_list(
        torch.sqrt(squared_spacing_sum / lengths_float)
    )
    mae_speed_values = cpu_list(mae_speed_sum / lengths_float)
    router_count_values = cpu_list(router_count)
    contribution_count_values = cpu_list(contribution_count)
    scene_count_values = cpu_list(scene_count)
    router_mean_values = cpu_list(router_mean)
    router_std_values = cpu_list(router_std)
    router_top_values = cpu_list(router_top_fraction)
    contribution_mean_values = cpu_list(contribution_mean)
    contribution_std_values = cpu_list(contribution_std)
    scene_mean_values = cpu_list(scene_mean)
    scene_hard_values = cpu_list(scene_hard_fraction)

    rows = []
    for local_index, event_id in enumerate(event_ids):
        horizon = int(horizon_values[local_index])
        row = {
            "dataset": dataset_name,
            "model": model_name,
            "seed": int(seed),
            "event_id": int(event_id),
            "available_future_steps": int(available_values[local_index]),
            "horizon_steps": horizon,
            "horizon_seconds": horizon * protocol.dt,
            "mae_acc": float(mae_acc_values[local_index]),
            "rmse_acc": float(rmse_acc_values[local_index]),
            "mae_spacing": float(mae_spacing_values[local_index]),
            "rmse_spacing": float(rmse_spacing_values[local_index]),
            "mae_speed": float(mae_speed_values[local_index]),
            "collision": int(collision_values[local_index]),
            "collision_step": int(collision_step_values[local_index]),
            "unsafe_event": int(unsafe_event_values[local_index]),
            "unsafe_frames": int(unsafe_frame_values[local_index]),
            "unsafe_frame_rate": unsafe_frame_values[local_index] / horizon,
            "unsafe_thw_threshold_seconds": protocol.unsafe_thw_threshold,
            "thw_moving_frames": int(
                thw_moving_frame_values[local_index]
            ),
            "minimum_predicted_thw": (
                float(minimum_thw_values[local_index])
                if np.isfinite(minimum_thw_values[local_index])
                else float("nan")
            ),
            "negative_speed_attempts": int(negative_values[local_index]),
            "negative_speed_attempt_rate": negative_values[local_index]
            / horizon,
            "overspeed_attempts": int(overspeed_values[local_index]),
            "overspeed_attempt_rate": overspeed_values[local_index] / horizon,
            "numerical_failure": int(numerical_values[local_index]),
            "minimum_predicted_spacing": float(minimum_values[local_index]),
        }
        if router_count_values[local_index] > 0:
            for index, expert in enumerate(PRIVATE_EXPERTS):
                row[f"router_mean_{expert}"] = float(
                    router_mean_values[local_index][index]
                )
                row[f"router_std_{expert}"] = float(
                    router_std_values[local_index][index]
                )
                row[f"router_top1_fraction_{expert}"] = float(
                    router_top_values[local_index][index]
                )
        if contribution_count_values[local_index] > 0:
            for index, expert in enumerate(ALL_EXPERTS):
                row[f"contribution_mean_{expert}"] = float(
                    contribution_mean_values[local_index][index]
                )
                row[f"contribution_std_{expert}"] = float(
                    contribution_std_values[local_index][index]
                )
        if scene_count_values[local_index] > 0:
            for index in range(4):
                row[f"scene_soft_mean_{index + 1}"] = float(
                    scene_mean_values[local_index][index]
                )
                row[f"scene_hard_fraction_{index + 1}"] = float(
                    scene_hard_values[local_index][index]
                )
        rows.append(row)
    return rows


def _physical_gpu_label(local_device_index):
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if local_device_index < len(visible):
        return visible[local_device_index]
    return str(local_device_index)


def assign_events_to_gpu_shards(event_ids, horizon_by_event, number_of_shards):
    """Deterministically assign each complete event to exactly one shard."""
    assignments = [[] for _ in range(number_of_shards)]
    estimated_loads = [0 for _ in range(number_of_shards)]
    ranked_events = sorted(
        event_ids,
        key=lambda event_id: (-horizon_by_event[event_id], event_id),
    )
    for event_id in ranked_events:
        shard_index = min(
            range(number_of_shards),
            key=lambda index: (estimated_loads[index], index),
        )
        assignments[shard_index].append(event_id)
        estimated_loads[shard_index] += horizon_by_event[event_id]
    for shard in assignments:
        shard.sort(key=lambda event_id: (horizon_by_event[event_id], event_id))
    return assignments, estimated_loads


def rollout_dataset_gpu_shards(
    args,
    model,
    dataset,
    event_ids,
    mean,
    std,
    protocol,
    dataset_name,
    model_name,
    seed,
    evaluation_batch_size,
    split_name,
    show_progress,
):
    """Assign every event permanently to one GPU for its complete rollout."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None

    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
        device_ids = [int(item) for item in model.device_ids]
    else:
        base_model = model
        device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) < 2:
        return None
    base_device = next(base_model.parameters()).device
    if base_device.type != "cuda":
        return None

    horizon_by_event = {
        event_id: min(
            int(dataset.fut_data[event_id].shape[0]),
            protocol.horizon_steps
            if protocol.horizon_steps > 0
            else int(dataset.fut_data[event_id].shape[0]),
        )
        for event_id in event_ids
    }
    assignments, estimated_loads = assign_events_to_gpu_shards(
        event_ids,
        horizon_by_event,
        len(device_ids),
    )

    devices = [torch.device("cuda", device_id) for device_id in device_ids]
    # torch.nn.parallel.replicate is the same replication primitive used by
    # DataParallel, but it is invoked once per validation/test run instead of
    # once for every autoregressive time step.
    replicas = torch.nn.parallel.replicate(
        base_model,
        devices,
        detach=True,
    )
    for replica in replicas:
        replica.eval()

    mapping_text = ", ".join(
        (
            f"cuda:{device_ids[index]}(physical "
            f"{_physical_gpu_label(device_ids[index])})="
            f"{len(assignments[index])} events/"
            f"{estimated_loads[index]} frames"
        )
        for index in range(len(device_ids))
    )
    print(f"[{split_name}] persistent event-to-GPU sharding: {mapping_text}")

    def run_shard(shard_index):
        device = devices[shard_index]
        torch.cuda.set_device(device)
        replica = replicas[shard_index]
        shard_rows = []
        shard_event_ids = assignments[shard_index]
        for start in range(0, len(shard_event_ids), evaluation_batch_size):
            batch_ids = shard_event_ids[
                start : start + evaluation_batch_size
            ]
            histories = dataset.hist_data[batch_ids]
            futures = [dataset.fut_data[event_id] for event_id in batch_ids]
            batch_rows = rollout_event_batch(
                replica,
                histories,
                futures,
                mean,
                std,
                protocol,
                dataset_name,
                model_name,
                seed,
                batch_ids,
                device_override=device,
            )
            physical_gpu = _physical_gpu_label(device_ids[shard_index])
            for row in batch_rows:
                row["evaluation_gpu_local"] = int(device_ids[shard_index])
                row["evaluation_gpu_physical"] = physical_gpu
            shard_rows.extend(batch_rows)
        torch.cuda.synchronize(device)
        return shard_rows

    rows = []
    active_shards = [
        index for index, shard in enumerate(assignments) if shard
    ]
    with ThreadPoolExecutor(max_workers=len(active_shards)) as executor:
        future_to_shard = {
            executor.submit(run_shard, shard_index): shard_index
            for shard_index in active_shards
        }
        iterator = as_completed(future_to_shard)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(future_to_shard),
                desc=(
                    f"{split_name}: {dataset_name}/{model_name}/seed={seed} "
                    "persistent GPU shards"
                ),
            )
        for completed in iterator:
            rows.extend(completed.result())

    # Keep only the original training model alive after validation.
    del replicas
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return rows


def summarize_event_dataframe(dataframe):
    if dataframe.empty:
        raise ValueError(
            "No event contains a future frame after the 20-frame "
            "initialization window. Verify the extracted event duration."
        )
    total_prediction_frames = int(dataframe["horizon_steps"].sum())
    collision_events = int(dataframe["collision"].sum())
    unsafe_events = int(dataframe["unsafe_event"].sum())
    unsafe_frames = int(dataframe["unsafe_frames"].sum())
    negative_speed_attempts = int(
        dataframe["negative_speed_attempts"].sum()
    )
    overspeed_attempts = int(dataframe["overspeed_attempts"].sum())
    return {
        "n_events": int(len(dataframe)),
        "total_prediction_frames": total_prediction_frames,
        "horizon_steps_min": int(dataframe["horizon_steps"].min()),
        "horizon_steps_mean": float(dataframe["horizon_steps"].mean()),
        "horizon_steps_median": float(dataframe["horizon_steps"].median()),
        "horizon_steps_max": int(dataframe["horizon_steps"].max()),
        "horizon_seconds_min": float(dataframe["horizon_seconds"].min()),
        "horizon_seconds_mean": float(dataframe["horizon_seconds"].mean()),
        "horizon_seconds_median": float(
            dataframe["horizon_seconds"].median()
        ),
        "horizon_seconds_max": float(dataframe["horizon_seconds"].max()),
        "mae_acc": float(dataframe["mae_acc"].mean()),
        "rmse_acc": float(dataframe["rmse_acc"].mean()),
        "mae_spacing": float(dataframe["mae_spacing"].mean()),
        "rmse_spacing": float(dataframe["rmse_spacing"].mean()),
        "mae_speed": float(dataframe["mae_speed"].mean()),
        "collision_events": collision_events,
        "collision_rate": collision_events / len(dataframe),
        "unsafe_events": unsafe_events,
        "unsafe_event_rate": unsafe_events / len(dataframe),
        "unsafe_frames": unsafe_frames,
        "unsafe_frame_rate": float(dataframe["unsafe_frame_rate"].mean()),
        "unsafe_frame_rate_micro": (
            unsafe_frames / total_prediction_frames
        ),
        "negative_speed_attempts": negative_speed_attempts,
        "negative_speed_attempt_rate": float(
            dataframe["negative_speed_attempt_rate"].mean()
        ),
        "negative_speed_attempt_rate_micro": (
            negative_speed_attempts / total_prediction_frames
        ),
        "overspeed_attempts": overspeed_attempts,
        "overspeed_attempt_rate": float(
            dataframe["overspeed_attempt_rate"].mean()
        ),
        "overspeed_attempt_rate_micro": (
            overspeed_attempts / total_prediction_frames
        ),
        "numerical_failure_rate": float(
            dataframe["numerical_failure"].mean()
        ),
    }


def evaluate_model(
    args,
    model,
    test_loader,
    mean=None,
    std=None,
    split_name="test",
    output_csv=None,
    show_progress=True,
):
    model.eval()
    protocol = protocol_from_args(args)
    dataset_name = args.get("dataset_name", "SH")
    model_name = args.get("model_name", "MoE")
    seed = int(args.get("seed", 42))
    evaluation_batch_size = max(1, int(args.get("eval_batch_size", 1)))
    rows = []
    excluded_short_events = 0
    dataset = getattr(test_loader, "dataset", None)
    can_batch = (
        evaluation_batch_size > 1
        and dataset is not None
        and hasattr(dataset, "hist_data")
        and hasattr(dataset, "fut_data")
        and isinstance(dataset.fut_data, list)
    )
    if can_batch:
        event_ids = [
            event_id
            for event_id, future in enumerate(dataset.fut_data)
            if int(future.shape[0]) > 0
        ]
        excluded_short_events = len(dataset.fut_data) - len(event_ids)
        used_gpu_sharding = False
        if bool(args.get("eval_gpu_sharding", True)):
            sharded_rows = rollout_dataset_gpu_shards(
                args,
                model,
                dataset,
                event_ids,
                mean,
                std,
                protocol,
                dataset_name,
                model_name,
                seed,
                evaluation_batch_size,
                split_name,
                show_progress,
            )
            if sharded_rows is not None:
                rows.extend(sharded_rows)
                used_gpu_sharding = True
        if not used_gpu_sharding:
            event_ids.sort(
                key=lambda event_id: min(
                    int(dataset.fut_data[event_id].shape[0]),
                    protocol.horizon_steps
                    if protocol.horizon_steps > 0
                    else int(dataset.fut_data[event_id].shape[0]),
                )
            )
            batch_starts = range(0, len(event_ids), evaluation_batch_size)
            iterator = tqdm(
                batch_starts,
                total=(len(event_ids) + evaluation_batch_size - 1)
                // evaluation_batch_size,
                desc=(
                    f"{split_name}: {dataset_name}/{model_name}/seed={seed} "
                    f"batch={evaluation_batch_size}"
                ),
                disable=not show_progress,
            )
            for start in iterator:
                batch_ids = event_ids[start : start + evaluation_batch_size]
                histories = dataset.hist_data[batch_ids]
                futures = [dataset.fut_data[event_id] for event_id in batch_ids]
                rows.extend(
                    rollout_event_batch(
                        model,
                        histories,
                        futures,
                        mean,
                        std,
                        protocol,
                        dataset_name,
                        model_name,
                        seed,
                        batch_ids,
                    )
                )
    else:
        iterator = tqdm(
            test_loader,
            desc=f"{split_name}: {dataset_name}/{model_name}/seed={seed}",
            disable=not show_progress,
        )
        for event_id, (history, future, _) in enumerate(iterator):
            row = rollout_event(
                model,
                history,
                future,
                mean,
                std,
                protocol,
                dataset_name,
                model_name,
                seed,
                event_id,
            )
            if row is None:
                excluded_short_events += 1
            else:
                rows.append(row)
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe = dataframe.sort_values("event_id").reset_index(drop=True)
    summary = summarize_event_dataframe(dataframe)
    summary["excluded_short_events"] = excluded_short_events
    summary["split"] = split_name
    summary["unsafe_thw_threshold_seconds"] = protocol.unsafe_thw_threshold
    summary["thw_speed_epsilon_mps"] = protocol.thw_speed_epsilon
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
    print(
        f"[{split_name}] {dataset_name}/{model_name}/seed={seed}: "
        f"MAE_acc={summary['mae_acc']:.4f}, "
        f"MAE_spacing={summary['mae_spacing']:.4f}, "
        f"RMSE_spacing={summary['rmse_spacing']:.4f}, "
        f"collision={100 * summary['collision_rate']:.2f}%, "
        f"unsafe={100 * summary['unsafe_event_rate']:.2f}%"
    )
    return summary, dataframe
