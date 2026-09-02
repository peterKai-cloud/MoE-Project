"""Deterministic data preparation for car-following experiments."""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_event_shape(event) -> Optional[np.ndarray]:
    """Convert a stored trajectory into [time, 4] float32 form."""
    event = np.asarray(event, dtype=object if np.asarray(event).dtype == object else None)

    if event.ndim == 1 and event.shape[0] == 4:
        try:
            event = np.vstack(event).astype(np.float32).T
        except (TypeError, ValueError):
            return None
    elif event.ndim == 2 and event.shape[0] == 4 and event.shape[1] != 4:
        event = event.T

    if event.ndim != 2 or event.shape[1] != 4:
        return None

    event = np.asarray(event, dtype=np.float32)
    if not np.isfinite(event).all():
        return None
    return event


def parse_trajectory(task):
    event, history_len, rollout_steps, mode = task
    event = normalize_event_shape(event)
    if event is None:
        return [], [], []

    total_steps = len(event)
    if mode == "test":
        if total_steps <= history_len:
            return [], [], []
        return [event[:history_len]], [], [event[history_len:]]

    if total_steps < history_len + rollout_steps:
        return [], [], []

    ego_speed = event[:, 1]
    acceleration = np.zeros_like(ego_speed)
    acceleration[:-1] = np.diff(ego_speed) / 0.1
    acceleration[-1] = acceleration[-2] if total_steps > 1 else 0.0

    histories, labels, futures = [], [], []
    final_start = total_steps - history_len - rollout_steps + 1

    for start in range(final_start):
        future_start = start + history_len
        histories.append(event[start:future_start])
        labels.append(
            acceleration[
                future_start - 1 : future_start - 1 + rollout_steps
            ]
        )
        futures.append(event[future_start : future_start + rollout_steps])

    return histories, labels, futures


class NpyDataset(Dataset):
    def __init__(
        self,
        data_path,
        mean=None,
        std=None,
        mode="train",
        history_len=20,
        rollout_steps=5,
        data_ratio=1.0,
        parse_workers=0,
    ):
        self.mode = mode
        self.history_len = int(history_len)
        self.rollout_steps = int(rollout_steps)
        self.mean = np.asarray(mean, dtype=np.float32) if mean is not None else None
        self.std = np.asarray(std, dtype=np.float32) if std is not None else None

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        raw_data = np.load(data_path, allow_pickle=True)
        if 0.0 < data_ratio < 1.0:
            keep = max(1, int(len(raw_data) * data_ratio))
            raw_data = raw_data[:keep]

        tasks = [
            (event, self.history_len, self.rollout_steps, self.mode)
            for event in raw_data
        ]

        if parse_workers and parse_workers > 0:
            workers = min(int(parse_workers), mp.cpu_count())
            with mp.Pool(processes=workers) as pool:
                parsed = pool.map(parse_trajectory, tasks)
        else:
            parsed = [parse_trajectory(task) for task in tasks]

        history_list, label_list, future_list = [], [], []
        for histories, labels, futures in parsed:
            history_list.extend(histories)
            label_list.extend(labels)
            future_list.extend(futures)

        if not history_list:
            raise ValueError(f"No valid trajectories were parsed from {data_path}")

        self.hist_data = torch.from_numpy(np.stack(history_list)).float()
        if self.mean is not None:
            mean_t = torch.from_numpy(self.mean)
            std_t = torch.from_numpy(self.std)
            self.hist_data = (self.hist_data - mean_t) / std_t

        if self.mode == "test":
            self.fut_data = [torch.from_numpy(item).float() for item in future_list]
            self.label_data = None
        else:
            self.label_data = torch.from_numpy(np.stack(label_list)).float()
            self.fut_data = torch.from_numpy(np.stack(future_list)).float()
            if torch.cuda.is_available():
                self.hist_data = self.hist_data.pin_memory()
                self.label_data = self.label_data.pin_memory()
                self.fut_data = self.fut_data.pin_memory()

        self.num_samples = len(self.hist_data)
        print(
            f"[Dataset] {mode}: {self.num_samples} samples from {data_path}; "
            f"history={self.history_len}, rollout={self.rollout_steps}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.mode == "test":
            return self.hist_data[index], self.fut_data[index], torch.zeros(1)
        return self.hist_data[index], self.label_data[index], self.fut_data[index]


class FastTensorDataLoader:
    """In-memory loader with deterministic epoch-dependent shuffling."""

    def __init__(self, dataset, batch_size=32, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.n_samples = len(dataset)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            self.indices = torch.randperm(self.n_samples, generator=generator)
        else:
            self.indices = torch.arange(self.n_samples)
        self.epoch += 1
        self.position = 0
        return self

    def __next__(self):
        if self.position >= self.n_samples:
            raise StopIteration

        indices = self.indices[self.position : self.position + self.batch_size]
        self.position += self.batch_size

        if self.dataset.mode == "test":
            # Evaluation uses batch_size=1 because event horizons are variable.
            index = int(indices[0])
            return (
                self.dataset.hist_data[index].unsqueeze(0),
                self.dataset.fut_data[index].unsqueeze(0),
                torch.zeros(1),
            )

        return (
            self.dataset.hist_data[indices],
            self.dataset.label_data[indices],
            self.dataset.fut_data[indices],
        )

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def compute_stats(npy_path, data_ratio=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact train-set moments without random subsampling."""
    raw_data = np.load(npy_path, allow_pickle=True)
    if 0.0 < data_ratio < 1.0:
        keep = max(1, int(len(raw_data) * data_ratio))
        raw_data = raw_data[:keep]

    count = 0
    sum_features = np.zeros(4, dtype=np.float64)
    sum_squares = np.zeros(4, dtype=np.float64)

    for raw_event in raw_data:
        event = normalize_event_shape(raw_event)
        if event is None:
            continue
        count += len(event)
        sum_features += event.sum(axis=0, dtype=np.float64)
        sum_squares += np.square(event, dtype=np.float64).sum(axis=0)

    if count == 0:
        raise ValueError(f"No valid data available for statistics: {npy_path}")

    mean = sum_features / count
    variance = np.maximum(sum_squares / count - mean**2, 0.0)
    std = np.sqrt(variance)
    std[std < 1e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def dataset_paths(args):
    base_dir = args.get("data_dir", "./data")
    dataset_name = args.get("dataset_name", "SH")
    return {
        split: os.path.join(base_dir, f"{split}_{dataset_name}.npy")
        for split in ("train", "val", "test")
    }


def get_dataloader(args, batch_size=32, num_workers=0, stats=None):
    paths = dataset_paths(args)
    history_len = int(args.get("history_len", 20))
    rollout_steps = int(args.get("train_rollout_steps", 5))
    data_ratio = float(args.get("data_ratio", 1.0))
    seed = int(args.get("seed", 42))
    parse_workers = int(args.get("parse_workers", num_workers))

    if stats is None:
        mean, std = compute_stats(paths["train"], data_ratio=data_ratio)
    else:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)

    train_loader = val_loader = test_loader = None
    mode = args.get("mode", "train")

    if mode in {"train", "tune"}:
        train_dataset = NpyDataset(
            paths["train"],
            mean,
            std,
            mode="train",
            history_len=history_len,
            rollout_steps=rollout_steps,
            data_ratio=data_ratio,
            parse_workers=parse_workers,
        )
        validation_dataset = NpyDataset(
            paths["val"],
            mean,
            std,
            mode="test",
            history_len=history_len,
            rollout_steps=rollout_steps,
            data_ratio=data_ratio,
            parse_workers=parse_workers,
        )
        train_loader = FastTensorDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        val_loader = FastTensorDataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            seed=seed,
        )

    if mode in {"evaluate", "visualize", "analyze", "repeated"}:
        test_dataset = NpyDataset(
            paths["test"],
            mean,
            std,
            mode="test",
            history_len=history_len,
            rollout_steps=rollout_steps,
            data_ratio=data_ratio,
            parse_workers=parse_workers,
        )
        test_loader = FastTensorDataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            seed=seed,
        )

    return train_loader, val_loader, test_loader, mean, std


