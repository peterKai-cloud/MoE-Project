"""Reproducibility and physical-parameter audit helpers."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


FINAL_SEEDS = [17, 42, 73, 101, 137, 211, 307, 401, 509, 607]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without silently hiding nondeterminism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


# These are finite optimization guardrails. The manuscript should report their
# units and justify them using the dataset range and/or car-following literature.
PHYSICAL_PARAMETER_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "GM_Expert": {
        "C": (1e-3, 10.0),
        "m": (-2.0, 2.0),
        "l": (0.0, 5.0),
    },
    "Gipps_Expert": {
        "a_n": (0.05, 8.0),
        "b_n": (0.05, 10.0),
        "b_hat": (0.05, 10.0),
        "tau": (0.1, 5.0),
        "v_des": (1.0, 60.0),
    },
    "IDM_Expert": {
        "a_max": (0.05, 8.0),
        "v_des": (1.0, 60.0),
        "beta": (1.0, 8.0),
        "s_jam": (0.1, 20.0),
        "T_headway": (0.1, 5.0),
        "a_comf": (0.05, 8.0),
    },
    "FVD_Expert": {
        "alpha": (1e-4, 5.0),
        "lambda_0": (1e-4, 5.0),
        "s_c": (0.1, 150.0),
        "v0": (1.0, 60.0),
        "b": (0.1, 100.0),
        "beta": (-10.0, 10.0),
    },
    "Wiedemann_Expert": {
        "th_dv": (0.0, 20.0),
        "th_gap": (0.1, 150.0),
    },
    "NETSIM_Expert": {
        "T": (0.1, 5.0),
        "a_max_accel": (0.05, 8.0),
        "a_min_brake": (0.05, 10.0),
        "b_max_brake": (0.05, 10.0),
    },
    "Mod_NETSIM_Expert": {
        "T": (0.1, 5.0),
        "a_min_brake": (0.05, 10.0),
        "a_max_brake": (0.05, 10.0),
        "b_max_brake": (0.05, 10.0),
        "v_max": (1.0, 60.0),
    },
    "Fritzsche_Expert": {
        "sdv": (0.1, 100.0),
        "cldv": (1e-4, 20.0),
        "d_safe": (0.1, 100.0),
        "a_emerg": (0.05, 10.0),
        "a_clos_k": (1e-4, 10.0),
        "v_des": (1.0, 60.0),
    },
}

EXPERT_LABELS = {
    "GM_Expert": "GM",
    "Gipps_Expert": "Gipps",
    "IDM_Expert": "IDM",
    "FVD_Expert": "FVD",
    "Wiedemann_Expert": "Wiedemann",
    "NETSIM_Expert": "NETSIM",
    "Mod_NETSIM_Expert": "Mod_NETSIM",
    "Fritzsche_Expert": "Fritzsche",
}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def iter_physical_experts(
    model: torch.nn.Module,
) -> Iterable[Tuple[str, torch.nn.Module]]:
    base = unwrap_model(model)
    seen = set()

    if hasattr(base, "shared_idm"):
        seen.add(id(base.shared_idm))
        yield "IDM_shared", base.shared_idm

    if hasattr(base, "private_experts"):
        for module in base.private_experts:
            class_name = module.__class__.__name__
            if class_name in PHYSICAL_PARAMETER_BOUNDS and id(module) not in seen:
                seen.add(id(module))
                yield EXPERT_LABELS[class_name], module

    if hasattr(base, "expert"):
        module = base.expert
        class_name = module.__class__.__name__
        if class_name in PHYSICAL_PARAMETER_BOUNDS and id(module) not in seen:
            yield EXPERT_LABELS[class_name], module


@torch.no_grad()
def project_physical_parameters(model: torch.nn.Module) -> None:
    """Project scalar physical parameters into the declared finite bounds."""
    for _, expert in iter_physical_experts(model):
        bounds = PHYSICAL_PARAMETER_BOUNDS[expert.__class__.__name__]
        for name, parameter in expert.named_parameters(recurse=False):
            if name in bounds and parameter.numel() == 1:
                lower, upper = bounds[name]
                parameter.clamp_(lower, upper)


def append_parameter_snapshot(
    model: torch.nn.Module,
    csv_path: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    epoch: int,
    is_best: bool = False,
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "seed",
                    "epoch",
                    "is_best",
                    "expert",
                    "parameter",
                    "value",
                    "lower_bound",
                    "upper_bound",
                    "at_lower_bound",
                    "at_upper_bound",
                ]
            )

        for expert_name, expert in iter_physical_experts(model):
            bounds = PHYSICAL_PARAMETER_BOUNDS[expert.__class__.__name__]
            for parameter_name, parameter in expert.named_parameters(recurse=False):
                if parameter_name not in bounds or parameter.numel() != 1:
                    continue
                lower, upper = bounds[parameter_name]
                value = float(parameter.detach().cpu().item())
                tolerance = 1e-6 * max(1.0, abs(upper - lower))
                writer.writerow(
                    [
                        dataset,
                        model_name,
                        seed,
                        epoch,
                        int(is_best),
                        expert_name,
                        parameter_name,
                        value,
                        lower,
                        upper,
                        int(abs(value - lower) <= tolerance),
                        int(abs(value - upper) <= tolerance),
                    ]
                )


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def serializable_stats(stats: dict) -> dict:
    return {
        "mean": np.asarray(stats["mean"], dtype=np.float32).tolist(),
        "std": np.asarray(stats["std"], dtype=np.float32).tolist(),
    }


