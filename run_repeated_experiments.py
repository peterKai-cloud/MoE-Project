"""Run validation-selected multi-seed experiments and aggregate event results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from data_loader import compute_stats, dataset_paths, get_dataloader
from evaluation import evaluate_model
from experiment_utils import write_json
from model_train import (
    MOE_MODEL_NAMES,
    load_model_from_checkpoint,
    train_model,
)
from statistical_analysis import analyze_experiment


CORE_MODELS = [
    "MoE",
    "Mamba",
    "Transformer",
    "GM",
    "Gipps",
    "IDM",
    "FVD",
    "Wiedemann",
    "NETSIM",
    "Mod_NETSIM",
    "Fritzsche",
]
ABLATION_MODELS = list(MOE_MODEL_NAMES[1:])
ALL_MODELS = CORE_MODELS + ABLATION_MODELS


def _parse_integer_list(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_string_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(values) == 1:
        selector = values[0].lower()
        if selector == "all":
            # Preserve the original meaning of --models all: proposed model
            # plus standalone baselines, without automatically adding costly
            # ablation runs.
            return list(CORE_MODELS)
        if selector == "ablations":
            return list(ABLATION_MODELS)
        if selector == "all_with_ablations":
            return list(ALL_MODELS)
    return values


def run_repeated_experiments(args):
    """
    For every dataset/model/seed:
      1. train for 200 epochs;
      2. select the checkpoint only on validation RMSE (collision is tie-breaker);
      3. evaluate the selected checkpoint on test exactly once;
      4. save one row per test event and all expert diagnostics.
    """
    args = dict(args)
    seeds = _parse_integer_list(args.get("seeds", "17,42,73,101,137,211,307,401,509,607"))
    models = _parse_string_list(args.get("models", "MoE,Mamba,Transformer"))
    datasets = _parse_string_list(args.get("datasets", "SH,NGSIM"))
    unknown_models = set(models).difference(ALL_MODELS)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    results_root = Path(args.get("results_dir", "./results"))
    results_root.mkdir(parents=True, exist_ok=True)
    write_json(
        results_root / "repeated_experiment_config.json",
        {**args, "seeds": seeds, "models": models, "datasets": datasets},
    )
    all_event_frames = []
    run_summaries = []
    all_parameter_frames = []

    for dataset_name in datasets:
        dataset_args = dict(args)
        dataset_args["dataset_name"] = dataset_name
        train_path = dataset_paths(dataset_args)["train"]
        mean, std = compute_stats(train_path, data_ratio=float(args.get("data_ratio", 1.0)))
        stats = {"mean": mean, "std": std}

        for model_name in models:
            for seed in seeds:
                run_args = dict(dataset_args)
                run_args.update(
                    {
                        "mode": "train",
                        "model_name": model_name,
                        "seed": seed,
                        "num_scenes": 4,
                    }
                )
                print("=" * 80)
                print(
                    f"Training dataset={dataset_name}, model={model_name}, seed={seed}"
                )
                expected_run_dir = (
                    Path(run_args.get("run_dir", "./runs"))
                    / dataset_name
                    / model_name
                    / f"seed_{seed}"
                )
                expected_checkpoint = expected_run_dir / "best_validation.pth"
                if bool(run_args.get("skip_existing", False)) and expected_checkpoint.exists():
                    _, _, existing_checkpoint = load_model_from_checkpoint(
                        expected_checkpoint
                    )
                    existing_validation = existing_checkpoint.get(
                        "validation_summary", {}
                    )
                    training_result = {
                        "best_checkpoint": str(expected_checkpoint),
                        "last_checkpoint": str(expected_run_dir / "last.pth"),
                        "best_epoch": int(existing_checkpoint.get("epoch", 0)),
                        "best_validation_rmse": float(
                            existing_validation.get("rmse_spacing", float("nan"))
                        ),
                        "best_validation_collision_rate": float(
                            existing_validation.get("collision_rate", float("nan"))
                        ),
                        "run_dir": str(expected_run_dir),
                    }
                    print(f"Reusing existing checkpoint: {expected_checkpoint}")
                else:
                    train_loader, validation_loader, _, _, _ = get_dataloader(
                        run_args,
                        batch_size=int(run_args.get("batch_size", 2048)),
                        num_workers=int(run_args.get("parse_workers", 0)),
                        stats=stats,
                    )
                    training_result = train_model(
                        run_args,
                        train_loader,
                        validation_loader,
                        None,
                        stats,
                    )
                parameter_path = (
                    Path(training_result["run_dir"])
                    / "expert_parameters_by_epoch.csv"
                )
                if parameter_path.exists():
                    all_parameter_frames.append(pd.read_csv(parameter_path))

                model, checkpoint_stats, checkpoint = load_model_from_checkpoint(
                    training_result["best_checkpoint"]
                )
                test_args = dict(run_args)
                test_args["mode"] = "evaluate"
                _, _, test_loader, _, _ = get_dataloader(
                    test_args,
                    batch_size=1,
                    num_workers=int(run_args.get("parse_workers", 0)),
                    stats=checkpoint_stats,
                )
                event_path = (
                    results_root
                    / dataset_name
                    / model_name
                    / f"seed_{seed}_event_metrics.csv"
                )
                summary, event_frame = evaluate_model(
                    test_args,
                    model,
                    test_loader,
                    mean=checkpoint_stats["mean"],
                    std=checkpoint_stats["std"],
                    split_name="test",
                    output_csv=event_path,
                )
                summary.update(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "best_epoch": training_result["best_epoch"],
                        "best_validation_rmse": training_result[
                            "best_validation_rmse"
                        ],
                        "epochs_completed": training_result.get(
                            "epochs_completed", float("nan")
                        ),
                        "stopped_early": training_result.get(
                            "stopped_early", ""
                        ),
                        "early_stopping_patience": training_result.get(
                            "early_stopping_patience",
                            run_args.get("early_stopping_patience", 0),
                        ),
                        "checkpoint": training_result["best_checkpoint"],
                    }
                )
                run_summaries.append(summary)
                all_event_frames.append(event_frame)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    combined_events = pd.concat(all_event_frames, ignore_index=True)
    combined_path = results_root / "all_test_event_metrics.csv"
    combined_events.to_csv(combined_path, index=False)
    pd.DataFrame(run_summaries).to_csv(
        results_root / "all_run_summaries.csv", index=False
    )
    if all_parameter_frames:
        all_parameters = pd.concat(all_parameter_frames, ignore_index=True)
        all_parameters.to_csv(
            results_root / "all_expert_parameters_by_epoch.csv", index=False
        )
        best_parameters = (
            all_parameters[all_parameters["is_best"] == 1]
            .sort_values("epoch")
            .groupby(
                ["dataset", "model", "seed", "expert", "parameter"],
                as_index=False,
            )
            .tail(1)
        )
        best_parameters.to_csv(
            results_root / "all_best_expert_parameters.csv", index=False
        )
        parameter_summary = (
            best_parameters.groupby(
                [
                    "dataset",
                    "model",
                    "expert",
                    "parameter",
                    "lower_bound",
                    "upper_bound",
                ]
            )["value"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
        parameter_summary.to_csv(
            results_root / "expert_parameter_summary_across_seeds.csv",
            index=False,
        )

    statistical_baselines = [
        model for model in models if model != "MoE"
    ]
    if "MoE" in models and statistical_baselines:
        analyze_experiment(
            combined_path,
            results_root / "statistics",
            proposed_model="MoE",
            baselines=tuple(statistical_baselines),
            n_bootstrap=int(args.get("bootstrap_samples", 10000)),
        )
    print(f"Combined event metrics: {combined_path}")
    return combined_events


if __name__ == "__main__":
    raise SystemExit(
        "Use main.py --mode repeated so every command-line setting is recorded."
    )
