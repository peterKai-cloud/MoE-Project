"""Command-line entry point for training, evaluation, analysis, and figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_HYPERPARAMETERS = {
    "scene_hidden_dim": 128,
    "num_scenes": 4,
    "router_hidden_dim": 128,
    "cnn_out_channels": 32,
    "lstm_hidden_dim": 128,
    "lstm_layers": 3,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Four-scene heterogeneous MoE car-following framework"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "evaluate",
            "visualize",
            "analyze",
            "tune",
            "repeated",
            "statistics",
        ],
        required=True,
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--dataset_name", default="SH")
    parser.add_argument("--model_name", default="MoE")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--input_path", default="")
    parser.add_argument("--run_dir", default="./runs")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--event_metrics_csv", default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="17,42,73,101,137,211,307,401,509,607")
    parser.add_argument("--models", default="MoE,Mamba,Transformer")
    parser.add_argument("--datasets", default="SH,NGSIM")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--standalone_lr", type=float, default=1e-4)
    parser.add_argument("--physical_baseline_lr", type=float, default=1e-3)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--train_rollout_steps", type=int, default=5)
    parser.add_argument(
        "--test_horizon_steps",
        type=int,
        default=0,
        help=(
            "Closed-loop horizon cap. Use 0 (default) to roll every event "
            "from its 20-frame initialization to its actual final frame."
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help=(
            "Number of independent variable-length events advanced in "
            "parallel during closed-loop validation/test. Time remains "
            "strictly autoregressive within every event."
        ),
    )
    evaluation_sharding_group = parser.add_mutually_exclusive_group()
    evaluation_sharding_group.add_argument(
        "--eval_gpu_sharding",
        dest="eval_gpu_sharding",
        action="store_true",
        help=(
            "Keep each event on one GPU for its complete closed-loop "
            "trajectory (default when multiple GPUs are visible)."
        ),
    )
    evaluation_sharding_group.add_argument(
        "--no-eval_gpu_sharding",
        dest="eval_gpu_sharding",
        action="store_false",
        help="Use the legacy per-time-step DataParallel evaluation path.",
    )
    parser.set_defaults(eval_gpu_sharding=True)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--parse_workers", type=int, default=0)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help=(
            "Stop a training run when its validation-selected checkpoint has "
            "not improved for this many epochs; 0 disables early stopping."
        ),
    )
    parser.add_argument("--input_noise_std", type=float, default=0.02)
    parser.add_argument("--minimum_scene_usage", type=float, default=0.01)
    parser.add_argument("--scene_occupancy_weight", type=float, default=10.0)
    # argparse.BooleanOptionalAction was added in Python 3.9.  Define the
    # equivalent pair explicitly so the cloud environment can also use
    # Python 3.8 and earlier argparse versions.
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable automatic mixed-precision training (default).",
    )
    amp_group.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed-precision training.",
    )
    parser.set_defaults(amp=True)
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse an existing best_validation.pth in repeated mode.",
    )

    parser.add_argument("--min_acceleration", type=float, default=-4.0)
    parser.add_argument("--max_acceleration", type=float, default=3.0)
    parser.add_argument("--max_jerk", type=float, default=5.0)
    parser.add_argument("--min_speed", type=float, default=0.0)
    parser.add_argument("--max_speed", type=float, default=55.0)
    parser.add_argument("--collision_gap", type=float, default=0.0)
    parser.add_argument(
        "--unsafe_thw_threshold",
        type=float,
        default=2.0,
        help=(
            "A frame is unsafe when net spacing / following speed is below "
            "this THW threshold in seconds."
        ),
    )
    parser.add_argument(
        "--thw_speed_epsilon",
        type=float,
        default=1e-3,
        help=(
            "Following speeds at or below this value are treated as stopped "
            "and assigned infinite THW."
        ),
    )

    parser.add_argument("--optuna_trials", type=int, default=30)
    parser.add_argument("--tune_epochs", type=int, default=50)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    return parser


def to_dict(namespace):
    values = vars(namespace).copy()
    values.update(DEFAULT_HYPERPARAMETERS)
    values["num_scenes"] = 4
    return values


def default_checkpoint_path(args):
    return (
        Path(args["run_dir"])
        / args["dataset_name"]
        / args["model_name"]
        / f"seed_{args['seed']}"
        / "best_validation.pth"
    )


def train_once(args):
    from data_loader import get_dataloader
    from model_train import train_model

    train_args = dict(args)
    train_args["mode"] = "train"
    train_loader, val_loader, _, mean, std = get_dataloader(
        train_args,
        batch_size=args["batch_size"],
        num_workers=args["parse_workers"],
    )
    result = train_model(
        train_args,
        train_loader,
        val_loader,
        None,
        {"mean": mean, "std": std},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def evaluate_once(args, split_name="test"):
    from data_loader import get_dataloader
    from evaluation import evaluate_model
    from model_train import load_model_from_checkpoint

    checkpoint_path = Path(args["model_path"]) if args["model_path"] else default_checkpoint_path(args)
    model, stats, checkpoint = load_model_from_checkpoint(checkpoint_path)
    evaluation_args = dict(args)
    evaluation_args["mode"] = "evaluate"
    evaluation_args["model_name"] = checkpoint.get("model_name", args["model_name"])
    evaluation_args["seed"] = int(checkpoint.get("seed", args["seed"]))
    _, _, test_loader, _, _ = get_dataloader(
        evaluation_args,
        batch_size=1,
        num_workers=args["parse_workers"],
        stats=stats,
    )
    output = (
        Path(args["results_dir"])
        / args["dataset_name"]
        / evaluation_args["model_name"]
        / f"seed_{evaluation_args['seed']}_event_metrics.csv"
    )
    summary, _ = evaluate_model(
        evaluation_args,
        model,
        test_loader,
        mean=stats["mean"],
        std=stats["std"],
        split_name=split_name,
        output_csv=output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def tune_with_validation(args):
    from data_loader import compute_stats, dataset_paths, get_dataloader
    from model_train import train_model

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install optuna before using --mode tune") from exc

    paths = dataset_paths(args)
    mean, std = compute_stats(paths["train"], data_ratio=args["data_ratio"])
    fixed_stats = {"mean": mean, "std": std}
    tuning_root = Path(args["run_dir"]) / "tuning" / args["dataset_name"]

    def objective(trial):
        trial_args = dict(args)
        trial_args.update(
            {
                "mode": "tune",
                "epochs": args["tune_epochs"],
                "run_dir": str(tuning_root / f"trial_{trial.number}"),
                "scene_hidden_dim": trial.suggest_categorical(
                    "scene_hidden_dim", [64, 128, 256]
                ),
                "router_hidden_dim": trial.suggest_categorical(
                    "router_hidden_dim", [128, 256, 512]
                ),
                "cnn_out_channels": trial.suggest_categorical(
                    "cnn_out_channels", [32, 64, 128]
                ),
                "lstm_hidden_dim": trial.suggest_categorical(
                    "lstm_hidden_dim", [64, 128, 256]
                ),
                "lstm_layers": trial.suggest_int("lstm_layers", 2, 4),
                "num_scenes": 4,
            }
        )
        train_loader, val_loader, _, _, _ = get_dataloader(
            trial_args,
            batch_size=trial_args["batch_size"],
            stats=fixed_stats,
        )
        result = train_model(
            trial_args, train_loader, val_loader, None, fixed_stats
        )
        return result["best_validation_rmse"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args["seed"]),
    )
    study.optimize(objective, n_trials=args["optuna_trials"])
    tuning_root.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(tuning_root / "trials.csv", index=False)
    with (tuning_root / "best_hyperparameters.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(study.best_params, file, ensure_ascii=False, indent=2)
    print("Best validation RMSE:", study.best_value)
    print("Best hyperparameters:", study.best_params)


def main():
    parser = build_parser()
    namespace = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = namespace.gpu
    args = to_dict(namespace)
    from experiment_utils import seed_everything

    seed_everything(args["seed"])

    if args["mode"] == "train":
        train_once(args)
    elif args["mode"] == "evaluate":
        evaluate_once(args)
    elif args["mode"] == "visualize":
        from data_loader import get_dataloader
        from model_train import load_model_from_checkpoint

        checkpoint_path = Path(args["model_path"]) if args["model_path"] else default_checkpoint_path(args)
        model, stats, checkpoint = load_model_from_checkpoint(checkpoint_path)
        visual_args = dict(args)
        visual_args["mode"] = "visualize"
        visual_args["model_name"] = checkpoint.get("model_name", "MoE")
        _, _, test_loader, _, _ = get_dataloader(
            visual_args, batch_size=1, stats=stats
        )
        from visualize_frame import VisualizationPlot
        import matplotlib.pyplot as plt

        VisualizationPlot(test_loader.dataset, model, stats)
        plt.show()
    elif args["mode"] == "analyze":
        from analysis_per_vehicle import analyze_dataset_per_vehicle

        analyze_dataset_per_vehicle(args)
    elif args["mode"] == "tune":
        tune_with_validation(args)
    elif args["mode"] == "repeated":
        from run_repeated_experiments import run_repeated_experiments

        run_repeated_experiments(args)
    elif args["mode"] == "statistics":
        from statistical_analysis import analyze_experiment

        if not args["event_metrics_csv"]:
            raise ValueError("--event_metrics_csv is required for statistics mode")
        analyze_experiment(
            args["event_metrics_csv"],
            args["results_dir"],
            n_bootstrap=args["bootstrap_samples"],
        )


if __name__ == "__main__":
    main()
