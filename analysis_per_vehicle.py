"""Per-event analysis using exactly the same protocol as test evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from data_loader import FastTensorDataLoader, NpyDataset, dataset_paths
from evaluation import evaluate_model
from model_train import load_model_from_checkpoint


def analyze_dataset_per_vehicle(args):
    args = dict(args) if isinstance(args, dict) else vars(args)
    checkpoint_path = args.get("model_path")
    if not checkpoint_path:
        checkpoint_path = (
            Path(args.get("run_dir", "./runs"))
            / args.get("dataset_name", "SH")
            / args.get("model_name", "MoE")
            / f"seed_{int(args.get('seed', 42))}"
            / "best_validation.pth"
        )

    model, stats, checkpoint = load_model_from_checkpoint(checkpoint_path)
    args["model_name"] = checkpoint.get("model_name", args.get("model_name", "MoE"))
    args["seed"] = int(checkpoint.get("seed", args.get("seed", 42)))
    args["mode"] = "analyze"

    input_path = args.get("input_path")
    if input_path:
        input_path = Path(input_path)
        if input_path.is_dir():
            input_path = input_path / f"test_{args.get('dataset_name', 'SH')}.npy"
    else:
        input_path = Path(dataset_paths(args)["test"])

    dataset = NpyDataset(
        input_path,
        mean=stats["mean"],
        std=stats["std"],
        mode="test",
        history_len=int(args.get("history_len", 20)),
        rollout_steps=int(args.get("train_rollout_steps", 5)),
        data_ratio=float(args.get("data_ratio", 1.0)),
        parse_workers=int(args.get("parse_workers", 0)),
    )
    loader = FastTensorDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        seed=args["seed"],
    )
    output_path = (
        Path(args.get("results_dir", "./results"))
        / args.get("dataset_name", "SH")
        / args["model_name"]
        / f"seed_{args['seed']}_per_vehicle.csv"
    )
    summary, dataframe = evaluate_model(
        args,
        model,
        loader,
        mean=stats["mean"],
        std=stats["std"],
        split_name="per_vehicle_analysis",
        output_csv=output_path,
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Per-event results: {output_path}")
    print(f"Summary: {summary_path}")
    return dataframe


if __name__ == "__main__":
    raise SystemExit(
        "Run this module through main.py --mode analyze so that all protocol "
        "arguments are recorded consistently."
    )


