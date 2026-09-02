"""Seed-aware uncertainty quantification and paired statistical tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_METRICS = [
    "mae_acc",
    "rmse_acc",
    "mae_spacing",
    "rmse_spacing",
    "mae_speed",
    "collision",
    "unsafe_event",
    "unsafe_frame_rate",
]


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    previous = 0.0
    number = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (number - rank) * p_values[index])
        value = max(value, previous)
        adjusted[index] = value
        previous = value
    return adjusted


def _hierarchical_absolute_ci(
    dataframe, model_name, metric, rng, n_bootstrap
):
    subset = dataframe[dataframe["model"] == model_name]
    seeds = subset["seed"].unique()
    bootstrap_values = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_values = []
        for seed in sampled_seeds:
            values = subset.loc[subset["seed"] == seed, metric].to_numpy()
            sampled_values = rng.choice(values, size=len(values), replace=True)
            seed_values.append(sampled_values.mean())
        bootstrap_values[bootstrap_index] = np.mean(seed_values)
    return np.percentile(bootstrap_values, [2.5, 97.5])


def _paired_comparison(
    dataframe,
    proposed_model,
    baseline_model,
    metric,
    rng,
    n_bootstrap,
):
    subset = dataframe[
        dataframe["model"].isin([proposed_model, baseline_model])
    ]
    paired = subset.pivot_table(
        index=["seed", "event_id"],
        columns="model",
        values=metric,
        aggfunc="first",
    )
    required = {proposed_model, baseline_model}
    if not required.issubset(paired.columns):
        raise ValueError(
            f"Missing paired results for {proposed_model} and {baseline_model}"
        )
    paired = paired.dropna(subset=[proposed_model, baseline_model])
    seeds = paired.index.get_level_values("seed").unique()
    seed_differences = []
    for seed in seeds:
        table = paired.xs(seed, level="seed")
        seed_differences.append(
            (table[baseline_model] - table[proposed_model]).mean()
        )
    seed_differences = np.asarray(seed_differences, dtype=float)
    observed_difference = float(seed_differences.mean())
    baseline_mean = float(paired[baseline_model].mean())
    relative_improvement = (
        100.0 * observed_difference / baseline_mean
        if baseline_mean != 0.0
        else np.nan
    )

    bootstrap_values = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_seed_differences = []
        for seed in sampled_seeds:
            table = paired.xs(seed, level="seed")
            sampled_indices = rng.integers(0, len(table), size=len(table))
            sampled = table.iloc[sampled_indices]
            sampled_seed_differences.append(
                (sampled[baseline_model] - sampled[proposed_model]).mean()
            )
        bootstrap_values[bootstrap_index] = np.mean(sampled_seed_differences)
    ci_low, ci_high = np.percentile(bootstrap_values, [2.5, 97.5])

    if len(seed_differences) < 2 or np.allclose(seed_differences, 0.0):
        p_value = 1.0
        effect_size = 0.0
    else:
        p_value = float(
            stats.wilcoxon(
                seed_differences,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
        deviation = seed_differences.std(ddof=1)
        effect_size = (
            float(seed_differences.mean() / deviation)
            if deviation > 0.0
            else np.inf
        )

    return {
        "difference_baseline_minus_moe": observed_difference,
        "relative_improvement_percent": relative_improvement,
        "hierarchical_ci_low": float(ci_low),
        "hierarchical_ci_high": float(ci_high),
        "wilcoxon_p": p_value,
        "cohens_dz": effect_size,
        "number_of_seeds": len(seeds),
        "number_of_paired_seed_events": len(paired),
        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def analyze_experiment(
    event_csv,
    output_directory,
    proposed_model="MoE",
    baselines=("Mamba", "Transformer"),
    metrics=None,
    n_bootstrap=10000,
    random_seed=2026,
):
    dataframe = pd.read_csv(event_csv)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics = [metric for metric in (metrics or DEFAULT_METRICS) if metric in dataframe]
    rng = np.random.default_rng(random_seed)

    required_columns = {"dataset", "model", "seed", "event_id"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Event CSV is missing columns: {sorted(missing)}")

    duplicate_count = dataframe.duplicated(
        ["dataset", "model", "seed", "event_id"]
    ).sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate seed-event rows")

    seed_level = (
        dataframe.groupby(["dataset", "model", "seed"])[metrics]
        .mean()
        .reset_index()
    )
    seed_level.to_csv(output_directory / "metrics_by_seed.csv", index=False)

    summary_rows = []
    for dataset_name in dataframe["dataset"].unique():
        dataset_frame = dataframe[dataframe["dataset"] == dataset_name]
        for model_name in dataset_frame["model"].unique():
            seed_frame = seed_level[
                (seed_level["dataset"] == dataset_name)
                & (seed_level["model"] == model_name)
            ]
            event_counts = dataset_frame[
                dataset_frame["model"] == model_name
            ].groupby("seed").size()
            for metric in metrics:
                ci_low, ci_high = _hierarchical_absolute_ci(
                    dataset_frame,
                    model_name,
                    metric,
                    rng,
                    n_bootstrap,
                )
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "metric": metric,
                        "mean_across_seeds": float(seed_frame[metric].mean()),
                        "std_across_seeds": float(seed_frame[metric].std(ddof=1)),
                        "hierarchical_ci_low": float(ci_low),
                        "hierarchical_ci_high": float(ci_high),
                        "number_of_seeds": int(len(seed_frame)),
                        "number_of_events_per_seed_min": int(
                            event_counts.min()
                        ),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_directory / "summary_mean_sd_ci.csv", index=False)

    comparison_rows = []
    available_models = set(dataframe["model"].unique())
    selected_baselines = [model for model in baselines if model in available_models]
    for dataset_name in dataframe["dataset"].unique():
        dataset_frame = dataframe[dataframe["dataset"] == dataset_name]
        for baseline in selected_baselines:
            for metric in metrics:
                result = _paired_comparison(
                    dataset_frame,
                    proposed_model,
                    baseline,
                    metric,
                    rng,
                    n_bootstrap,
                )
                result.update(
                    {
                        "dataset": dataset_name,
                        "proposed_model": proposed_model,
                        "baseline_model": baseline,
                        "metric": metric,
                    }
                )
                comparison_rows.append(result)

    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["holm_adjusted_p"] = np.nan
        for dataset_name in comparisons["dataset"].unique():
            mask = comparisons["dataset"] == dataset_name
            comparisons.loc[mask, "holm_adjusted_p"] = holm_adjust(
                comparisons.loc[mask, "wilcoxon_p"].to_numpy()
            )
        comparisons["significant_0_05"] = comparisons["holm_adjusted_p"] < 0.05
    comparisons.to_csv(
        output_directory / "paired_statistical_tests.csv", index=False
    )
    return summary, comparisons


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("event_csv")
    parser.add_argument("output_directory")
    parser.add_argument("--bootstrap", type=int, default=10000)
    arguments = parser.parse_args()
    analyze_experiment(
        arguments.event_csv,
        arguments.output_directory,
        n_bootstrap=arguments.bootstrap,
    )

