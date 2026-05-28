from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, wilcoxon


DEFAULT_METRICS = [
    "cam_top20_dice",
    "cam_top_mask_area_dice",
    "cam_pointing_game_hit",
    "cam_active_area_frac_50",
    "cam_gini",
    "cam_inside_frac",
    "cam_top20_iou",
    "cam_top_mask_area_iou",
]


METRIC_LABELS = {
    "cam_top20_dice": "Dice@20",
    "cam_top_mask_area_dice": "Dice@area mascara",
    "cam_pointing_game_hit": "Pointing Game",
    "cam_active_area_frac_50": "Area activa >= 0.5",
    "cam_gini": "Gini",
    "cam_inside_frac": "Activacion dentro tumor",
    "cam_top20_iou": "IoU@20",
    "cam_top_mask_area_iou": "IoU@area mascara",
    "cam_top_mask_area_precision": "Precision@area mascara",
    "cam_top_mask_area_recall": "Recall@area mascara",
    "cam_top_mask_area_outside_precision": "Outside@area mascara",
    "cam_thr50_dice": "Dice@0.5",
    "cam_thr50_iou": "IoU@0.5",
    "cam_thr50_precision": "Precision@0.5",
    "cam_thr50_recall": "Recall@0.5",
    "cam_thr50_outside_precision": "Outside@0.5",
    "cam_thr75_dice": "Dice@0.75",
    "cam_thr75_iou": "IoU@0.75",
    "cam_thr75_precision": "Precision@0.75",
    "cam_thr75_recall": "Recall@0.75",
    "cam_thr75_outside_precision": "Outside@0.75",
    "cam_tumor_activation_frac": "Activacion en tumor",
    "cam_peritumor_r5_activation_frac": "Activacion peritumoral r5",
    "cam_outside_peritumor_r5_activation_frac": "Activacion fuera tumor+peritumor",
    "cam_tumor_mean": "Media CAM tumor",
    "cam_peritumor_r5_mean": "Media CAM peritumor r5",
    "cam_outside_peritumor_r5_mean": "Media CAM fuera tumor+peritumor",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analisis estadistico y visual de metricas Grad-CAM multi-output vs OVA."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "resultados_actualizados/explicabilidad/brisc/seed_1_true_target/gradcam_index.csv",
            "resultados_actualizados/explicabilidad/brisc/seed_1/gradcam_index.csv",
        ],
        help="CSV gradcam_index.csv a analizar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados_actualizados/analisis_explicabilidad/brisc"),
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--bootstrap-replicas", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, replicas: int) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(replicas, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_direction_effect(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero))


def safe_wilcoxon(diff: np.ndarray) -> tuple[float, float]:
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return math.nan, math.nan
    result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def paired_sign_p(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return math.nan
    positives = int(np.sum(nonzero > 0))
    return float(binomtest(positives, len(nonzero), p=0.5, alternative="two-sided").pvalue)


def target_name(path: Path, frame: pd.DataFrame) -> str:
    if "cam_target_mode" in frame.columns and frame["cam_target_mode"].notna().any():
        return str(frame["cam_target_mode"].iloc[0])
    if "true_target" in str(path):
        return "true"
    return "predicted"


def metric_stats(
    frame: pd.DataFrame,
    metric: str,
    target: str,
    group: str,
    rng: np.random.Generator,
    bootstrap_replicas: int,
) -> dict[str, object]:
    multi_col = f"multi_{metric}"
    ova_col = f"ova_{metric}"
    values = frame[[multi_col, ova_col]].dropna()
    multi = values[multi_col].to_numpy(dtype=float)
    ova = values[ova_col].to_numpy(dtype=float)
    diff = ova - multi
    wilcoxon_stat, wilcoxon_p = safe_wilcoxon(diff)
    ci_low, ci_high = bootstrap_ci(diff, rng, bootstrap_replicas)
    return {
        "target_mode": target,
        "group": group,
        "metric": metric,
        "metric_label": METRIC_LABELS.get(metric, metric),
        "n": len(diff),
        "multi_mean": float(np.mean(multi)),
        "ova_mean": float(np.mean(ova)),
        "diff_ova_minus_multi_mean": float(np.mean(diff)),
        "diff_ci95_low": ci_low,
        "diff_ci95_high": ci_high,
        "diff_median": float(np.median(diff)),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "paired_sign_p": paired_sign_p(diff),
        "paired_direction_effect": paired_direction_effect(diff),
    }


def long_metric_frame(frame: pd.DataFrame, metrics: list[str], target: str) -> pd.DataFrame:
    rows = []
    id_cols = ["test_index", "true_label", "outcome"]
    for metric in metrics:
        for model in ("multi", "ova"):
            col = f"{model}_{metric}"
            if col not in frame.columns:
                continue
            chunk = frame[id_cols].copy()
            chunk["target_mode"] = target
            chunk["metric"] = metric
            chunk["metric_label"] = METRIC_LABELS.get(metric, metric)
            chunk["model"] = "Multi-output" if model == "multi" else "OVA"
            chunk["value"] = frame[col].astype(float)
            rows.append(chunk)
    return pd.concat(rows, ignore_index=True)


def save_metric_plots(frame: pd.DataFrame, output_dir: Path, target: str, metric: str) -> None:
    label = METRIC_LABELS.get(metric, metric)
    multi_col = f"multi_{metric}"
    ova_col = f"ova_{metric}"
    values = frame[["test_index", "outcome", multi_col, ova_col]].dropna().copy()
    values["diff_ova_minus_multi"] = values[ova_col].astype(float) - values[multi_col].astype(float)
    long_df = pd.concat(
        [
            pd.DataFrame({"model": "Multi-output", "value": values[multi_col].astype(float)}),
            pd.DataFrame({"model": "OVA", "value": values[ova_col].astype(float)}),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.violinplot(data=long_df, x="model", y="value", inner="quartile", ax=axes[0])
    axes[0].set_title(f"{label}: distribucion")
    axes[0].set_xlabel("")
    axes[0].set_ylabel(label)

    sns.histplot(values["diff_ova_minus_multi"], bins=40, kde=True, ax=axes[1])
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Diferencia OVA - multi")
    axes[1].set_xlabel("Diferencia pareada")

    sns.scatterplot(data=values, x=multi_col, y=ova_col, hue="outcome", s=18, ax=axes[2])
    lower = float(min(values[multi_col].min(), values[ova_col].min()))
    upper = float(max(values[multi_col].max(), values[ova_col].max()))
    axes[2].plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axes[2].set_title("OVA vs multi por imagen")
    axes[2].set_xlabel("Multi-output")
    axes[2].set_ylabel("OVA")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / f"{target}_{metric}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "figuras"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sns.set_theme(style="whitegrid")

    stat_rows = []
    long_rows = []
    for input_path in args.inputs:
        path = Path(input_path)
        frame = pd.read_csv(path)
        target = target_name(path, frame)
        available_metrics = [
            metric for metric in args.metrics if f"multi_{metric}" in frame.columns and f"ova_{metric}" in frame.columns
        ]
        stat_rows.extend(
            metric_stats(frame, metric, target, "all", rng, args.bootstrap_replicas)
            for metric in available_metrics
        )
        for outcome, group_frame in frame.groupby("outcome", sort=False):
            stat_rows.extend(
                metric_stats(group_frame, metric, target, str(outcome), rng, args.bootstrap_replicas)
                for metric in available_metrics
            )
        long_rows.append(long_metric_frame(frame, available_metrics, target))
        for metric in available_metrics:
            save_metric_plots(frame, plots_dir, target, metric)

    stats_df = pd.DataFrame(stat_rows)
    stats_path = args.output_dir / "gradcam_paired_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    long_df = pd.concat(long_rows, ignore_index=True)
    long_path = args.output_dir / "gradcam_metrics_long.csv"
    long_df.to_csv(long_path, index=False)

    print(f"Saved paired stats to {stats_path}")
    print(f"Saved long metrics to {long_path}")
    print(f"Saved figures to {plots_dir}")
    print(stats_df[stats_df["group"] == "all"].to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
