from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "multi_cam_top_mask_area_dice",
    "ova_cam_top_mask_area_dice",
    "multi_cam_top_mask_area_outside_precision",
    "ova_cam_top_mask_area_outside_precision",
    "multi_cam_outside_peritumor_r5_activation_frac",
    "ova_cam_outside_peritumor_r5_activation_frac",
    "multi_cam_tumor_activation_frac",
    "ova_cam_tumor_activation_frac",
    "multi_cam_gini",
    "ova_cam_gini",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate Grad-CAM metrics across seeds and target layers."
    )
    parser.add_argument("--dataset", default="brisc")
    parser.add_argument("--root", default="resultados_actualizados/explicabilidad")
    parser.add_argument(
        "--output-dir",
        default="resultados_actualizados/analisis_explicabilidad",
    )
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    return parser


def parse_run_dir(path: Path) -> dict[str, object] | None:
    name = path.name
    parts = name.split("_")
    if len(parts) < 2 or parts[0] != "seed":
        return None

    try:
        seed = int(parts[1])
    except ValueError:
        return None

    cam_target = "true" if "true" in parts and "target" in parts else "predicted"
    if "rise" in parts:
        cam_method = "rise"
    elif "gradcampp" in parts:
        cam_method = "gradcam++"
    else:
        cam_method = "gradcam"
    layer = "last"
    if "_layer_" in name:
        layer = name.split("_layer_", maxsplit=1)[1]
    return {
        "seed": seed,
        "cam_target": cam_target,
        "cam_method": cam_method,
        "target_layer_tag": layer,
        "run_dir": str(path),
    }


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.root) / args.dataset
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    for summary_path in sorted(dataset_root.glob("seed_*/gradcam_metrics_summary.csv")):
        metadata = parse_run_dir(summary_path.parent)
        if metadata is None:
            continue
        summary = pd.read_csv(summary_path)
        all_rows = summary[(summary["group"] == "all") & (summary["value"] == "all")]
        if all_rows.empty:
            continue
        row = dict(metadata)
        for metric in args.metrics:
            row[metric] = float(all_rows.iloc[0][metric]) if metric in all_rows.columns else np.nan
        run_rows.append(row)

    if not run_rows:
        raise SystemExit(f"No Grad-CAM summaries found under {dataset_root}")

    per_run_path = output_dir / "gradcam_seed_layer_runs.csv"
    with per_run_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    runs = pd.DataFrame(run_rows)
    group_cols = ["cam_method", "cam_target", "target_layer_tag"]
    aggregate_rows: list[dict[str, object]] = []
    for group_values, group_df in runs.groupby(group_cols):
        row = dict(zip(group_cols, group_values))
        row["n_seeds"] = int(group_df["seed"].nunique())
        for metric in args.metrics:
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(row)

    aggregate_path = output_dir / "gradcam_seed_layer_aggregate.csv"
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Per-run CSV: {per_run_path}")
    print(f"Aggregate CSV: {aggregate_path}")


if __name__ == "__main__":
    main()
