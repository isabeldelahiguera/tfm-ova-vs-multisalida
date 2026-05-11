from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired Wilcoxon test between OVA and multi-output.")
    parser.add_argument("--csv", required=True, help="Path to the detailed experiment CSV.")
    parser.add_argument("--metric", required=True, help="Metric column to compare, e.g. balanced_accuracy.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    required_columns = {"seed", "model_type", args.metric}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    model_a = "OVA"
    model_b = "multi-output"

    subset = df[df["model_type"].isin([model_a, model_b])].copy()
    if subset.empty:
        raise ValueError("No rows found for the requested model types.")

    wide = subset.pivot(index="seed", columns="model_type", values=args.metric)
    if model_a not in wide.columns or model_b not in wide.columns:
        raise ValueError("Both OVA and multi-output must be present for the selected metric.")

    paired = wide[[model_a, model_b]].dropna().sort_index()
    diffs = paired[model_a] - paired[model_b]

    if len(paired) == 0:
        raise ValueError("No paired seeds available after aligning both models.")

    result = wilcoxon(
        paired[model_a],
        paired[model_b],
        alternative="two-sided",
        zero_method="wilcox",
    )

    print(f"CSV: {csv_path}")
    print(f"Metric: {args.metric}")
    print(f"Seeds used: {len(paired)}")
    print("H0: The median of the paired differences (OVA - multi-output) is 0.")
    print("H1: The median of the paired differences (OVA - multi-output) is not 0.")
    print()
    print("Paired values by seed:")
    print(paired.to_string())
    print()
    print(f"Median difference: {diffs.median():.6f}")
    print(f"Mean difference: {diffs.mean():.6f}")
    print(f"Wilcoxon statistic: {result.statistic}")
    print(f"p-value: {result.pvalue:.10f}")


if __name__ == "__main__":
    main()
