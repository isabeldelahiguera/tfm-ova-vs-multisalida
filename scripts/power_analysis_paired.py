from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import pandas as pd
from statsmodels.stats.power import TTestPower


DEFAULT_DATASETS = ("iris", "wine", "breast_cancer", "digits", "mnist", "cifar10", "brisc", "tuberculosis")
MEDICAL_DATASETS = {"brisc", "tb_chest_xray"}
DEFAULT_PATTERNS = {
    "iris": "resultados_slurm/exp_iris_mlp_*.csv",
    "wine": "resultados_slurm/exp_wine_mlp_*.csv",
    "breast_cancer": "resultados_slurm/exp_breast_cancer_mlp_*.csv",
    "digits": "resultados_slurm/exp_digits_mlp_*.csv",
    "mnist": "resultados_slurm/exp_mnist_vgg_*.csv",
    "cifar10": "resultados_slurm/exp_cifar10_vgg_*.csv",
    "brisc": "resultados_slurm/exp_brisc_vgg_*.csv",
    "tuberculosis": "resultados_slurm/exp_tb_vgg_*.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "A priori power analysis for paired OVA vs multi-output comparisons, "
            "using an existing run as a pilot to estimate the SD of paired differences."
        )
    )
    parser.add_argument(
        "csv",
        nargs="*",
        help="Detailed experiment CSV files. If omitted, latest configured CSVs from resultados_slurm are used.",
    )
    parser.add_argument(
        "--metric",
        default="f1_macro",
        help="Primary metric used for the paired differences. Default: f1_macro.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level. Default: 0.05.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=None,
        help=(
            "Target statistical power for all datasets. If omitted, defaults are 0.80 "
            "for classic datasets and 0.90 for medical datasets."
        ),
    )
    parser.add_argument(
        "--min-diff",
        type=float,
        default=0.01,
        help="Minimum relevant difference to detect in the selected metric. Default: 0.01.",
    )
    parser.add_argument(
        "--model-a",
        default="OVA",
        help="First model in the paired difference model_a - model_b. Default: OVA.",
    )
    parser.add_argument(
        "--model-b",
        default="multi-output",
        help="Second model in the paired difference model_a - model_b. Default: multi-output.",
    )
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "larger", "smaller"),
        default="two-sided",
        help="Alternative hypothesis for the power calculation. Default: two-sided.",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=3,
        help="Minimum practical number of paired seeds to report. Default: 3.",
    )
    return parser


def default_csv_paths() -> list[Path]:
    paths = []
    for dataset in DEFAULT_DATASETS:
        candidates = sorted(
            path for path in Path(".").glob(DEFAULT_PATTERNS[dataset]) if not path.name.endswith("_summary.csv")
        )
        if candidates:
            paths.append(candidates[-1])
    return paths


def dataset_name(df: pd.DataFrame, csv_path: Path) -> str:
    if "dataset" in df.columns and not df["dataset"].dropna().empty:
        return str(df["dataset"].dropna().iloc[0])
    return csv_path.stem.replace("exp_", "")


def target_power_for_dataset(dataset: str, override_power: float | None) -> float:
    if override_power is not None:
        return override_power
    normalized_dataset = dataset.lower()
    return 0.90 if normalized_dataset in MEDICAL_DATASETS else 0.80


def paired_differences(
    df: pd.DataFrame,
    metric: str,
    model_a: str,
    model_b: str,
) -> pd.Series:
    required_columns = {"seed", "model_type", metric}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    subset = df[df["model_type"].isin([model_a, model_b])].copy()
    wide = subset.pivot_table(index="seed", columns="model_type", values=metric, aggfunc="mean")

    if model_a not in wide.columns or model_b not in wide.columns:
        raise ValueError(f"Both {model_a!r} and {model_b!r} must be present.")

    paired = wide[[model_a, model_b]].dropna().sort_index()
    if paired.empty:
        raise ValueError("No paired seeds available after aligning both models.")

    return paired[model_a] - paired[model_b]


def required_seeds(
    min_diff: float,
    sd_diff: float,
    alpha: float,
    target_power: float,
    alternative: str,
    min_seeds: int,
) -> tuple[float, int, float]:
    if min_diff <= 0:
        raise ValueError("--min-diff must be positive.")
    if sd_diff <= 0:
        raise ValueError("Pilot SD of paired differences must be positive.")

    effect_size = min_diff / sd_diff
    analysis = TTestPower()
    n_required = max(2, min_seeds)
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            achieved_power = analysis.power(
                effect_size=effect_size,
                nobs=n_required,
                alpha=alpha,
                alternative=alternative,
            )
        if math.isfinite(achieved_power) and achieved_power >= target_power:
            break
        n_required += 1

    return effect_size, n_required, float(n_required)


def main() -> None:
    args = build_parser().parse_args()
    csv_paths = [Path(path) for path in args.csv] if args.csv else default_csv_paths()

    if not csv_paths:
        raise SystemExit("No CSV files provided and no default classic exp_*.csv files found.")

    rows = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        diffs = paired_differences(df, args.metric, args.model_a, args.model_b)
        dataset = dataset_name(df, csv_path)
        target_power = target_power_for_dataset(dataset, args.power)
        sd_diff = diffs.std(ddof=1)
        mean_diff = diffs.mean()
        effect_size, n_required, n_float = required_seeds(
            min_diff=args.min_diff,
            sd_diff=sd_diff,
            alpha=args.alpha,
            target_power=target_power,
            alternative=args.alternative,
            min_seeds=args.min_seeds,
        )
        rows.append(
            {
                "dataset": dataset,
                "csv": str(csv_path),
                "metric": args.metric,
                "pilot_seeds": len(diffs),
                "pilot_mean_diff": mean_diff,
                "pilot_sd_diff": sd_diff,
                "min_relevant_diff": args.min_diff,
                "effect_size_dz": effect_size,
                "alpha": args.alpha,
                "target_power": target_power,
                "alternative": args.alternative,
                "required_seeds": n_required,
                "raw_required_seeds": n_float,
            }
        )

    result = pd.DataFrame(rows)
    display_columns = [
        "dataset",
        "metric",
        "pilot_seeds",
        "pilot_mean_diff",
        "pilot_sd_diff",
        "min_relevant_diff",
        "effect_size_dz",
        "target_power",
        "required_seeds",
    ]
    print(result[display_columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(
        "Interpretation: required_seeds is the estimated number of paired seeds needed "
        f"to detect a difference of {args.min_diff:g} with alpha={args.alpha:g}."
    )
    if args.power is None:
        print("Target power defaults: 0.80 for classic datasets and 0.90 for medical datasets.")
    else:
        print(f"Target power override used for all datasets: {args.power:g}.")
    print(
        "The SD is estimated from the provided CSVs as a pilot, using paired differences "
        f"{args.model_a} - {args.model_b}."
    )


if __name__ == "__main__":
    main()
