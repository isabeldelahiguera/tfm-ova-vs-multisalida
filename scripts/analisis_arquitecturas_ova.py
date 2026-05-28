from __future__ import annotations

from pathlib import Path

import pandas as pd


BASELINE_CSV = {
    "iris": Path("resultados_actualizados/secuencial/exp_iris_mlp_138892.csv"),
    "wine": Path("resultados_actualizados/secuencial/exp_wine_mlp_138892.csv"),
    "breast_cancer": Path("resultados_actualizados/secuencial/exp_breast_cancer_mlp_138892.csv"),
    "digits": Path("resultados_actualizados/secuencial/exp_digits_mlp_138892.csv"),
    "mnist": Path("resultados_actualizados/secuencial/exp_mnist_vgg_138893.csv"),
    "cifar10": Path("resultados_actualizados/secuencial/exp_cifar10_vgg_138894.csv"),
    "brisc": Path("resultados_actualizados/secuencial/exp_brisc_vgg_128_139535.csv"),
    "tb_chest_xray": Path("resultados_actualizados/secuencial/exp_tb_chest_xray_vgg_128_139536.csv"),
}

REDUCED_OVA_CSV = {
    "iris": {
        "[24, 12]": Path("resultados_actualizados/arquitecturas_ova/exp_iris_mlp_OVA_24_12.csv"),
        "[16, 8]": Path("resultados_actualizados/arquitecturas_ova/exp_iris_mlp_OVA_16_8.csv"),
    },
    "wine": {
        "[24, 12]": Path("resultados_actualizados/arquitecturas_ova/exp_wine_mlp_OVA_24_12.csv"),
        "[16, 8]": Path("resultados_actualizados/arquitecturas_ova/exp_wine_mlp_OVA_16_8.csv"),
    },
    "breast_cancer": {
        "[24, 12]": Path("resultados_actualizados/arquitecturas_ova/exp_breast_cancer_mlp_OVA_24_12.csv"),
        "[16, 8]": Path("resultados_actualizados/arquitecturas_ova/exp_breast_cancer_mlp_OVA_16_8.csv"),
    },
    "digits": {
        "[24, 12]": Path("resultados_actualizados/arquitecturas_ova/exp_digits_mlp_OVA_24_12.csv"),
        "[16, 8]": Path("resultados_actualizados/arquitecturas_ova/exp_digits_mlp_OVA_16_8.csv"),
    },
    "mnist": {
        "[24, 48, 96]": "resultados_actualizados/arquitecturas_ova_vgg/vgg_24_48_96/exp_mnist_vgg_*.csv",
        "[16, 32, 64]": "resultados_actualizados/arquitecturas_ova_vgg/vgg_16_32_64/exp_mnist_vgg_*.csv",
    },
    "cifar10": {
        "[24, 48, 96]": "resultados_actualizados/arquitecturas_ova_vgg/vgg_24_48_96/exp_cifar10_vgg_*.csv",
        "[16, 32, 64]": "resultados_actualizados/arquitecturas_ova_vgg/vgg_16_32_64/exp_cifar10_vgg_*.csv",
    },
    "brisc": {
        "[24, 48, 96]": "resultados_actualizados/arquitecturas_ova_vgg/vgg_24_48_96/exp_brisc_vgg_128_*.csv",
        "[16, 32, 64]": "resultados_actualizados/arquitecturas_ova_vgg/vgg_16_32_64/exp_brisc_vgg_128_*.csv",
    },
    "tb_chest_xray": {
        "[24, 48, 96]": "resultados_actualizados/arquitecturas_ova_vgg/vgg_24_48_96/exp_tb_chest_xray_vgg_128_*.csv",
        "[16, 32, 64]": "resultados_actualizados/arquitecturas_ova_vgg/vgg_16_32_64/exp_tb_chest_xray_vgg_128_*.csv",
    },
}

DATASET_BALANCE = {
    "iris": "balanceado",
    "wine": "desbalanceado",
    "breast_cancer": "desbalanceado",
    "digits": "balanceado",
    "mnist": "balanceado",
    "cifar10": "balanceado",
    "brisc": "desbalanceado",
    "tb_chest_xray": "desbalanceado",
}

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "tpr_macro",
    "fpr_macro",
    "tnr_macro",
    "fnr_macro",
]

OUTPUT_DIR = Path("resultados_actualizados/analisis_arquitecturas_ova")
DETAIL_CSV = OUTPUT_DIR / "arquitecturas_ova_10semillas_detalle.csv"
SUMMARY_CSV = OUTPUT_DIR / "arquitecturas_ova_10semillas_resumen.csv"


def resolve_csv(path_spec: str | Path) -> Path:
    if isinstance(path_spec, Path):
        return path_spec

    matches = sorted(
        path
        for path in Path().glob(path_spec)
        if not path.name.endswith("_summary.csv")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one CSV for pattern {path_spec!r}, found {len(matches)}: {matches}"
        )
    return matches[0]


def require_columns(df: pd.DataFrame, path: Path) -> None:
    required = {
        "dataset",
        "seed",
        "model_type",
        "model_arch",
        "hidden_layers",
        "vgg_channels",
        "target_dim",
        "train_time_seconds",
        *METRICS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def load_result(path_spec: str | Path, architecture_role: str) -> pd.DataFrame:
    path = resolve_csv(path_spec)
    df = pd.read_csv(path)
    require_columns(df, path)
    detail_columns = [
        "dataset",
        "seed",
        "target_dim",
        "model_type",
        "model_arch",
        "hidden_layers",
        "vgg_channels",
        "train_time_seconds",
        *METRICS,
    ]
    detail = df[detail_columns].copy()
    detail["architecture_role"] = architecture_role
    return detail


def load_architecture_detail() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset, baseline_path in BASELINE_CSV.items():
        baseline = load_result(baseline_path, "referencia")
        frames.append(baseline)
        for architecture, reduced_path in REDUCED_OVA_CSV[dataset].items():
            reduced = load_result(reduced_path, f"ova_reducida_{architecture}")
            frames.append(reduced)

    detail = pd.concat(frames, ignore_index=True)
    detail["dataset_balance"] = detail["dataset"].map(DATASET_BALANCE)
    if detail["dataset_balance"].isna().any():
        missing = sorted(detail.loc[detail["dataset_balance"].isna(), "dataset"].unique())
        raise ValueError(f"Missing balance label for datasets: {missing}")

    detail["architecture"] = detail["hidden_layers"].where(
        detail["model_arch"] == "mlp",
        detail["vgg_channels"],
    )
    detail["approach_architecture"] = detail["model_type"] + " " + detail["architecture"].astype(str)
    ordered_columns = [
        "dataset",
        "dataset_balance",
        "seed",
        "target_dim",
        "model_type",
        "model_arch",
        "architecture",
        "hidden_layers",
        "vgg_channels",
        "approach_architecture",
        "architecture_role",
        *METRICS,
        "train_time_seconds",
    ]
    return detail[ordered_columns]


def summarize_architectures(detail: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "dataset",
        "dataset_balance",
        "target_dim",
        "model_type",
        "model_arch",
        "architecture",
        "hidden_layers",
        "vgg_channels",
        "approach_architecture",
        "architecture_role",
    ]
    summary = (
        detail.groupby(grouping, as_index=False, dropna=False, sort=False)
        .agg(
            seeds_used=("seed", "nunique"),
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in METRICS
            },
            **{
                f"{metric}_std": (metric, "std")
                for metric in METRICS
            },
            train_time_seconds_mean=("train_time_seconds", "mean"),
            train_time_seconds_std=("train_time_seconds", "std"),
        )
    )
    return summary


def main() -> None:
    detail = load_architecture_detail()
    summary = summarize_architectures(detail)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved per-seed architecture detail to {DETAIL_CSV}")
    print(f"Saved architecture metric summary to {SUMMARY_CSV}\n")
    print(
        summary[
            [
                "dataset",
                "dataset_balance",
                "approach_architecture",
                "seeds_used",
                "accuracy_mean",
                "balanced_accuracy_mean",
                "f1_macro_mean",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()
