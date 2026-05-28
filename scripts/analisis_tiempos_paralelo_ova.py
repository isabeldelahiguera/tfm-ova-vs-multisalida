from __future__ import annotations

from pathlib import Path

import pandas as pd


SEQUENTIAL_CSV = {
    "iris": Path("resultados_actualizados/secuencial/exp_iris_mlp_138892.csv"),
    "wine": Path("resultados_actualizados/secuencial/exp_wine_mlp_138892.csv"),
    "breast_cancer": Path("resultados_actualizados/secuencial/exp_breast_cancer_mlp_138892.csv"),
    "digits": Path("resultados_actualizados/secuencial/exp_digits_mlp_138892.csv"),
    "mnist": Path("resultados_actualizados/secuencial/exp_mnist_vgg_138893.csv"),
    "cifar10": Path("resultados_actualizados/secuencial/exp_cifar10_vgg_138894.csv"),
    "brisc": Path("resultados_actualizados/secuencial/exp_brisc_vgg_128_139535.csv"),
    "tb_chest_xray": Path("resultados_actualizados/secuencial/exp_tb_chest_xray_vgg_128_139536.csv"),
}

PARALLEL_DIR = Path("resultados_actualizados/paralelo/ova_parallel_actual")
PARALLEL_ARCH = {
    "iris": "mlp",
    "wine": "mlp",
    "breast_cancer": "mlp",
    "digits": "mlp",
    "mnist": "vgg",
    "cifar10": "vgg",
    "brisc": "vgg",
    "tb_chest_xray": "vgg",
}
OUTPUT_DIR = Path("resultados_actualizados/analisis_tiempos")
DETAIL_CSV = OUTPUT_DIR / "tiempos_ova_paralelo_actual_detalle.csv"
SUMMARY_CSV = OUTPUT_DIR / "tiempos_ova_paralelo_actual_resumen.csv"


def require_columns(df: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def parallel_result_path(dataset: str) -> Path:
    arch = PARALLEL_ARCH[dataset]
    return PARALLEL_DIR / f"exp_{dataset}_{arch}_parallel_ova.csv"


def parallel_class_times_path(dataset: str) -> Path:
    arch = PARALLEL_ARCH[dataset]
    return PARALLEL_DIR / f"exp_{dataset}_{arch}_parallel_ova_class_times.csv"


def load_dataset_times(dataset: str) -> pd.DataFrame:
    sequential_path = SEQUENTIAL_CSV[dataset]
    parallel_path = parallel_result_path(dataset)
    class_times_path = parallel_class_times_path(dataset)
    sequential_df = pd.read_csv(sequential_path)
    parallel_df = pd.read_csv(parallel_path)
    class_times_df = pd.read_csv(class_times_path)
    require_columns(
        sequential_df,
        {"dataset", "seed", "target_dim", "model_arch", "hidden_layers", "model_type", "train_time_seconds"},
        sequential_path,
    )
    require_columns(
        parallel_df,
        {
            "seed",
            "train_time_seconds",
            "parallel_train_time_seconds",
            "ova_model_train_time_seconds_mean",
        },
        parallel_path,
    )
    require_columns(class_times_df, {"seed", "train_time_seconds"}, class_times_path)

    sequential_times = sequential_df.pivot_table(
        index="seed",
        columns="model_type",
        values="train_time_seconds",
        aggfunc="mean",
    )
    if "OVA" not in sequential_times or "multi-output" not in sequential_times:
        raise ValueError(f"Expected OVA and multi-output rows in {sequential_path}")

    metadata = sequential_df.iloc[0]
    parallel_ova = parallel_df.set_index("seed")
    ideal_parallel_time = class_times_df.groupby("seed")["train_time_seconds"].max()
    common_seeds = sequential_times.index.intersection(parallel_ova.index).intersection(
        ideal_parallel_time.index
    )
    detail = pd.DataFrame(
        {
            "dataset": dataset,
            "seed": common_seeds,
            "class_count": int(metadata["target_dim"]),
            "model_arch": metadata["model_arch"],
            "hidden_layers": "n/a" if pd.isna(metadata["hidden_layers"]) else metadata["hidden_layers"],
            "multi_time_seconds": sequential_times.loc[common_seeds, "multi-output"].to_numpy(),
            "ova_sequential_time_seconds": sequential_times.loc[common_seeds, "OVA"].to_numpy(),
            "ova_parallel_ideal_time_seconds": ideal_parallel_time.loc[common_seeds].to_numpy(),
        }
    )
    detail["ova_ideal_speedup"] = (
        detail["ova_sequential_time_seconds"] / detail["ova_parallel_ideal_time_seconds"]
    )
    detail["ova_ideal_vs_multi_ratio"] = (
        detail["ova_parallel_ideal_time_seconds"] / detail["multi_time_seconds"]
    )
    return detail


def summarize_times(detail: pd.DataFrame) -> pd.DataFrame:
    grouped = detail.groupby(
        ["dataset", "class_count", "model_arch", "hidden_layers"],
        as_index=False,
        dropna=False,
        sort=False,
    )
    summary = grouped.agg(
        seeds_used=("seed", "nunique"),
        multi_time_seconds_mean=("multi_time_seconds", "mean"),
        multi_time_seconds_std=("multi_time_seconds", "std"),
        ova_sequential_time_seconds_mean=("ova_sequential_time_seconds", "mean"),
        ova_sequential_time_seconds_std=("ova_sequential_time_seconds", "std"),
        ova_parallel_ideal_time_seconds_mean=("ova_parallel_ideal_time_seconds", "mean"),
        ova_parallel_ideal_time_seconds_std=("ova_parallel_ideal_time_seconds", "std"),
        ova_ideal_speedup_mean=("ova_ideal_speedup", "mean"),
        ova_ideal_vs_multi_ratio_mean=("ova_ideal_vs_multi_ratio", "mean"),
    )
    return summary


def main() -> None:
    detail = pd.concat([load_dataset_times(dataset) for dataset in SEQUENTIAL_CSV], ignore_index=True)
    summary = summarize_times(detail)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved per-seed detail to {DETAIL_CSV}")
    print(f"Saved dataset summary to {SUMMARY_CSV}\n")
    print(
        summary[
            [
                "dataset",
                "class_count",
                "seeds_used",
                "multi_time_seconds_mean",
                "ova_sequential_time_seconds_mean",
                "ova_parallel_ideal_time_seconds_mean",
                "ova_ideal_speedup_mean",
                "ova_ideal_vs_multi_ratio_mean",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()
