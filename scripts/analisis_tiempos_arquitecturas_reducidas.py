from __future__ import annotations

from pathlib import Path

import pandas as pd


ARCHITECTURE_DETAIL_CSV = Path(
    "resultados_actualizados/analisis_arquitecturas_ova/arquitecturas_ova_10semillas_detalle.csv"
)
BASE_PARALLEL_DIR = Path("resultados_actualizados/paralelo/ova_parallel_actual")

COMPARISONS = {
    "wine": {
        "architecture": "OVA [16, 8]",
        "parallel_dir": Path("resultados_actualizados/paralelo/ova_parallel_reducida_mlp_16_8"),
        "parallel_arch": "mlp",
    },
    "breast_cancer": {
        "architecture": "OVA [16, 8]",
        "parallel_dir": Path("resultados_actualizados/paralelo/ova_parallel_reducida_mlp_16_8"),
        "parallel_arch": "mlp",
    },
    "digits": {
        "architecture": "OVA [16, 8]",
        "parallel_dir": Path("resultados_actualizados/paralelo/ova_parallel_reducida_mlp_16_8"),
        "parallel_arch": "mlp",
    },
    "mnist": {
        "architecture": "OVA [16, 32, 64]",
        "parallel_dir": Path("resultados_actualizados/paralelo/ova_parallel_reducida_vgg_16_32_64"),
        "parallel_arch": "vgg",
    },
    "tb_chest_xray": {
        "architecture": "OVA [16, 32, 64]",
        "parallel_dir": Path("resultados_actualizados/paralelo/ova_parallel_reducida_vgg_16_32_64"),
        "parallel_arch": "vgg",
    },
}

OUTPUT_DIR = Path("resultados_actualizados/analisis_tiempos")
DETAIL_CSV = OUTPUT_DIR / "tiempos_arquitecturas_reducidas_detalle.csv"
SUMMARY_CSV = OUTPUT_DIR / "tiempos_arquitecturas_reducidas_resumen.csv"


def require_columns(df: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def parallel_result_path(dataset: str, config: dict[str, object]) -> Path:
    return Path(config["parallel_dir"]) / f"exp_{dataset}_{config['parallel_arch']}_parallel_ova.csv"


def parallel_class_times_path(dataset: str, config: dict[str, object]) -> Path:
    return Path(config["parallel_dir"]) / f"exp_{dataset}_{config['parallel_arch']}_parallel_ova_class_times.csv"


def base_parallel_class_times_path(dataset: str, config: dict[str, object]) -> Path:
    return BASE_PARALLEL_DIR / f"exp_{dataset}_{config['parallel_arch']}_parallel_ova_class_times.csv"


def load_reduced_times(dataset: str, config: dict[str, object], detail_df: pd.DataFrame) -> pd.DataFrame:
    architecture = str(config["architecture"])
    parallel_path = parallel_result_path(dataset, config)
    class_times_path = parallel_class_times_path(dataset, config)
    base_class_times_path = base_parallel_class_times_path(dataset, config)
    parallel_df = pd.read_csv(parallel_path)
    class_times_df = pd.read_csv(class_times_path)
    base_class_times_df = pd.read_csv(base_class_times_path)
    require_columns(
        parallel_df,
        {"seed", "parallel_train_time_seconds"},
        parallel_path,
    )
    require_columns(class_times_df, {"seed", "train_time_seconds"}, class_times_path)
    require_columns(base_class_times_df, {"seed", "train_time_seconds"}, base_class_times_path)

    subset = detail_df[
        (detail_df["dataset"] == dataset)
        & detail_df["approach_architecture"].isin(
            ["multi-output [32, 16]", "multi-output [32, 64, 128]", architecture]
        )
    ].copy()
    if subset.empty:
        raise ValueError(f"No architecture detail rows found for {dataset} / {architecture}")

    wide = subset.pivot_table(
        index="seed",
        columns="approach_architecture",
        values="train_time_seconds",
        aggfunc="mean",
    )
    multi_columns = [column for column in wide.columns if str(column).startswith("multi-output")]
    if len(multi_columns) != 1:
        raise ValueError(f"Expected one multi-output reference for {dataset}: {multi_columns}")
    if architecture not in wide.columns:
        raise ValueError(f"Missing reduced architecture {architecture!r} for {dataset}")

    parallel_ova = parallel_df.set_index("seed")
    reduced_ideal_parallel_time = class_times_df.groupby("seed")["train_time_seconds"].max()
    base_ideal_parallel_time = base_class_times_df.groupby("seed")["train_time_seconds"].max()
    common_seeds = (
        wide.index.intersection(parallel_ova.index)
        .intersection(reduced_ideal_parallel_time.index)
        .intersection(base_ideal_parallel_time.index)
    )
    metadata = subset.iloc[0]
    result = pd.DataFrame(
        {
            "dataset": dataset,
            "seed": common_seeds,
            "class_count": int(metadata["target_dim"]),
            "model_arch": metadata["model_arch"],
            "architecture": architecture.replace("OVA ", ""),
            "multi_time_seconds": wide.loc[common_seeds, multi_columns[0]].to_numpy(),
            "ova_reduced_sequential_time_seconds": wide.loc[common_seeds, architecture].to_numpy(),
            "ova_base_parallel_ideal_time_seconds": base_ideal_parallel_time.loc[common_seeds].to_numpy(),
            "ova_reduced_parallel_ideal_time_seconds": reduced_ideal_parallel_time.loc[common_seeds].to_numpy(),
        }
    )
    result["ova_reduced_ideal_speedup"] = (
        result["ova_reduced_sequential_time_seconds"]
        / result["ova_reduced_parallel_ideal_time_seconds"]
    )
    result["ova_reduced_ideal_vs_multi_ratio"] = (
        result["ova_reduced_parallel_ideal_time_seconds"]
        / result["multi_time_seconds"]
    )
    result["ova_base_ideal_vs_multi_ratio"] = (
        result["ova_base_parallel_ideal_time_seconds"]
        / result["multi_time_seconds"]
    )
    result["ova_reduced_vs_base_parallel_ratio"] = (
        result["ova_reduced_parallel_ideal_time_seconds"]
        / result["ova_base_parallel_ideal_time_seconds"]
    )
    result["ova_reduced_sequential_vs_multi_ratio"] = (
        result["ova_reduced_sequential_time_seconds"]
        / result["multi_time_seconds"]
    )
    return result


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(
            ["dataset", "class_count", "model_arch", "architecture"],
            as_index=False,
            sort=False,
        )
        .agg(
            seeds_used=("seed", "nunique"),
            multi_time_seconds_mean=("multi_time_seconds", "mean"),
            multi_time_seconds_std=("multi_time_seconds", "std"),
            ova_reduced_sequential_time_seconds_mean=("ova_reduced_sequential_time_seconds", "mean"),
            ova_reduced_sequential_time_seconds_std=("ova_reduced_sequential_time_seconds", "std"),
            ova_base_parallel_ideal_time_seconds_mean=("ova_base_parallel_ideal_time_seconds", "mean"),
            ova_base_parallel_ideal_time_seconds_std=("ova_base_parallel_ideal_time_seconds", "std"),
            ova_reduced_parallel_ideal_time_seconds_mean=("ova_reduced_parallel_ideal_time_seconds", "mean"),
            ova_reduced_parallel_ideal_time_seconds_std=("ova_reduced_parallel_ideal_time_seconds", "std"),
            ova_reduced_ideal_speedup_mean=("ova_reduced_ideal_speedup", "mean"),
            ova_base_ideal_vs_multi_ratio_mean=("ova_base_ideal_vs_multi_ratio", "mean"),
            ova_reduced_ideal_vs_multi_ratio_mean=("ova_reduced_ideal_vs_multi_ratio", "mean"),
            ova_reduced_vs_base_parallel_ratio_mean=("ova_reduced_vs_base_parallel_ratio", "mean"),
            ova_reduced_sequential_vs_multi_ratio_mean=("ova_reduced_sequential_vs_multi_ratio", "mean"),
        )
    )


def main() -> None:
    detail_df = pd.read_csv(ARCHITECTURE_DETAIL_CSV)
    require_columns(
        detail_df,
        {
            "dataset",
            "seed",
            "target_dim",
            "model_arch",
            "approach_architecture",
            "train_time_seconds",
        },
        ARCHITECTURE_DETAIL_CSV,
    )
    detail = pd.concat(
        [load_reduced_times(dataset, config, detail_df) for dataset, config in COMPARISONS.items()],
        ignore_index=True,
    )
    summary = summarize(detail)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved reduced architecture time detail to {DETAIL_CSV}")
    print(f"Saved reduced architecture time summary to {SUMMARY_CSV}\n")
    print(
        summary[
            [
                "dataset",
                "architecture",
                "seeds_used",
                "multi_time_seconds_mean",
                "ova_base_parallel_ideal_time_seconds_mean",
                "ova_reduced_parallel_ideal_time_seconds_mean",
                "ova_reduced_vs_base_parallel_ratio_mean",
                "ova_base_ideal_vs_multi_ratio_mean",
                "ova_reduced_ideal_vs_multi_ratio_mean",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()
