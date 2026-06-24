from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_LABELS = {
    "iris": "Iris",
    "wine": "Wine",
    "breast_cancer": "Breast Cancer",
    "digits": "Digits",
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "brisc": "BRISC",
    "tb_chest_xray": "TB X-ray",
    "tuberculosis": "TB X-ray",
}

DATASET_ORDER = [
    "iris",
    "wine",
    "breast_cancer",
    "digits",
    "mnist",
    "cifar10",
    "brisc",
    "tb_chest_xray",
]

PRACTICAL_OVA_SELECTION = [
    {
        "dataset": "iris",
        "architecture": "[32, 16]",
        "batch_size": 32,
        "source_contains": "ova_parallel_actual",
    },
    {
        "dataset": "wine",
        "architecture": "[16, 8]",
        "batch_size": 32,
        "source_contains": "ova_parallel_reducida_mlp_16_8",
    },
    {
        "dataset": "breast_cancer",
        "architecture": "[16, 8]",
        "batch_size": 64,
        "source_contains": "ova_parallel_h16_8_bs64",
    },
    {
        "dataset": "digits",
        "architecture": "[32, 16]",
        "batch_size": 64,
        "source_contains": "ova_digits_mlp32_16_bs64",
    },
    {
        "dataset": "mnist",
        "architecture": "[32, 64, 128]",
        "batch_size": 128,
        "source_contains": "ova_vgg32_64_128_bs128",
    },
    {
        "dataset": "cifar10",
        "architecture": "[32, 64, 128]",
        "batch_size": 128,
        "source_contains": "ova_vgg32_64_128_bs128",
    },
    {
        "dataset": "brisc",
        "architecture": "[32, 64, 128]",
        "batch_size": 32,
        "source_contains": "pat7",
    },
    {
        "dataset": "tb_chest_xray",
        "architecture": "[16, 32, 64]",
        "batch_size": 32,
        "source_contains": "ova_parallel_reducida_vgg_16_32_64",
    },
]

COLOR_MULTI = "#5C528E"
COLOR_OVA = "#D0ADCE"
COLOR_SEQ = "#D0ADCE"
COLOR_PAR = "#8678A7"
COLOR_POSITIVE = "#D0ADCE"
COLOR_NEGATIVE = "#5C528E"
COLOR_MARGIN = "#6E6E6E"


def label_dataset(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def compact_architecture(architecture: str) -> str:
    return architecture.replace(" ", "")


def label_practical_config(row: pd.Series) -> str:
    label = f"{label_dataset(row.dataset)}\n{compact_architecture(row.architecture)}; b={int(row.batch_size)}"
    if row.dataset == "brisc":
        label += "; p=7"
    return label


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {dataset: idx for idx, dataset in enumerate(DATASET_ORDER)}
    return df.assign(_order=df["dataset"].map(order).fillna(999)).sort_values("_order").drop(columns="_order")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_reference_delta(reference_csv: Path, output_dir: Path) -> pd.DataFrame:
    df = ordered(pd.read_csv(reference_csv))
    df["delta_f1"] = df["ova_parallel_f1"] - df["multi_f1"]
    labels = [label_dataset(dataset) for dataset in df["dataset"]]
    colors = np.where(df["delta_f1"] >= 0, COLOR_POSITIVE, COLOR_NEGATIVE)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(labels, df["delta_f1"], color=colors, edgecolor="black", linewidth=0.35)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(r"$F1_{macro}^{OVA} - F1_{macro}^{multi}$")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, output_dir, "fig_delta_f1_referencia")

    summary = df[
        [
            "dataset",
            "multi_f1",
            "ova_parallel_f1",
            "delta_f1",
            "multi_time_seconds",
            "ova_seq_time_seconds",
            "ova_parallel_wall_time_seconds",
        ]
    ].copy()
    summary.to_csv(output_dir / "tabla_delta_f1_referencia.csv", index=False)
    return summary


def plot_reference_f1_grouped(reference_csv: Path, output_dir: Path) -> pd.DataFrame:
    df = ordered(pd.read_csv(reference_csv))
    labels = [label_dataset(dataset) for dataset in df["dataset"]]
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(
        x - width / 2,
        df["multi_f1"],
        width,
        label="Multi-salida",
        color=COLOR_MULTI,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x + width / 2,
        df["ova_parallel_f1"],
        width,
        label="OVA",
        color=COLOR_OVA,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.set_ylabel(r"$F1_{macro}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        borderaxespad=0,
        handlelength=1.4,
        columnspacing=1.6,
    )
    save_figure(fig, output_dir, "fig_f1_referencia_multi_vs_ova")

    summary = df[["dataset", "multi_f1", "ova_parallel_f1"]].copy()
    summary["delta_f1"] = summary["ova_parallel_f1"] - summary["multi_f1"]
    summary.to_csv(output_dir / "tabla_f1_referencia_multi_vs_ova.csv", index=False)
    return summary


def plot_time_ratios(reference_csv: Path, output_dir: Path) -> pd.DataFrame:
    df = ordered(pd.read_csv(reference_csv))
    labels = [label_dataset(dataset) for dataset in df["dataset"]]
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(
        x - width / 2,
        df["ova_seq_vs_multi_time_ratio"],
        width,
        label="OVA secuencial / multi",
        color=COLOR_SEQ,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x + width / 2,
        df["ova_parallel_vs_multi_time_ratio"],
        width,
        label="OVA paralelo / multi",
        color=COLOR_PAR,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Ratio de tiempo")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        borderaxespad=0,
        handlelength=1.4,
        columnspacing=1.6,
    )
    save_figure(fig, output_dir, "fig_ratios_tiempo_referencia")

    summary = df[
        [
            "dataset",
            "ova_seq_vs_multi_time_ratio",
            "ova_parallel_vs_multi_time_ratio",
            "ova_parallel_speedup_vs_ova_seq",
        ]
    ].copy()
    summary.to_csv(output_dir / "tabla_ratios_tiempo_referencia.csv", index=False)
    return summary


def plot_reference_times_grouped(
    reference_csv: Path,
    output_dir: Path,
    dataset_filter: list[str] | None = None,
    output_name: str = "fig_tiempos_referencia_multi_ova_seq_ova_par",
) -> pd.DataFrame:
    df = ordered(pd.read_csv(reference_csv))
    if dataset_filter is not None:
        df = df[df["dataset"].isin(dataset_filter)].copy()
    labels = [label_dataset(dataset) for dataset in df["dataset"]]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.bar(
        x - width,
        df["multi_time_seconds"],
        width,
        label="Multi-salida",
        color=COLOR_MULTI,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x,
        df["ova_seq_time_seconds"],
        width,
        label="OVA secuencial",
        color=COLOR_SEQ,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x + width,
        df["ova_parallel_wall_time_seconds"],
        width,
        label="OVA paralelo",
        color=COLOR_PAR,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.set_ylabel("Tiempo de entrenamiento (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        borderaxespad=0,
        handlelength=1.4,
        columnspacing=1.3,
    )
    save_figure(fig, output_dir, output_name)

    summary = df[
        [
            "dataset",
            "multi_time_seconds",
            "ova_seq_time_seconds",
            "ova_parallel_wall_time_seconds",
        ]
    ].copy()
    summary.to_csv(output_dir / f"tabla_{output_name.replace('fig_', '')}.csv", index=False)
    return summary


def plot_reduced_vs_reference_delta(reduced_stats_csv: Path, output_dir: Path) -> pd.DataFrame:
    df = ordered(pd.read_csv(reduced_stats_csv))
    labels = [label_dataset(dataset) for dataset in df["dataset"]]
    colors = np.where(df["diferencia_media"] >= 0, COLOR_POSITIVE, COLOR_NEGATIVE)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(labels, df["diferencia_media"], color=colors, edgecolor="black", linewidth=0.35)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.02, color=COLOR_MARGIN, linewidth=0.8, linestyle=":", label="Margen 0.02")
    ax.axhline(-0.02, color=COLOR_MARGIN, linewidth=0.8, linestyle=":")
    ax.set_ylabel(r"$F1_{macro}^{OVA\ reducido} - F1_{macro}^{multi\ ref.}$")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        borderaxespad=0,
        handlelength=1.4,
    )
    save_figure(fig, output_dir, "fig_delta_f1_ova_reducido_vs_multi_ref")

    summary = df[
        [
            "dataset",
            "comparacion",
            "multi_output_media",
            "ova_reducido_media",
            "diferencia_media",
            "wilcoxon_p",
            "tost_equivalente",
        ]
    ].copy()
    summary.to_csv(output_dir / "tabla_ova_reducido_vs_multi_ref.csv", index=False)
    return summary


def best_row(df: pd.DataFrame, dataset: str, model_type: str, architecture: str) -> pd.Series | None:
    subset = df[
        (df["dataset"] == dataset)
        & (df["model_type"] == model_type)
        & (df["architecture"] == architecture)
    ].copy()
    if subset.empty:
        return None
    return subset.sort_values("f1_macro", ascending=False).iloc[0]


def plot_same_architecture_reductions(comparisons_csv: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(comparisons_csv)
    pairs = [
        ("wine", "[16, 8]"),
        ("breast_cancer", "[16, 8]"),
        ("digits", "[16, 8]"),
        ("mnist", "[16, 32, 64]"),
        ("cifar10", "[16, 32, 64]"),
        ("brisc", "[24, 48, 96]"),
        ("tb_chest_xray", "[16, 32, 64]"),
    ]

    rows = []
    for dataset, architecture in pairs:
        multi = best_row(df, dataset, "multi-output", architecture)
        ova = best_row(df, dataset, "OVA", architecture)
        if multi is None or ova is None:
            continue
        rows.append(
            {
                "dataset": dataset,
                "architecture": architecture,
                "multi_f1": float(multi["f1_macro"]),
                "ova_f1": float(ova["f1_macro"]),
                "delta_f1": float(ova["f1_macro"] - multi["f1_macro"]),
                "multi_source": multi["source_path"],
                "ova_source": ova["source_path"],
            }
        )

    out = ordered(pd.DataFrame(rows))
    labels = [f"{label_dataset(row.dataset)}\n{row.architecture}" for row in out.itertuples()]
    colors = np.where(out["delta_f1"] >= 0, COLOR_POSITIVE, COLOR_NEGATIVE)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(labels, out["delta_f1"], color=colors, edgecolor="black", linewidth=0.35)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(r"$F1_{macro}^{OVA} - F1_{macro}^{multi}$")
    save_figure(fig, output_dir, "fig_delta_f1_reducciones_misma_arquitectura")
    out.to_csv(output_dir / "tabla_reducciones_misma_arquitectura.csv", index=False)
    return out


def selected_practical_ova(comparisons_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(comparisons_csv)
    rows = []
    for spec in PRACTICAL_OVA_SELECTION:
        subset = df[
            (df["dataset"] == spec["dataset"])
            & (df["model_type"] == "OVA")
            & (df["architecture"] == spec["architecture"])
            & (df["batch_size"] == spec["batch_size"])
        ].copy()
        if spec["source_contains"]:
            hinted = subset[
                subset["source_path"].astype(str).str.contains(spec["source_contains"], regex=False)
                | subset["familia"].astype(str).str.contains(spec["source_contains"], regex=False)
            ]
            if not hinted.empty:
                subset = hinted
        if subset.empty:
            continue
        row = subset.sort_values("f1_macro", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": row["dataset"],
                "architecture": row["architecture"],
                "batch_size": int(row["batch_size"]),
                "patience": int(row["early_stopping_patience"]),
                "learning_rate": float(row["learning_rate"]),
                "epochs": int(row["epochs"]),
                "multi_ref_f1": float(row["ref_multi_f1"]),
                "ova_ref_f1": float(row["ref_ova_parallel_f1"]),
                "ova_practical_f1": float(row["f1_macro"]),
                "delta_f1": float(row["f1_macro"] - row["ref_multi_f1"]),
                "multi_ref_time_seconds": float(row["ref_multi_time_seconds"]),
                "ova_ref_seq_time_seconds": float(row["ref_ova_seq_time_seconds"]),
                "ova_ref_parallel_time_seconds": float(row["ref_ova_parallel_wall_time_seconds"]),
                "ova_practical_time_seconds": float(row["parallel_train_time_seconds"]),
                "time_ratio_vs_multi": float(row["parallel_train_time_seconds"] / row["ref_multi_time_seconds"]),
                "source_path": row["source_path"],
            }
        )
    return ordered(pd.DataFrame(rows))


def selected_ova_with_comparable_multi(comparisons_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(comparisons_csv)
    selected = selected_practical_ova(comparisons_csv)
    rows = []
    for row in selected.itertuples(index=False):
        multi = df[
            (df["dataset"] == row.dataset)
            & (df["model_type"] == "multi-output")
            & (df["architecture"] == row.architecture)
            & (df["batch_size"] == row.batch_size)
        ].copy()
        if multi.empty:
            continue
        # If there are several comparable multi-output runs, keep the strongest one.
        multi_row = multi.sort_values("f1_macro", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": row.dataset,
                "architecture": row.architecture,
                "batch_size": row.batch_size,
                "multi_reference_f1": float(row.multi_ref_f1),
                "multi_comparable_f1": float(multi_row["f1_macro"]),
                "ova_selected_f1": float(row.ova_practical_f1),
                "delta_f1": float(row.ova_practical_f1 - multi_row["f1_macro"]),
                "multi_comparable_time_seconds": float(multi_row["train_time_seconds"]),
                "ova_selected_time_seconds": float(row.ova_practical_time_seconds),
                "time_ratio_ova_vs_multi_comparable": float(
                    row.ova_practical_time_seconds / multi_row["train_time_seconds"]
                ),
                "multi_source": multi_row["source_path"],
                "ova_source": row.source_path,
            }
        )
    return ordered(pd.DataFrame(rows))


def plot_comparable_architecture_f1(comparisons_csv: Path, output_dir: Path) -> pd.DataFrame:
    df = selected_ova_with_comparable_multi(comparisons_csv)
    labels = [f"{label_dataset(row.dataset)}\n{row.architecture}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar(
        x - width,
        df["multi_reference_f1"],
        width,
        label="Multi ref.",
        color=COLOR_PAR,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x,
        df["multi_comparable_f1"],
        width,
        label="Multi equivalente",
        color=COLOR_MULTI,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x + width,
        df["ova_selected_f1"],
        width,
        label="OVA seleccionado",
        color=COLOR_OVA,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.set_ylabel(r"$F1_{macro}$")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        borderaxespad=0,
        handlelength=1.4,
        columnspacing=1.3,
    )
    save_figure(fig, output_dir, "fig_multi_equivalente_vs_ova_seleccionado_f1")
    df.to_csv(output_dir / "tabla_multi_equivalente_vs_ova_seleccionado.csv", index=False)
    return df


def plot_comparable_architecture_times(
    comparisons_csv: Path,
    output_dir: Path,
    dataset_filter: list[str],
    output_name: str,
) -> pd.DataFrame:
    df = selected_ova_with_comparable_multi(comparisons_csv)
    df = df[df["dataset"].isin(dataset_filter)].copy()
    labels = [f"{label_dataset(row.dataset)}\n{row.architecture}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.bar(
        x - width / 2,
        df["multi_comparable_time_seconds"],
        width,
        label="Multi equivalente",
        color=COLOR_MULTI,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x + width / 2,
        df["ova_selected_time_seconds"],
        width,
        label="OVA seleccionado par.",
        color=COLOR_OVA,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.set_ylabel("Tiempo paralelo efectivo (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        borderaxespad=0,
        handlelength=1.4,
        columnspacing=1.3,
    )
    save_figure(fig, output_dir, output_name)
    df.to_csv(output_dir / f"tabla_{output_name.replace('fig_', '')}.csv", index=False)
    return df


def plot_practical_ova_f1(comparisons_csv: Path, output_dir: Path) -> pd.DataFrame:
    df = selected_practical_ova(comparisons_csv)
    labels = [label_practical_config(row) for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.bar(
        x - width,
        df["multi_ref_f1"],
        width,
        label="Multi-salida ref.",
        color=COLOR_MULTI,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x,
        df["ova_ref_f1"],
        width,
        label="OVA ref.",
        color=COLOR_PAR,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x + width,
        df["ova_practical_f1"],
        width,
        label="OVA selec.",
        color=COLOR_OVA,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.set_ylabel(r"$F1_{macro}$")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        borderaxespad=0,
        handlelength=1.4,
        columnspacing=1.1,
    )
    save_figure(fig, output_dir, "fig_ova_practico_f1")
    df.to_csv(output_dir / "tabla_ova_practico_f1_tiempo.csv", index=False)
    return df


def plot_practical_ova_times(
    comparisons_csv: Path,
    output_dir: Path,
    dataset_filter: list[str],
    output_name: str,
) -> pd.DataFrame:
    df = selected_practical_ova(comparisons_csv)
    df = df[df["dataset"].isin(dataset_filter)].copy()
    labels = [label_dataset(dataset) for dataset in df["dataset"]]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.bar(
        x - width,
        df["multi_ref_time_seconds"],
        width,
        label="Multi-salida ref.",
        color=COLOR_MULTI,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x,
        df["ova_ref_parallel_time_seconds"],
        width,
        label="OVA ref. par.",
        color=COLOR_SEQ,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x + width,
        df["ova_practical_time_seconds"],
        width,
        label="OVA selec. par.",
        color=COLOR_PAR,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.set_ylabel("Tiempo paralelo efectivo (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0,
        handlelength=1.4,
        columnspacing=1.1,
    )
    save_figure(fig, output_dir, output_name)
    df.to_csv(output_dir / f"tabla_{output_name.replace('fig_', '')}.csv", index=False)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera figuras de resultados para el artículo CAEPIA.")
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("resultados_actualizados/analisis_arquitecturas_ova/referencias_secuencial_paralelo.csv"),
    )
    parser.add_argument(
        "--comparisons-csv",
        type=Path,
        default=Path("resultados_actualizados/analisis_arquitecturas_ova/comparaciones_pruebas_vs_referencias.csv"),
    )
    parser.add_argument(
        "--reduced-stats-csv",
        type=Path,
        default=Path("resultados_estadisticos/test_arquitecturas_reducidas.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figuras_caepia"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    if args.reference_csv.exists():
        plot_reference_f1_grouped(args.reference_csv, args.output_dir)
        plot_reference_delta(args.reference_csv, args.output_dir)
        plot_reference_times_grouped(args.reference_csv, args.output_dir)
        plot_reference_times_grouped(
            args.reference_csv,
            args.output_dir,
            dataset_filter=["iris", "wine", "breast_cancer", "digits"],
            output_name="fig_tiempos_referencia_clasicos",
        )
        plot_reference_times_grouped(
            args.reference_csv,
            args.output_dir,
            dataset_filter=["mnist", "cifar10", "brisc", "tb_chest_xray"],
            output_name="fig_tiempos_referencia_imagenes",
        )
        plot_time_ratios(args.reference_csv, args.output_dir)
        generated.extend(
            [
                "fig_f1_referencia_multi_vs_ova",
                "fig_delta_f1_referencia",
                "fig_tiempos_referencia_multi_ova_seq_ova_par",
                "fig_tiempos_referencia_clasicos",
                "fig_tiempos_referencia_imagenes",
                "fig_ratios_tiempo_referencia",
            ]
        )
    else:
        print(f"No existe {args.reference_csv}")

    if args.reduced_stats_csv.exists():
        plot_reduced_vs_reference_delta(args.reduced_stats_csv, args.output_dir)
        generated.append("fig_delta_f1_ova_reducido_vs_multi_ref")
    else:
        print(f"No existe {args.reduced_stats_csv}")

    if args.comparisons_csv.exists():
        plot_same_architecture_reductions(args.comparisons_csv, args.output_dir)
        plot_practical_ova_f1(args.comparisons_csv, args.output_dir)
        plot_practical_ova_times(
            args.comparisons_csv,
            args.output_dir,
            dataset_filter=["iris", "wine", "breast_cancer", "digits"],
            output_name="fig_ova_practico_tiempos_clasicos",
        )
        plot_practical_ova_times(
            args.comparisons_csv,
            args.output_dir,
            dataset_filter=["mnist", "cifar10", "brisc", "tb_chest_xray"],
            output_name="fig_ova_practico_tiempos_imagenes",
        )
        plot_comparable_architecture_f1(args.comparisons_csv, args.output_dir)
        plot_comparable_architecture_times(
            args.comparisons_csv,
            args.output_dir,
            dataset_filter=["iris", "wine", "breast_cancer", "digits"],
            output_name="fig_multi_equivalente_vs_ova_seleccionado_tiempos_clasicos",
        )
        plot_comparable_architecture_times(
            args.comparisons_csv,
            args.output_dir,
            dataset_filter=["mnist", "cifar10", "brisc", "tb_chest_xray"],
            output_name="fig_multi_equivalente_vs_ova_seleccionado_tiempos_imagenes",
        )
        generated.extend(
            [
                "fig_delta_f1_reducciones_misma_arquitectura",
                "fig_ova_practico_f1",
                "fig_ova_practico_tiempos_clasicos",
                "fig_ova_practico_tiempos_imagenes",
                "fig_multi_equivalente_vs_ova_seleccionado_f1",
                "fig_multi_equivalente_vs_ova_seleccionado_tiempos_clasicos",
                "fig_multi_equivalente_vs_ova_seleccionado_tiempos_imagenes",
            ]
        )
    else:
        print(f"No existe {args.comparisons_csv}")

    print("Figuras generadas en:", args.output_dir)
    for name in generated:
        print(f"- {name}.pdf")
        print(f"- {name}.png")


if __name__ == "__main__":
    main()
