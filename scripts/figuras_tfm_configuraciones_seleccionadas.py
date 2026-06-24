from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "Memoria TFM" / "img"

DETAIL_CSV = (
    ROOT
    / "resultados_actualizados"
    / "analisis_arquitecturas_ova"
    / "arquitecturas_ova_configuraciones_seleccionadas_detalle.csv"
)

EXPANDED = {
    "iris": ROOT / "resultados_actualizados" / "ampliados" / "exp_iris_mlp_139551.csv",
    "brisc": ROOT / "resultados_actualizados" / "ampliados" / "exp_brisc_vgg_128_139552.csv",
    "wine": (
        ROOT
        / "resultados_actualizados"
        / "configuraciones_seleccionadas"
        / "exp_wine_multi32_16_vs_ova16_8_bs32_28semillas.csv"
    ),
}

OVA_PARALLEL_FILES = {
    "iris": (
        ROOT
        / "resultados_actualizados"
        / "paralelo"
        / "ova_parallel_actual"
        / "exp_iris_mlp_parallel_ova.csv"
    ),
    "wine": (
        ROOT
        / "resultados_actualizados"
        / "paralelo"
        / "ova_parallel_reducida_mlp_16_8"
        / "exp_wine_mlp_parallel_ova.csv"
    ),
    "breast_cancer": (
        ROOT
        / "resultados_actualizados"
        / "paralelo"
        / "ova_bc_parallel_h16_8_bs64"
        / "exp_breast_cancer_mlp_parallel_ova.csv"
    ),
    "digits": (
        ROOT
        / "resultados_actualizados"
        / "paralelo"
        / "ova_digits_mlp32_16_bs64_lr1e3_pat10_ep50"
        / "exp_digits_mlp_parallel_ova.csv"
    ),
    "mnist": (
        ROOT
        / "resultados_actualizados"
        / "paralelo"
        / "ova_vgg32_64_128_bs128_lr1e3_pat10_ep50"
        / "exp_mnist_vgg_parallel_ova.csv"
    ),
    "cifar10": (
        ROOT
        / "resultados_actualizados"
        / "paralelo"
        / "ova_vgg32_64_128_bs128_lr1e3_pat10_ep50"
        / "exp_cifar10_vgg_parallel_ova.csv"
    ),
    "brisc": (
        ROOT
        / "resultados_actualizados"
        / "paralelo"
        / "ova_parallel_actual"
        / "exp_brisc_vgg_parallel_ova.csv"
    ),
    "tb_chest_xray": (
        ROOT
        / "resultados_actualizados"
        / "paralelo"
        / "ova_parallel_reducida_vgg_16_32_64"
        / "exp_tb_chest_xray_vgg_parallel_ova.csv"
    ),
}

SELECTED_APPROACH = {
    "iris": "OVA [32, 16]",
    "wine": "OVA [16, 8]",
    "breast_cancer": "OVA [16, 8]",
    "digits": "OVA [32, 16]",
    "mnist": "OVA [32, 64, 128]",
    "cifar10": "OVA [32, 64, 128]",
    "brisc": "OVA [32, 64, 128]",
    "tb_chest_xray": "OVA [16, 32, 64]",
}

MULTI_APPROACH = {
    "iris": "multi-output [32, 16]",
    "wine": "multi-output [32, 16]",
    "breast_cancer": "multi-output [32, 16]",
    "digits": "multi-output [32, 16]",
    "mnist": "multi-output [32, 64, 128]",
    "cifar10": "multi-output [32, 64, 128]",
    "brisc": "multi-output [32, 64, 128]",
    "tb_chest_xray": "multi-output [32, 64, 128]",
}

DATASET_LABELS = {
    "iris": "Iris",
    "wine": "Wine",
    "breast_cancer": "Breast Cancer",
    "digits": "Digits",
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "brisc": "BRISC",
    "tb_chest_xray": "TB X-ray",
}

CLASICOS = ["iris", "wine", "breast_cancer", "digits"]
IMAGEN = ["mnist", "cifar10", "brisc", "tb_chest_xray"]

COLOR_MULTI = "#5C528E"
COLOR_OVA = "#D0ADCE"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def numeric_seeds(df: pd.DataFrame) -> pd.DataFrame:
    out = df[pd.to_numeric(df["seed"], errors="coerce").notna()].copy()
    out["seed"] = out["seed"].astype(int)
    return out


def stats(values: pd.Series) -> tuple[float, float, int]:
    return float(values.mean()), float(values.std(ddof=1)), int(values.count())


def f1_rows() -> pd.DataFrame:
    detail = numeric_seeds(pd.read_csv(DETAIL_CSV))
    detail = detail[detail["seed"].between(1, 10)].copy()
    rows = []
    for dataset in CLASICOS + IMAGEN:
        multi = detail[
            detail["dataset"].eq(dataset)
            & detail["model_type"].eq("multi-output")
            & detail["approach_architecture"].eq(MULTI_APPROACH[dataset])
        ]["f1_macro"]
        ova_ref = detail[
            detail["dataset"].eq(dataset)
            & detail["model_type"].eq("OVA")
            & detail["approach_architecture"].eq(MULTI_APPROACH[dataset].replace("multi-output", "OVA"))
        ]["f1_macro"]
        ova_sel_source = numeric_seeds(pd.read_csv(OVA_PARALLEL_FILES[dataset]))
        ova_sel_source = ova_sel_source[ova_sel_source["seed"].between(1, 10)]
        ova_sel = ova_sel_source["f1_macro"]

        multi_mean, multi_std, multi_n = stats(multi)
        ova_ref_mean, ova_ref_std, ova_ref_n = stats(ova_ref)
        ova_sel_mean, ova_sel_std, ova_sel_n = stats(ova_sel)
        rows.append(
            {
                "dataset": dataset,
                "multi_mean": multi_mean,
                "multi_std": multi_std,
                "multi_n": multi_n,
                "ova_ref_mean": ova_ref_mean,
                "ova_ref_std": ova_ref_std,
                "ova_ref_n": ova_ref_n,
                "ova_sel_mean": ova_sel_mean,
                "ova_sel_std": ova_sel_std,
                "ova_sel_n": ova_sel_n,
            }
        )
    return pd.DataFrame(rows)


def time_rows() -> pd.DataFrame:
    detail = numeric_seeds(pd.read_csv(DETAIL_CSV))
    detail = detail[detail["seed"].between(1, 10)].copy()
    rows = []
    for dataset in CLASICOS + IMAGEN:
        multi = detail[
            detail["dataset"].eq(dataset)
            & detail["model_type"].eq("multi-output")
            & detail["approach_architecture"].eq(MULTI_APPROACH[dataset])
        ]["train_time_seconds"]

        ova_ref_source = numeric_seeds(
            pd.read_csv(
                ROOT
                / "resultados_actualizados"
                / "paralelo"
                / "ova_parallel_actual"
                / f"exp_{dataset}_mlp_parallel_ova.csv"
                if dataset in CLASICOS
                else ROOT
                / "resultados_actualizados"
                / "paralelo"
                / "ova_parallel_actual"
                / f"exp_{dataset}_vgg_parallel_ova.csv"
            )
        )
        ova_ref_source = ova_ref_source[ova_ref_source["seed"].between(1, 10)]
        ova_ref = ova_ref_source["parallel_train_time_seconds"]

        ova_sel_source = numeric_seeds(pd.read_csv(OVA_PARALLEL_FILES[dataset]))
        ova_sel_source = ova_sel_source[ova_sel_source["seed"].between(1, 10)]
        ova_sel = ova_sel_source["parallel_train_time_seconds"]

        multi_mean, multi_std, multi_n = stats(multi)
        ova_ref_mean, ova_ref_std, ova_ref_n = stats(ova_ref)
        ova_sel_mean, ova_sel_std, ova_sel_n = stats(ova_sel)
        rows.append(
            {
                "dataset": dataset,
                "multi_mean": multi_mean,
                "multi_std": multi_std,
                "multi_n": multi_n,
                "ova_ref_mean": ova_ref_mean,
                "ova_ref_std": ova_ref_std,
                "ova_ref_n": ova_ref_n,
                "ova_sel_mean": ova_sel_mean,
                "ova_sel_std": ova_sel_std,
                "ova_sel_n": ova_sel_n,
            }
        )
    return pd.DataFrame(rows)


def plot_group(
    df: pd.DataFrame,
    datasets: list[str],
    ylabel: str,
    output_name: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    subset = df[df["dataset"].isin(datasets)].copy()
    subset["_order"] = subset["dataset"].map({d: i for i, d in enumerate(datasets)})
    subset = subset.sort_values("_order")

    x = np.arange(len(subset))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.9, 3.5))
    error_kw = {
        "elinewidth": 0.6,
        "ecolor": "#2F2F2F",
        "capsize": 2.2,
        "capthick": 0.6,
    }
    ax.bar(
        x - width,
        subset["multi_mean"],
        width,
        yerr=subset["multi_std"],
        error_kw=error_kw,
        label="Multi-salida ref.",
        color=COLOR_MULTI,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x,
        subset["ova_ref_mean"],
        width,
        yerr=subset["ova_ref_std"],
        error_kw=error_kw,
        label="OVA ref.",
        color="#8678A7",
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar(
        x + width,
        subset["ova_sel_mean"],
        width,
        yerr=subset["ova_sel_std"],
        error_kw=error_kw,
        label="OVA seleccionada",
        color=COLOR_OVA,
        edgecolor="black",
        linewidth=0.35,
    )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in subset["dataset"]], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0,
        handlelength=1.3,
        columnspacing=1.2,
    )
    fig.tight_layout(pad=0.7)
    for suffix in (".pdf", ".png"):
        fig.savefig(IMG_DIR / f"{output_name}{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    configure_style()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    f1 = f1_rows()
    times = time_rows()

    f1.to_csv(IMG_DIR / "tabla_ova_seleccionada_f1_resumen.csv", index=False)
    times.to_csv(IMG_DIR / "tabla_ova_seleccionada_tiempos_resumen.csv", index=False)

    plot_group(
        f1,
        CLASICOS,
        r"$F1_{macro}$",
        "fig_ova_seleccionada_f1_clasicos",
        ylim=(0, 1.05),
    )
    plot_group(
        f1,
        IMAGEN,
        r"$F1_{macro}$",
        "fig_ova_seleccionada_f1_imagen",
        ylim=(0, 1.05),
    )
    plot_group(
        times,
        CLASICOS,
        "Tiempo de entrenamiento (s)",
        "fig_ova_seleccionada_tiempo_clasicos",
    )
    plot_group(
        times,
        IMAGEN,
        "Tiempo de entrenamiento (s)",
        "fig_ova_seleccionada_tiempo_imagen",
    )

    print(f"Figuras guardadas en {IMG_DIR}")


if __name__ == "__main__":
    main()
