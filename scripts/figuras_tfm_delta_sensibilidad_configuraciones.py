from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "Memoria TFM" / "img"
SOURCE = (
    ROOT
    / "resultados_actualizados"
    / "analisis_arquitecturas_ova"
    / "comparaciones_pruebas_vs_referencias.csv"
)

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


def comparable_configurations(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "dataset",
        "model_type",
        "architecture",
        "batch_size",
        "learning_rate",
        "early_stopping_patience",
        "epochs",
    ]
    value_cols = [
        "f1_macro",
        "ref_multi_f1",
        "ref_ova_seq_f1",
        "source_path",
    ]
    out = df.copy()
    out = out[~out["familia"].astype(str).str.contains("ampliado", na=False)]
    out = out[~out["familia"].astype(str).str.contains("referencia", na=False)]
    out = out[key_cols + value_cols].copy()

    # The same OVA configuration may appear as sequential and parallel. For this
    # sensitivity plot the predictive value is the same, so keep one row.
    out = out.sort_values("source_path").drop_duplicates(subset=key_cols)

    piv = out.pivot_table(
        index=[
            "dataset",
            "architecture",
            "batch_size",
            "learning_rate",
            "early_stopping_patience",
            "epochs",
        ],
        columns="model_type",
        values=["f1_macro", "ref_multi_f1", "ref_ova_seq_f1"],
        aggfunc="first",
    )
    piv.columns = ["_".join(col).strip() for col in piv.columns.to_flat_index()]
    piv = piv.reset_index()

    piv = piv.dropna(subset=["f1_macro_multi-output", "f1_macro_OVA"])
    piv["delta_multi"] = piv["f1_macro_multi-output"] - piv["ref_multi_f1_multi-output"]
    piv["delta_ova"] = piv["f1_macro_OVA"] - piv["ref_ova_seq_f1_OVA"]

    return piv


def summary_rows(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in paired.groupby("dataset", sort=False):
        rows.append(
            {
                "dataset": dataset,
                "n_configuraciones": int(len(group)),
                "delta_multi_medio": float(group["delta_multi"].mean()),
                "delta_ova_medio": float(group["delta_ova"].mean()),
                "delta_multi_min": float(group["delta_multi"].min()),
                "delta_multi_max": float(group["delta_multi"].max()),
                "delta_ova_min": float(group["delta_ova"].min()),
                "delta_ova_max": float(group["delta_ova"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_group(summary: pd.DataFrame, datasets: list[str], filename: str) -> None:
    subset = summary.set_index("dataset").loc[datasets].reset_index()
    x = np.arange(len(subset))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        x - width / 2,
        subset["delta_multi_medio"],
        width,
        label="Multi-salida",
        color=COLOR_MULTI,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        subset["delta_ova_medio"],
        width,
        label="One-vs-All",
        color=COLOR_OVA,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in subset["dataset"]], rotation=28, ha="right")
    ax.set_ylabel(r"$\Delta F1_{macro}$ medio respecto a referencia")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False)

    ymin = min(subset["delta_multi_medio"].min(), subset["delta_ova_medio"].min(), 0)
    ymax = max(subset["delta_multi_medio"].max(), subset["delta_ova_medio"].max(), 0)
    margin = max(0.005, (ymax - ymin) * 0.25)
    ax.set_ylim(ymin - margin, ymax + margin)

    fig.tight_layout(pad=0.8)
    fig.savefig(IMG_DIR / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(IMG_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE)
    paired = comparable_configurations(df)
    summary = summary_rows(paired)

    paired.to_csv(IMG_DIR / "tabla_delta_configuraciones_pareadas.csv", index=False)
    summary.to_csv(IMG_DIR / "tabla_delta_medio_configuraciones.csv", index=False)

    plot_group(summary, CLASICOS, "fig_delta_medio_configuraciones_clasicos")
    plot_group(summary, IMAGEN, "fig_delta_medio_configuraciones_imagen")


if __name__ == "__main__":
    main()
