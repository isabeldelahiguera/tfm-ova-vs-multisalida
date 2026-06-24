from __future__ import annotations

from pathlib import Path

import pandas as pd


DETAIL_CSV = Path(
    "resultados_actualizados/analisis_arquitecturas_ova/"
    "arquitecturas_ova_10semillas_detalle.csv"
)

DATASETS_POR_DEFECTO = (
    "iris",
    "wine",
    "breast_cancer",
    "digits",
    "mnist",
    "cifar10",
    "tb_chest_xray",
    "brisc",
)
DATASETS_MEDICOS = {"brisc", "tb_chest_xray"}

CONFIGURACIONES_SELECCIONADAS = {
    "iris": {
        "label": "OVA [32,16] bs32",
        "source": "detail",
        "approach_architecture": "OVA [32, 16]",
    },
    "wine": {
        "label": "OVA [16,8] bs32",
        "source": "detail",
        "approach_architecture": "OVA [16, 8]",
    },
    "breast_cancer": {
        "label": "OVA [16,8] bs64",
        "source": Path(
            "resultados_actualizados/paralelo/ova_bc_parallel_h16_8_bs64/"
            "exp_breast_cancer_mlp_parallel_ova.csv"
        ),
    },
    "digits": {
        "label": "OVA [32,16] bs64",
        "source": Path(
            "resultados_actualizados/paralelo/ova_digits_mlp32_16_bs64_lr1e3_pat10_ep50/"
            "exp_digits_mlp_parallel_ova.csv"
        ),
    },
    "mnist": {
        "label": "OVA [32,64,128] bs128",
        "source": Path(
            "resultados_actualizados/paralelo/ova_vgg32_64_128_bs128_lr1e3_pat10_ep50/"
            "exp_mnist_vgg_parallel_ova.csv"
        ),
    },
    "cifar10": {
        "label": "OVA [32,64,128] bs128",
        "source": Path(
            "resultados_actualizados/paralelo/ova_vgg32_64_128_bs128_lr1e3_pat10_ep50/"
            "exp_cifar10_vgg_parallel_ova.csv"
        ),
    },
    "tb_chest_xray": {
        "label": "OVA [16,32,64] bs32",
        "source": Path(
            "resultados_actualizados/paralelo/ova_parallel_reducida_vgg_16_32_64/"
            "exp_tb_chest_xray_vgg_parallel_ova.csv"
        ),
    },
    "brisc": {
        "label": "OVA [32,64,128] bs32",
        "source": "detail",
        "approach_architecture": "OVA [32, 64, 128]",
    },
}


def potencia_objetivo_por_dataset(dataset: str, potencia_forzada: float | None) -> float:
    if potencia_forzada is not None:
        return potencia_forzada
    return 0.90 if dataset.lower() in DATASETS_MEDICOS else 0.80


def cargar_detalle(detail_csv: Path = DETAIL_CSV) -> pd.DataFrame:
    if not detail_csv.exists():
        raise FileNotFoundError(detail_csv)
    return pd.read_csv(detail_csv)


def multi_reference(detail: pd.DataFrame, dataset: str, metrica: str) -> pd.DataFrame:
    columnas_necesarias = {"dataset", "seed", "model_type", metrica}
    columnas_ausentes = columnas_necesarias - set(detail.columns)
    if columnas_ausentes:
        raise ValueError(f"Faltan columnas necesarias: {sorted(columnas_ausentes)}")

    subset = detail[(detail["dataset"] == dataset) & (detail["model_type"] == "multi-output")]
    if subset.empty:
        raise ValueError(f"No hay multi-output de referencia para {dataset}")

    return subset[["seed", metrica]].rename(columns={metrica: "multi-output referencia"})


def ova_from_detail(
    detail: pd.DataFrame,
    dataset: str,
    approach_architecture: str,
    metrica: str,
) -> pd.DataFrame:
    columnas_necesarias = {"dataset", "seed", "approach_architecture", metrica}
    columnas_ausentes = columnas_necesarias - set(detail.columns)
    if columnas_ausentes:
        raise ValueError(f"Faltan columnas necesarias: {sorted(columnas_ausentes)}")

    subset = detail[
        (detail["dataset"] == dataset)
        & (detail["approach_architecture"] == approach_architecture)
    ]
    if subset.empty:
        raise ValueError(
            f"No hay filas para dataset={dataset} y arquitectura={approach_architecture}"
        )

    return subset[["seed", metrica]].rename(columns={metrica: "OVA seleccionada"})


def ova_from_file(path: Path, metrica: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    columnas_necesarias = {"seed", metrica}
    columnas_ausentes = columnas_necesarias - set(df.columns)
    if columnas_ausentes:
        raise ValueError(f"Faltan columnas necesarias en {path}: {sorted(columnas_ausentes)}")

    subset = df[pd.to_numeric(df["seed"], errors="coerce").notna()].copy()
    subset["seed"] = subset["seed"].astype(int)
    return subset[["seed", metrica]].rename(columns={metrica: "OVA seleccionada"})


def diferencias_configuracion_seleccionada(
    detail: pd.DataFrame,
    dataset: str,
    metrica: str,
) -> tuple[str, pd.Series]:
    spec = CONFIGURACIONES_SELECCIONADAS[dataset]
    multi = multi_reference(detail, dataset, metrica)
    source = spec["source"]

    if source == "detail":
        ova = ova_from_detail(
            detail,
            dataset,
            str(spec["approach_architecture"]),
            metrica,
        )
    else:
        ova = ova_from_file(Path(source), metrica)

    paired = multi.merge(ova, on="seed", how="inner").sort_values("seed")
    if paired.empty:
        raise ValueError(f"No hay semillas pareadas para {dataset}")

    difs = paired["OVA seleccionada"] - paired["multi-output referencia"]
    return str(spec["label"]), difs
