from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from configuraciones_seleccionadas_utils import (
    DATASETS_POR_DEFECTO,
    DETAIL_CSV,
    cargar_detalle,
    diferencias_configuracion_seleccionada,
    potencia_objetivo_por_dataset,
)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analisis de potencia Wilcoxon para las configuraciones One-vs-All "
            "seleccionadas frente al multi-output de referencia."
        )
    )
    parser.add_argument("--csv", type=Path, default=DETAIL_CSV)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metrica", default="f1_macro")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--potencia", type=float, default=None)
    parser.add_argument("--dif-relevante", type=float, default=0.02)
    parser.add_argument("--min-semillas", type=int, default=3)
    parser.add_argument("--max-semillas", type=int, default=100)
    parser.add_argument("--simulaciones", type=int, default=10000)
    parser.add_argument("--semilla-rng", type=int, default=12345)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS_POR_DEFECTO),
        help="Datasets a analizar. Por defecto: todos los seleccionados.",
    )
    return parser


def simular_potencia_wilcoxon(
    difs_piloto: pd.Series,
    n_semillas: int,
    dif_relevante: float,
    alpha: float,
    simulaciones: int,
    rng: np.random.Generator,
) -> float:
    ruido_piloto = difs_piloto.to_numpy(dtype=float)
    ruido_piloto = ruido_piloto - np.median(ruido_piloto)

    rechazos = 0
    for _ in range(simulaciones):
        muestra = rng.choice(ruido_piloto, size=n_semillas, replace=True) + dif_relevante
        try:
            resultado = wilcoxon(muestra, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            continue
        if resultado.pvalue < alpha:
            rechazos += 1

    return rechazos / simulaciones


def buscar_semillas_necesarias(
    difs_piloto: pd.Series,
    dif_relevante: float,
    alpha: float,
    potencia_objetivo: float,
    min_semillas: int,
    max_semillas: int,
    simulaciones: int,
    rng: np.random.Generator,
) -> tuple[int | None, float]:
    ultima_potencia = 0.0
    for n_semillas in range(min_semillas, max_semillas + 1):
        ultima_potencia = simular_potencia_wilcoxon(
            difs_piloto=difs_piloto,
            n_semillas=n_semillas,
            dif_relevante=dif_relevante,
            alpha=alpha,
            simulaciones=simulaciones,
            rng=rng,
        )
        if ultima_potencia >= potencia_objetivo:
            return n_semillas, ultima_potencia
    return None, ultima_potencia


def main() -> None:
    args = crear_parser().parse_args()
    if args.min_semillas < 2:
        raise ValueError("--min-semillas debe ser al menos 2.")
    if args.max_semillas < args.min_semillas:
        raise ValueError("--max-semillas debe ser mayor o igual que --min-semillas.")
    if args.simulaciones <= 0:
        raise ValueError("--simulaciones debe ser positivo.")

    detail = cargar_detalle(args.csv)
    rng = np.random.default_rng(args.semilla_rng)
    filas = []

    for dataset in args.datasets:
        comparacion, difs = diferencias_configuracion_seleccionada(
            detail,
            dataset,
            args.metrica,
        )
        potencia_objetivo = potencia_objetivo_por_dataset(dataset, args.potencia)
        semillas_necesarias, potencia_estimada = buscar_semillas_necesarias(
            difs_piloto=difs,
            dif_relevante=args.dif_relevante,
            alpha=args.alpha,
            potencia_objetivo=potencia_objetivo,
            min_semillas=args.min_semillas,
            max_semillas=args.max_semillas,
            simulaciones=args.simulaciones,
            rng=rng,
        )
        filas.append(
            {
                "dataset": dataset,
                "comparacion": f"{comparacion} - multi-output referencia",
                "metrica": args.metrica,
                "dif_relevante_simulada": args.dif_relevante,
                "alpha": args.alpha,
                "potencia_objetivo": potencia_objetivo,
                "semillas_necesarias": semillas_necesarias
                if semillas_necesarias is not None
                else f">{args.max_semillas}",
                "potencia_estimada": potencia_estimada,
            }
        )

    resultado = pd.DataFrame(filas)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        resultado.to_csv(args.output, index=False)

    print(resultado.to_string(index=False, float_format=lambda valor: f"{valor:.6f}"))
    print()
    print(
        "Interpretacion: semillas_necesarias es el menor numero de semillas pareadas "
        "para el que la simulacion Monte Carlo alcanza la potencia objetivo aplicando "
        f"Wilcoxon con alpha={args.alpha:g}."
    )
    if args.output is not None:
        print(f"CSV guardado en: {args.output}")


if __name__ == "__main__":
    main()
