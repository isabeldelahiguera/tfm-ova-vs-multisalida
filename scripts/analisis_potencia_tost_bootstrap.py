from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap




DATASETS_POR_DEFECTO = ("iris", "wine", "breast_cancer", "digits", "mnist", "cifar10", "brisc", "tuberculosis")
DATASETS_MEDICOS = {"brisc", "tb_chest_xray", "tuberculosis"}
CSV_POR_DEFECTO = {
    "iris": "resultados_actualizados/secuencial/exp_iris_mlp_138892.csv",
    "wine": "resultados_actualizados/secuencial/exp_wine_mlp_138892.csv",
    "breast_cancer": "resultados_actualizados/secuencial/exp_breast_cancer_mlp_138892.csv",
    "digits": "resultados_actualizados/secuencial/exp_digits_mlp_138892.csv",
    "mnist": "resultados_actualizados/secuencial/exp_mnist_vgg_138893.csv",
    "cifar10": "resultados_actualizados/secuencial/exp_cifar10_vgg_138894.csv",
    "brisc": "resultados_actualizados/secuencial/exp_brisc_vgg_128_139535.csv",
    "tuberculosis": "resultados_actualizados/secuencial/exp_tb_chest_xray_vgg_128_139536.csv",
}


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Análisis de potencia para equivalencia mediante TOST/bootstrap. "
            "Se estima cuántas semillas son necesarias para que el IC bootstrap "
            "del 90% de las diferencias quede dentro del margen de equivalencia."
        )
    )
    parser.add_argument(
        "csv",
        nargs="*",
        help=(
            "Ficheros CSV detallados del experimento. Si se omiten, se usan los CSV "
            "configurados en resultados_actualizados/secuencial."
        ),
    )
    parser.add_argument(
        "--metrica",
        "--metric",
        dest="metrica",
        default="f1_macro",
        help="Métrica usada para calcular las diferencias pareadas. Por defecto: f1_macro.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nivel de significacion del TOST. Con alpha=0.05 se usa IC bootstrap del 90%%.",
    )
    parser.add_argument(
        "--potencia",
        "--power",
        dest="potencia",
        type=float,
        default=None,
        help=(
            "Potencia objetivo. Si se omite, se usa 0.80 para datasets clasicos y 0.90 "
            "para datasets medicos."
        ),
    )
    parser.add_argument(
        "--margen",
        "--equivalence-margin",
        dest="margen",
        type=float,
        default=0.02,
        help="Margen de equivalencia absoluto para la metrica. Por defecto: 0.02.",
    )
    parser.add_argument(
        "--dif-verdadera",
        "--true-diff",
        dest="dif_verdadera",
        type=float,
        default=0.005,
        help=(
            "Diferencia verdadera simulada entre modelo_a - modelo_b. Debe estar dentro "
            "del margen si se quiere estimar potencia de equivalencia. Por defecto: 0.005."
        ),
    )
    parser.add_argument(
        "--modelo-a",
        "--model-a",
        dest="modelo_a",
        default="OVA",
        help="Primer modelo en la diferencia pareada modelo_a - modelo_b. Por defecto: OVA.",
    )
    parser.add_argument(
        "--modelo-b",
        "--model-b",
        dest="modelo_b",
        default="multi-output",
        help="Segundo modelo en la diferencia pareada modelo_a - modelo_b. Por defecto: multi-output.",
    )
    parser.add_argument(
        "--min-semillas",
        "--min-seeds",
        dest="min_semillas",
        type=int,
        default=3,
        help="Número mínimo de semillas pareadas que se prueba. Por defecto: 3.",
    )
    parser.add_argument(
        "--max-semillas",
        "--max-seeds",
        dest="max_semillas",
        type=int,
        default=100,
        help="Número máximo de semillas pareadas que se prueba. Por defecto: 100.",
    )
    parser.add_argument(
        "--simulaciones",
        "--simulations",
        dest="simulaciones",
        type=int,
        default=2000,
        help="Número de experimentos simulados para cada n de semillas. Por defecto: 2000.",
    )
    parser.add_argument(
        "--bootstrap-replicas",
        "--bootstrap-replicates",
        dest="bootstrap_replicas",
        type=int,
        default=1000,
        help="Número de remuestreos bootstrap para calcular cada IC. Por defecto: 1000.",
    )
    parser.add_argument(
        "--semilla-rng",
        "--rng-seed",
        dest="semilla_rng",
        type=int,
        default=12345,
        help="Semilla del generador aleatorio para reproducibilidad. Por defecto: 12345.",
    )
    return parser


def rutas_csv_por_defecto() -> list[Path]:
    rutas = []
    for dataset in DATASETS_POR_DEFECTO:
        ruta = Path(CSV_POR_DEFECTO[dataset])
        if ruta.exists():
            rutas.append(ruta)
    return rutas


def nombre_dataset(df: pd.DataFrame, ruta_csv: Path) -> str:
    if "dataset" in df.columns and not df["dataset"].dropna().empty:
        return str(df["dataset"].dropna().iloc[0])
    return ruta_csv.stem


def potencia_objetivo_por_dataset(dataset: str, potencia_forzada: float | None) -> float:
    if potencia_forzada is not None:
        return potencia_forzada
    dataset_normalizado = dataset.lower()
    return 0.90 if dataset_normalizado in DATASETS_MEDICOS else 0.80


def diferencias_pareadas(
    df: pd.DataFrame,
    metrica: str,
    modelo_a: str,
    modelo_b: str,
) -> pd.Series:
    columnas_necesarias = {"seed", "model_type", metrica}
    columnas_ausentes = columnas_necesarias - set(df.columns) # devuelve set de columnas necesarias que no están en el DataFrame
    if columnas_ausentes:
        # si hay columnas necesarias que faltan, se lanza un error indicando cuáles son
        raise ValueError(f"Faltan columnas necesarias: {sorted(columnas_ausentes)}")

    subconjunto = df[df["model_type"].isin([modelo_a, modelo_b])].copy()
    if subconjunto.empty:
        raise ValueError("No se han encontrado filas para los tipos de modelo solicitados.")

    # Se pivota el DataFrame para tener una fila por semilla y una columna por modelo, con los valores de la métrica.
    ancho = subconjunto.pivot_table(index="seed", columns="model_type", values=metrica, aggfunc="mean")

    if modelo_a not in ancho.columns or modelo_b not in ancho.columns:
        raise ValueError(f"Deben estar presentes tanto {modelo_a!r} como {modelo_b!r}.")

    pareado = ancho[[modelo_a, modelo_b]].dropna().sort_index() # el índice es "seed"
    if pareado.empty:
        raise ValueError("No hay semillas pareadas disponibles tras alinear ambos modelos.")

    return pareado[modelo_a] - pareado[modelo_b]

def intervalo_bootstrap_mediana(
    muestra: np.ndarray,
    alpha: float,
    bootstrap_replicas: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    resultado = bootstrap(
        (muestra,),
        np.median,
        confidence_level=1 - 2 * alpha,
        n_resamples=bootstrap_replicas,
        method="percentile",
        random_state=rng,
    )
    return float(resultado.confidence_interval.low), float(resultado.confidence_interval.high)


def simular_potencia_tost_bootstrap(
    difs_piloto: pd.Series,
    n_semillas: int,
    margen: float,
    dif_verdadera: float,
    alpha: float,
    simulaciones: int,
    bootstrap_replicas: int,
    rng: np.random.Generator,
) -> float:
    difs_centradas = difs_piloto.to_numpy(dtype=float)
    difs_centradas = difs_centradas - np.median(difs_centradas) # centramos la muestra

    equivalencias = 0
    for _ in range(simulaciones):
        muestra = rng.choice(difs_centradas, size=n_semillas, replace=True) + dif_verdadera
        ic_bajo, ic_alto = intervalo_bootstrap_mediana(
            muestra=muestra,
            alpha=alpha,
            bootstrap_replicas=bootstrap_replicas,
            rng=rng,
        )
        # Se declara equivalencia si el IC bootstrap queda completamente dentro del margen [-margen, margen]
        if ic_bajo > -margen and ic_alto < margen:
            equivalencias += 1

    return equivalencias / simulaciones


def buscar_semillas_necesarias(
    difs_piloto: pd.Series,
    margen: float,
    dif_verdadera: float,
    alpha: float,
    potencia_objetivo: float,
    min_semillas: int,
    max_semillas: int,
    simulaciones: int,
    bootstrap_replicas: int,
    rng: np.random.Generator,
) -> tuple[int | None, float]:
    ultima_potencia = 0.0
    for n_semillas in range(min_semillas, max_semillas + 1):
        potencia_estimada = simular_potencia_tost_bootstrap(
            difs_piloto=difs_piloto,
            n_semillas=n_semillas,
            margen=margen,
            dif_verdadera=dif_verdadera,
            alpha=alpha,
            simulaciones=simulaciones,
            bootstrap_replicas=bootstrap_replicas,
            rng=rng,
        )
        ultima_potencia = potencia_estimada
        if potencia_estimada >= potencia_objetivo:
            return n_semillas, potencia_estimada

    return None, ultima_potencia


def main() -> None:
    args = crear_parser().parse_args()
    rutas_csv = [Path(ruta) for ruta in args.csv] if args.csv else rutas_csv_por_defecto()

    if not rutas_csv:
        raise SystemExit(
            "No se han indicado CSV y no se han encontrado CSV por defecto en "
            "resultados_actualizados/secuencial."
        )
    if args.min_semillas < 2:
        raise ValueError("--min-semillas debe ser al menos 2.")
    if args.max_semillas < args.min_semillas:
        raise ValueError("--max-semillas debe ser mayor o igual que --min-semillas.")
    if args.simulaciones <= 0:
        raise ValueError("--simulaciones debe ser positivo.")
    if args.bootstrap_replicas <= 0:
        raise ValueError("--bootstrap-replicas debe ser positivo.")
    if args.margen <= 0:
        raise ValueError("--margen debe ser positivo.")
    if abs(args.dif_verdadera) >= args.margen:
        raise ValueError("--dif-verdadera debe estar estrictamente dentro del margen de equivalencia.")

    rng = np.random.default_rng(args.semilla_rng)
    filas = []

    for ruta_csv in rutas_csv:
        df = pd.read_csv(ruta_csv)
        difs = diferencias_pareadas(df, args.metrica, args.modelo_a, args.modelo_b)
        dataset = nombre_dataset(df, ruta_csv)
        potencia_objetivo = potencia_objetivo_por_dataset(dataset, args.potencia)
        semillas_necesarias, potencia_estimada = buscar_semillas_necesarias(
            difs_piloto=difs,
            margen=args.margen,
            dif_verdadera=args.dif_verdadera,
            alpha=args.alpha,
            potencia_objetivo=potencia_objetivo,
            min_semillas=args.min_semillas,
            max_semillas=args.max_semillas,
            simulaciones=args.simulaciones,
            bootstrap_replicas=args.bootstrap_replicas,
            rng=rng,
        )
        filas.append(
            {
                "dataset": dataset,
                "metrica": args.metrica,
                "margen_equivalencia": args.margen,
                "dif_verdadera_simulada": args.dif_verdadera,
                "alpha": args.alpha,
                "potencia_objetivo": potencia_objetivo,
                "semillas_necesarias": semillas_necesarias if semillas_necesarias is not None else f">{args.max_semillas}",
                "potencia_estimada": potencia_estimada,
            }
        )

    resultado = pd.DataFrame(filas)
    columnas_mostradas = [
        "dataset",
        "metrica",
        "margen_equivalencia",
        "dif_verdadera_simulada",
        "alpha",
        "potencia_objetivo",
        "semillas_necesarias",
        "potencia_estimada",
    ]
    print(resultado[columnas_mostradas].to_string(index=False, float_format=lambda valor: f"{valor:.6f}"))
    print()
    print(
        "Interpretacion: semillas_necesarias es el menor numero de semillas pareadas para el que "
        "la simulacion Monte Carlo alcanza la potencia objetivo declarando equivalencia."
    )
    print(
        f"Equivalencia: el IC bootstrap del {100 * (1 - 2 * args.alpha):.0f}% de la mediana de "
        f"las diferencias debe quedar dentro de +/-{args.margen:g}. "
        f"La diferencia verdadera simulada es {args.dif_verdadera:g}."
    )


if __name__ == "__main__":
    main()
