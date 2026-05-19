from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon



DATASETS_POR_DEFECTO = ("iris", "wine", "breast_cancer", "digits", "mnist", "cifar10", "brisc", "tuberculosis")
DATASETS_MEDICOS = {"brisc", "tb_chest_xray", "tuberculosis"}
CSV_POR_DEFECTO = {
    "iris": "resultados_10semillas/iris_mlp_10semillas.csv",
    "wine": "resultados_10semillas/wine_mlp_10semillas.csv",
    "breast_cancer": "resultados_10semillas/breast_cancer_mlp_10semillas.csv",
    "digits": "resultados_10semillas/digits_mlp_10semillas.csv",
    "mnist": "resultados_10semillas/mnist_vgg_10semillas.csv",
    "cifar10": "resultados_10semillas/cifar10_vgg_10semillas.csv",
    "brisc": "resultados_10semillas/brisc_vgg_128_10semillas.csv",
    "tuberculosis": "resultados_10semillas/tuberculosis_vgg_128_10semillas.csv",
}




def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Análisis de potencia para el test de Wilcoxon pareado mediante simulación de "
            "Monte Carlo. Se estiman cuántas semillas son necesarias para detectar una "
            "diferencia relevante entre los resultados de OVA y multi-output."
        )
    )
    parser.add_argument(
        "csv",
        nargs="*",
        help=(
            "Ficheros CSV detallados del experimento. Si se omiten, se usan los últimos "
            "CSV configurados en resultados_10semillas."
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
        help="Nivel de significación del test de Wilcoxon. Por defecto: 0.05.",
    )
    parser.add_argument(
        "--potencia",
        "--power",
        dest="potencia",
        type=float,
        default=None,
        help=(
            "Potencia objetivo. Si se omite, se usa 0.80 para datasets clásicos y 0.90 "
            "para datasets médicos."
        ),
    )
    parser.add_argument(
        "--dif-relevante",
        "--min-diff",
        dest="dif_relevante",
        type=float,
        default=0.02,
        help=(
            "Diferencia verdadera que se simula en la métrica, con signo según modelo_a - "
            "modelo_b. Por defecto: 0.02."
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
        default=10000,
        help="Número de experimentos simulados para cada n de semillas. Por defecto: 10000.",
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
        # Si la columna "dataset" existe y tiene al menos un valor no nulo, se usa ese valor como nombre del dataset.
        return str(df["dataset"].dropna().iloc[0]) 
    return ruta_csv.stem


def potencia_objetivo_por_dataset(dataset: str, potencia_forzada: float | None) -> float:
    if potencia_forzada is not None:
        return potencia_forzada # si se ha especificado una potencia por argumento
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


def simular_potencia_wilcoxon(
    difs_piloto: pd.Series,
    n_semillas: int,
    dif_relevante: float,
    alpha: float,
    alternativa: str,
    simulaciones: int,
    rng: np.random.Generator,
) -> float:
    ruido_piloto = difs_piloto.to_numpy(dtype=float)
    ruido_piloto = ruido_piloto - np.median(ruido_piloto) # datos centrados en cero (mediana = 0)

    rechazos = 0 # contador de simulaciones en las que se rechaza la hipótesis nula de diferencia cero
    for _ in range(simulaciones):
        muestra = rng.choice(ruido_piloto, size=n_semillas, replace=True) + dif_relevante # bootstrap de la muestra piloto, centrada en cero y con la diferencia relevante añadida
        try:
            resultado = wilcoxon(muestra, alternative=alternativa, zero_method="wilcox") # se ignoran las diferencias iguales a 0
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
    alternativa: str,
    min_semillas: int,
    max_semillas: int,
    simulaciones: int,
    rng: np.random.Generator,
) -> tuple[int | None, float]:
    mejor_potencia = 0.0
    for n_semillas in range(min_semillas, max_semillas + 1):
        potencia_estimada = simular_potencia_wilcoxon(
            difs_piloto=difs_piloto,
            n_semillas=n_semillas,
            dif_relevante=dif_relevante,
            alpha=alpha,
            alternativa=alternativa,
            simulaciones=simulaciones,
            rng=rng,
        )
        mejor_potencia = potencia_estimada
        if potencia_estimada >= potencia_objetivo:
            return n_semillas, potencia_estimada

    return None, mejor_potencia


def main() -> None:
    args = crear_parser().parse_args()
    rutas_csv = [Path(ruta) for ruta in args.csv] if args.csv else rutas_csv_por_defecto()

    if not rutas_csv:
        raise SystemExit("No se han indicado CSV y no se han encontrado CSV por defecto en resultados_10semillas.")
    if args.min_semillas < 2:
        raise ValueError("--min-semillas debe ser al menos 2.")
    if args.max_semillas < args.min_semillas:
        raise ValueError("--max-semillas debe ser mayor o igual que --min-semillas.")
    if args.simulaciones <= 0:
        raise ValueError("--simulaciones debe ser positivo.")

    rng = np.random.default_rng(args.semilla_rng)
    filas = []

    for ruta_csv in rutas_csv:
        df = pd.read_csv(ruta_csv)
        difs = diferencias_pareadas(df, args.metrica, args.modelo_a, args.modelo_b)
        dataset = nombre_dataset(df, ruta_csv)
        potencia_objetivo = potencia_objetivo_por_dataset(dataset, args.potencia)
        semillas_necesarias, potencia_estimada = buscar_semillas_necesarias(
            difs_piloto=difs,
            dif_relevante=args.dif_relevante,
            alpha=args.alpha,
            potencia_objetivo=potencia_objetivo,
            alternativa="two-sided",
            min_semillas=args.min_semillas,
            max_semillas=args.max_semillas,
            simulaciones=args.simulaciones,
            rng=rng,
        )
        filas.append(
            {
                "dataset": dataset,
                "metrica": args.metrica,
                "dif_relevante_simulada": args.dif_relevante,
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
        "dif_relevante_simulada",
        "alpha",
        "potencia_objetivo",
        "semillas_necesarias",
        "potencia_estimada",
    ]
    print(resultado[columnas_mostradas].to_string(index=False, float_format=lambda valor: f"{valor:.6f}"))
    print()
    print(
        "Interpretación: semillas_necesarias es el menor número de semillas pareadas para el que "
        "la simulación de Monte Carlo alcanza la potencia objetivo aplicando Wilcoxon con "
        f"alpha={args.alpha:g}."
    )


if __name__ == "__main__":
    main()
