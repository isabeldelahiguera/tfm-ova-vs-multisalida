from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd



CSV_POR_DEFECTO = [
    "resultados_10semillas/iris_mlp_22semillas.csv",
    "resultados_10semillas/wine_mlp_10semillas.csv",
    "resultados_10semillas/breast_cancer_mlp_10semillas.csv",
    "resultados_10semillas/digits_mlp_10semillas.csv",
    "resultados_10semillas/mnist_vgg_10semillas.csv",
    "resultados_10semillas/cifar10_vgg_10semillas.csv",
    "resultados_10semillas/brisc_vgg_128_24semillas.csv",
    "resultados_10semillas/tuberculosis_vgg_128_10semillas.csv",
]


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test de equivalencia TOST/bootstrap entre OVA y multi-output. "
            "Se usa un IC bootstrap de la mediana de las diferencias pareadas."
        )
    )
    parser.add_argument(
        "--csv",
        nargs="*",
        default=None,
        help=(
            "Ficheros CSV detallados del experimento. Si se omiten, se usan los CSV "
            "configurados en resultados_10semillas."
        ),
    )
    parser.add_argument(
        "--metrica",
        "--metric",
        dest="metrica",
        default="f1_macro",
        help="Columna de la métrica que se quiere comparar. Por defecto: f1_macro.",
    )
    parser.add_argument(
        "--margen",
        "--equivalence-margin",
        dest="margen",
        type=float,
        default=0.02,
        help="Margen de equivalencia absoluto. Por defecto: 0.02.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nivel de significancia. Con alpha=0.05 se usa IC bootstrap del 90%%.",
    )
    parser.add_argument(
        "--bootstrap-replicas",
        "--bootstrap-replicates",
        dest="bootstrap_replicas",
        type=int,
        default=10000,
        help="Número de remuestreos bootstrap para calcular el IC. Por defecto: 10000.",
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
    # Se generan índices aleatorios para las muestras bootstrap. Cada fila de 'indices' corresponde a una muestra bootstrap, y cada columna corresponde a un elemento de la muestra original.
    indices = rng.integers(0, len(muestra), size=(bootstrap_replicas, len(muestra)))
    medianas_bootstrap = np.median(muestra[indices], axis=1) # mediana de cada muestra bootstrap
    percentil_bajo = 100 * alpha # el percentil inferior para un IC de (1 - 2*alpha)*100% es alpha*100
    percentil_alto = 100 * (1 - alpha) # el percentil superior para un IC de (1 - 2*alpha)*100% es (1 - alpha)*100
    ic_bajo, ic_alto = np.percentile(medianas_bootstrap, [percentil_bajo, percentil_alto])
    return float(ic_bajo), float(ic_alto)


def nombre_dataset(df: pd.DataFrame, ruta_csv: Path) -> str:
    if "dataset" in df.columns and not df["dataset"].dropna().empty:
        return str(df["dataset"].dropna().iloc[0])
    return ruta_csv.stem


def ejecutar_equivalencia(
    ruta_csv: Path,
    metrica: str,
    margen: float,
    alpha: float,
    bootstrap_replicas: int,
    rng: np.random.Generator,
) -> None:
    modelo_a = "OVA"
    modelo_b = "multi-output"
    df = pd.read_csv(ruta_csv)
    dataset = nombre_dataset(df, ruta_csv)
    diferencias = diferencias_pareadas(df, metrica, modelo_a, modelo_b)
    ic_bajo, ic_alto = intervalo_bootstrap_mediana(
        muestra=diferencias.to_numpy(dtype=float),
        alpha=alpha,
        bootstrap_replicas=bootstrap_replicas,
        rng=rng,
    )
    equivalente = ic_bajo > -margen and ic_alto < margen

    print(f"CSV: {ruta_csv}")
    print(f"Dataset: {dataset}")
    print(f"Metrica: {metrica}")
    print(f"Diferencia: {modelo_a} - {modelo_b}")
    print(f"Margen de equivalencia: [-{margen:.6f}, {margen:.6f}]")
    print(f"Semillas usadas: {len(diferencias)}")
    print(f"Diferencia mediana: {diferencias.median():.6f}")
    print(f"Diferencia media: {diferencias.mean():.6f}")
    print(f"IC bootstrap del {100 * (1 - 2 * alpha):.0f}% para la mediana: [{ic_bajo:.6f}, {ic_alto:.6f}]")
    if equivalente:
        print(f"Decisión TOST/bootstrap: equivalencia dentro de +/-{margen:g}")
    else:
        print("Decisión TOST/bootstrap: no se establece equivalencia")


def main() -> None:
    args = crear_parser().parse_args()
    rutas_csv = [Path(csv_file) for csv_file in args.csv] if args.csv else [Path(csv_file) for csv_file in CSV_POR_DEFECTO]

    if args.margen <= 0:
        raise ValueError("--margen debe ser positivo.")
    if args.bootstrap_replicas <= 0:
        raise ValueError("--bootstrap-replicas debe ser positivo.")

    rng = np.random.default_rng(args.semilla_rng)
    for ruta_csv in rutas_csv:
        print("=" * 80, flush=True)
        print(ruta_csv, flush=True)
        print("=" * 80, flush=True)
        ejecutar_equivalencia(
            ruta_csv=ruta_csv,
            metrica=args.metrica,
            margen=args.margen,
            alpha=args.alpha,
            bootstrap_replicas=args.bootstrap_replicas,
            rng=rng,
        )
        print(flush=True)


if __name__ == "__main__":
    main()
