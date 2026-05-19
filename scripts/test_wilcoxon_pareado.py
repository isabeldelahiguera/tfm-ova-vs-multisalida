from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon



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
    parser = argparse.ArgumentParser(description="Test de Wilcoxon pareado entre OVA y multi-output.")
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
    return parser


def ejecutar_wilcoxon(ruta_csv: Path, metrica: str) -> None:
    df = pd.read_csv(ruta_csv)

    columnas_necesarias = {"seed", "model_type", metrica}
    columnas_ausentes = columnas_necesarias - set(df.columns) # devuelve set de columnas necesarias que no están en el DataFrame
    if columnas_ausentes:
        # si hay columnas necesarias que faltan, se lanza un error indicando cuáles son
        raise ValueError(f"Faltan columnas necesarias: {sorted(columnas_ausentes)}")

    modelo_a = "OVA"
    modelo_b = "multi-output"

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

    diferencias = pareado[modelo_a] - pareado[modelo_b]
    
    resultado = wilcoxon(
        pareado[modelo_a],
        pareado[modelo_b],
        alternative="two-sided",
        zero_method="wilcox",
    )

    print(f"CSV: {ruta_csv}")
    print(f"Metrica: {metrica}")
    print(f"Semillas usadas: {len(pareado)}")
    print("H0: la mediana de las diferencias pareadas (OVA - multi-output) es 0.")
    print("H1: la mediana de las diferencias pareadas (OVA - multi-output) no es 0.")
    print()
    print("Valores pareados por semilla:")
    print(pareado.to_string())
    print()
    print(f"Diferencia mediana: {diferencias.median():.6f}")
    print(f"Diferencia media: {diferencias.mean():.6f}")
    print(f"Estadistico de Wilcoxon: {resultado.statistic}")
    print(f"p-valor: {resultado.pvalue:.10f}")
    print(f"Conclusión al nivel de significancia 0.05: {'Rechazamos H0' if resultado.pvalue < 0.05 else 'No hay evidencia estadística suficiente para rechazar H0'}.")


def main() -> None:
    args = crear_parser().parse_args()
    rutas_csv = [Path(csv_file) for csv_file in args.csv] if args.csv else [Path(csv_file) for csv_file in CSV_POR_DEFECTO]

    for ruta_csv in rutas_csv:
        print("=" * 80, flush=True)
        print(ruta_csv, flush=True)
        print("=" * 80, flush=True)
        ejecutar_wilcoxon(ruta_csv, args.metrica)
        print(flush=True)


if __name__ == "__main__":
    main()
