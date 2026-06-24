from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_DETAIL_CSV = Path(
    "resultados_actualizados/analisis_arquitecturas_ova/arquitecturas_ova_10semillas_detalle.csv"
)
DEFAULT_OUTPUT_DIR = Path("resultados_estadisticos")
MEDICAL_DATASETS = {"brisc", "tb_chest_xray"}

SELECTED_CONFIGS = {
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analisis estadistico complementario para las configuraciones "
            "One-vs-All seleccionadas frente al multi-output de referencia."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrica", default="f1_macro")
    parser.add_argument("--margen", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dif-relevante", type=float, default=0.02)
    parser.add_argument("--dif-verdadera", type=float, default=0.005)
    parser.add_argument("--min-semillas", type=int, default=3)
    parser.add_argument("--max-semillas", type=int, default=100)
    parser.add_argument("--simulaciones", type=int, default=2000)
    parser.add_argument("--bootstrap-replicas-potencia", type=int, default=1000)
    parser.add_argument("--bootstrap-replicas-test", type=int, default=10000)
    parser.add_argument("--semilla-rng", type=int, default=12345)
    return parser


def potencia_objetivo(dataset: str) -> float:
    return 0.90 if dataset in MEDICAL_DATASETS else 0.80


def multi_reference(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    required = {"dataset", "seed", "model_type", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias: {sorted(missing)}")

    subset = df[(df["dataset"] == dataset) & (df["model_type"] == "multi-output")]
    if subset.empty:
        raise ValueError(f"No hay multi-output de referencia para {dataset}")

    return subset[["seed", metric]].rename(columns={metric: "multi-output referencia"})


def selected_ova_from_detail(
    df: pd.DataFrame,
    dataset: str,
    approach_architecture: str,
    metric: str,
) -> pd.DataFrame:
    required = {"dataset", "seed", "approach_architecture", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias: {sorted(missing)}")

    subset = df[
        (df["dataset"] == dataset)
        & (df["approach_architecture"] == approach_architecture)
    ]
    if subset.empty:
        raise ValueError(
            f"No hay filas para dataset={dataset} y arquitectura={approach_architecture}"
        )

    return subset[["seed", metric]].rename(columns={metric: "OVA seleccionada"})


def selected_ova_from_file(path: Path, metric: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {"seed", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias en {path}: {sorted(missing)}")

    selected = df[pd.to_numeric(df["seed"], errors="coerce").notna()].copy()
    selected["seed"] = selected["seed"].astype(int)
    return selected[["seed", metric]].rename(columns={metric: "OVA seleccionada"})


def paired_values(
    detail: pd.DataFrame,
    dataset: str,
    spec: dict[str, object],
    metric: str,
) -> pd.DataFrame:
    multi = multi_reference(detail, dataset, metric)
    source = spec["source"]

    if source == "detail":
        selected = selected_ova_from_detail(
            detail,
            dataset,
            str(spec["approach_architecture"]),
            metric,
        )
    else:
        selected = selected_ova_from_file(Path(source), metric)

    paired = multi.merge(selected, on="seed", how="inner").sort_values("seed")
    if paired.empty:
        raise ValueError(f"No hay semillas pareadas para {dataset}")
    return paired


def bootstrap_median_ci(
    sample: np.ndarray,
    alpha: float,
    bootstrap_replicas: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, len(sample), size=(bootstrap_replicas, len(sample)))
    medians = np.median(sample[indices], axis=1)
    low, high = np.percentile(medians, [100 * alpha, 100 * (1 - alpha)])
    return float(low), float(high)


def wilcoxon_power(
    diffs: np.ndarray,
    n_seeds: int,
    relevant_diff: float,
    alpha: float,
    simulations: int,
    rng: np.random.Generator,
) -> float:
    centered = diffs - np.median(diffs)
    rejections = 0
    for _ in range(simulations):
        sample = rng.choice(centered, size=n_seeds, replace=True) + relevant_diff
        try:
            result = wilcoxon(sample, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            continue
        if result.pvalue < alpha:
            rejections += 1
    return rejections / simulations


def tost_power(
    diffs: np.ndarray,
    n_seeds: int,
    margin: float,
    true_diff: float,
    alpha: float,
    simulations: int,
    bootstrap_replicas: int,
    rng: np.random.Generator,
) -> float:
    centered = diffs - np.median(diffs)
    equivalents = 0
    for _ in range(simulations):
        sample = rng.choice(centered, size=n_seeds, replace=True) + true_diff
        low, high = bootstrap_median_ci(sample, alpha, bootstrap_replicas, rng)
        if low > -margin and high < margin:
            equivalents += 1
    return equivalents / simulations


def required_seeds_wilcoxon(
    diffs: np.ndarray,
    relevant_diff: float,
    alpha: float,
    target_power: float,
    min_seeds: int,
    max_seeds: int,
    simulations: int,
    rng: np.random.Generator,
) -> tuple[int | None, float]:
    last_power = 0.0
    for n_seeds in range(min_seeds, max_seeds + 1):
        last_power = wilcoxon_power(
            diffs,
            n_seeds,
            relevant_diff,
            alpha,
            simulations,
            rng,
        )
        if last_power >= target_power:
            return n_seeds, last_power
    return None, last_power


def required_seeds_tost(
    diffs: np.ndarray,
    margin: float,
    true_diff: float,
    alpha: float,
    target_power: float,
    min_seeds: int,
    max_seeds: int,
    simulations: int,
    bootstrap_replicas: int,
    rng: np.random.Generator,
) -> tuple[int | None, float]:
    last_power = 0.0
    for n_seeds in range(min_seeds, max_seeds + 1):
        last_power = tost_power(
            diffs,
            n_seeds,
            margin,
            true_diff,
            alpha,
            simulations,
            bootstrap_replicas,
            rng,
        )
        if last_power >= target_power:
            return n_seeds, last_power
    return None, last_power


def run_observed_tests(
    dataset: str,
    label: str,
    paired: pd.DataFrame,
    metric: str,
    margin: float,
    alpha: float,
    bootstrap_replicas: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    diffs = paired["OVA seleccionada"] - paired["multi-output referencia"]
    try:
        wilcoxon_result = wilcoxon(
            paired["OVA seleccionada"],
            paired["multi-output referencia"],
            alternative="two-sided",
            zero_method="wilcox",
        )
        wilcoxon_p = float(wilcoxon_result.pvalue)
        wilcoxon_stat = float(wilcoxon_result.statistic)
    except ValueError:
        wilcoxon_p = np.nan
        wilcoxon_stat = np.nan

    ci_low, ci_high = bootstrap_median_ci(
        diffs.to_numpy(dtype=float),
        alpha,
        bootstrap_replicas,
        rng,
    )

    return {
        "dataset": dataset,
        "comparacion": f"{label} - multi-output referencia",
        "metrica": metric,
        "semillas_usadas": len(paired),
        "multi_output_media": paired["multi-output referencia"].mean(),
        "ova_seleccionada_media": paired["OVA seleccionada"].mean(),
        "diferencia_media": diffs.mean(),
        "diferencia_mediana": diffs.median(),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "wilcoxon_rechaza_h0_005": bool(wilcoxon_p < 0.05)
        if not np.isnan(wilcoxon_p)
        else False,
        "margen_equivalencia": margin,
        "ic_bootstrap_90_mediana_bajo": ci_low,
        "ic_bootstrap_90_mediana_alto": ci_high,
        "tost_equivalente": bool(ci_low > -margin and ci_high < margin),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.margen <= 0:
        raise ValueError("--margen debe ser positivo")
    if abs(args.dif_verdadera) >= args.margen:
        raise ValueError("--dif-verdadera debe estar dentro del margen")

    detail = pd.read_csv(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.semilla_rng)

    observed_rows = []
    wilcoxon_power_rows = []
    tost_power_rows = []

    for dataset, spec in SELECTED_CONFIGS.items():
        paired = paired_values(detail, dataset, spec, args.metrica)
        diffs = (
            paired["OVA seleccionada"] - paired["multi-output referencia"]
        ).to_numpy(dtype=float)
        target_power = potencia_objetivo(dataset)
        label = str(spec["label"])
        comparison = f"{label} - multi-output referencia"

        observed_rows.append(
            run_observed_tests(
                dataset,
                label,
                paired,
                args.metrica,
                args.margen,
                args.alpha,
                args.bootstrap_replicas_test,
                rng,
            )
        )

        wilcoxon_seeds, wilcoxon_power_value = required_seeds_wilcoxon(
            diffs,
            args.dif_relevante,
            args.alpha,
            target_power,
            args.min_semillas,
            args.max_semillas,
            args.simulaciones,
            rng,
        )
        wilcoxon_power_rows.append(
            {
                "dataset": dataset,
                "comparacion": comparison,
                "metrica": args.metrica,
                "dif_relevante_simulada": args.dif_relevante,
                "alpha": args.alpha,
                "potencia_objetivo": target_power,
                "semillas_necesarias": wilcoxon_seeds
                if wilcoxon_seeds is not None
                else f">{args.max_semillas}",
                "potencia_estimada": wilcoxon_power_value,
            }
        )

        tost_seeds, tost_power_value = required_seeds_tost(
            diffs,
            args.margen,
            args.dif_verdadera,
            args.alpha,
            target_power,
            args.min_semillas,
            args.max_semillas,
            args.simulaciones,
            args.bootstrap_replicas_potencia,
            rng,
        )
        tost_power_rows.append(
            {
                "dataset": dataset,
                "comparacion": comparison,
                "metrica": args.metrica,
                "margen_equivalencia": args.margen,
                "dif_verdadera_simulada": args.dif_verdadera,
                "alpha": args.alpha,
                "potencia_objetivo": target_power,
                "semillas_necesarias": tost_seeds
                if tost_seeds is not None
                else f">{args.max_semillas}",
                "potencia_estimada": tost_power_value,
            }
        )

    observed = pd.DataFrame(observed_rows)
    wilcoxon_power_df = pd.DataFrame(wilcoxon_power_rows)
    tost_power_df = pd.DataFrame(tost_power_rows)

    observed_path = args.output_dir / "test_configuraciones_seleccionadas.csv"
    wilcoxon_power_path = (
        args.output_dir / "potencia_wilcoxon_configuraciones_seleccionadas.csv"
    )
    tost_power_path = args.output_dir / "potencia_tost_configuraciones_seleccionadas.csv"

    observed.to_csv(observed_path, index=False)
    wilcoxon_power_df.to_csv(wilcoxon_power_path, index=False)
    tost_power_df.to_csv(tost_power_path, index=False)

    print(f"Saved observed tests to {observed_path}")
    print(f"Saved Wilcoxon power to {wilcoxon_power_path}")
    print(f"Saved TOST power to {tost_power_path}\n")
    print("Tests observados:")
    print(observed.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPotencia Wilcoxon:")
    print(
        wilcoxon_power_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("\nPotencia TOST/bootstrap:")
    print(
        tost_power_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()
