from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PLANE_LABELS = {
    "ax": "axial",
    "co": "coronal",
    "sa": "sagital",
}
DEFAULT_COMPARE_CLASSES = ("glioma", "pituitary")

DEFAULT_METRICS = (
    "cam_pointing_game_hit",
    "cam_top5_dice",
    "cam_top5_iou",
    "cam_top5_outside_precision",
    "cam_top10_dice",
    "cam_top10_iou",
    "cam_top10_outside_precision",
    "cam_top15_dice",
    "cam_top15_iou",
    "cam_top15_outside_precision",
    "cam_top20_dice",
    "cam_top20_iou",
    "cam_top20_outside_precision",
    "cam_top_mask_area_dice",
    "cam_top_mask_area_iou",
    "cam_top_mask_area_outside_precision",
    "cam_top_2x_mask_area_dice",
    "cam_top_2x_mask_area_iou",
    "cam_top_2x_mask_area_outside_precision",
    "cam_tumor_activation_frac",
    "cam_peritumor_r5_activation_frac",
    "cam_outside_peritumor_r5_activation_frac",
    "cam_gini",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume metricas de explicabilidad BRISC estratificando por plano "
            "de imagen (ax/co/sa) y clase verdadera."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        required=True,
        help="Uno o varios ficheros gradcam_index.csv generados por explicabilidad_gradcam_vgg.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados_actualizados/analisis_explicabilidad/brisc/plano_clase"),
        help="Carpeta donde guardar las tablas resumen.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_METRICS),
        help="Metricas base sin prefijo multi_/ova_.",
    )
    parser.add_argument(
        "--include-no-tumor",
        action="store_true",
        help="Incluye la clase no_tumor si aparece en el CSV.",
    )
    parser.add_argument(
        "--compare-classes",
        nargs=2,
        default=list(DEFAULT_COMPARE_CLASSES),
        metavar=("CLASS_A", "CLASS_B"),
        help="Dos clases para generar una comparacion especifica por plano.",
    )
    return parser


def extract_plane(path_value: object) -> str:
    filename = Path(str(path_value)).name
    match = re.search(r"_(ax|co|sa)_", filename)
    if not match:
        return "unknown"
    return match.group(1)


def read_inputs(paths: list[Path], include_no_tumor: bool) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        if "cam_method" not in df.columns:
            df["cam_method"] = path.parent.name
        if "target_layer" not in df.columns:
            df["target_layer"] = "unknown"
        if "cam_target_mode" not in df.columns:
            df["cam_target_mode"] = "unknown"
        frames.append(df)

    if not frames:
        raise ValueError("No se ha podido cargar ningun CSV.")

    df = pd.concat(frames, ignore_index=True)
    if "image_path" not in df.columns:
        raise ValueError("El CSV debe contener la columna image_path.")
    if "true_label" not in df.columns:
        raise ValueError("El CSV debe contener la columna true_label.")

    df["plane"] = df["image_path"].map(extract_plane)
    df["plane_name"] = df["plane"].map(PLANE_LABELS).fillna(df["plane"])

    if not include_no_tumor:
        df = df[df["true_label"] != "no_tumor"].copy()

    return df


def available_metrics(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    present = []
    missing = []
    for metric in metrics:
        multi_col = f"multi_{metric}"
        ova_col = f"ova_{metric}"
        if multi_col in df.columns and ova_col in df.columns:
            present.append(metric)
        else:
            missing.append(metric)

    if missing:
        print("Metricas omitidas porque no estan en ambos modelos:")
        for metric in missing:
            print(f"  - {metric}")

    if not present:
        raise ValueError("Ninguna metrica solicitada existe con prefijos multi_/ova_.")

    return present


def summarize_long(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    run_cols = ["source_csv", "cam_method", "target_layer", "cam_target_mode"]
    full_group_cols = run_cols + group_cols

    for group_values, group in df.groupby(full_group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base = dict(zip(full_group_cols, group_values))
        base["n"] = int(len(group))

        if "multi_mask_area_frac" in group.columns:
            base["mask_area_frac_mean"] = float(group["multi_mask_area_frac"].mean())
            base["mask_area_frac_median"] = float(group["multi_mask_area_frac"].median())
            base["mask_area_frac_min"] = float(group["multi_mask_area_frac"].min())
            base["mask_area_frac_max"] = float(group["multi_mask_area_frac"].max())

        for metric in metrics:
            multi = pd.to_numeric(group[f"multi_{metric}"], errors="coerce")
            ova = pd.to_numeric(group[f"ova_{metric}"], errors="coerce")
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "multi_mean": float(multi.mean()),
                    "ova_mean": float(ova.mean()),
                    "diff_ova_minus_multi": float(ova.mean() - multi.mean()),
                    "multi_median": float(multi.median()),
                    "ova_median": float(ova.median()),
                }
            )

    return pd.DataFrame(rows)


def summarize_compact(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    run_cols = ["source_csv", "cam_method", "target_layer", "cam_target_mode"]
    full_group_cols = run_cols + group_cols
    rows = []

    for group_values, group in df.groupby(full_group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(full_group_cols, group_values))
        row["n"] = int(len(group))

        if "multi_mask_area_frac" in group.columns:
            row["mask_area_frac_mean"] = float(group["multi_mask_area_frac"].mean())
            row["mask_area_frac_median"] = float(group["multi_mask_area_frac"].median())
            row["mask_area_frac_min"] = float(group["multi_mask_area_frac"].min())
            row["mask_area_frac_max"] = float(group["multi_mask_area_frac"].max())

        for metric in metrics:
            multi = pd.to_numeric(group[f"multi_{metric}"], errors="coerce")
            ova = pd.to_numeric(group[f"ova_{metric}"], errors="coerce")
            row[f"multi_{metric}_mean"] = float(multi.mean())
            row[f"ova_{metric}_mean"] = float(ova.mean())
            row[f"diff_{metric}_ova_minus_multi"] = float(ova.mean() - multi.mean())

        rows.append(row)

    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, output_dir: Path, name: str, group_cols: list[str], metrics: list[str]) -> None:
    long_df = summarize_long(df, group_cols, metrics)
    compact_df = summarize_compact(df, group_cols, metrics)
    long_df.to_csv(output_dir / f"{name}_long.csv", index=False)
    compact_df.to_csv(output_dir / f"{name}.csv", index=False)


def write_class_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    class_a: str,
    class_b: str,
    metrics: list[str],
) -> None:
    compare_df = df[df["true_label"].isin([class_a, class_b])].copy()
    if compare_df.empty:
        print(f"No hay filas para comparar {class_a} vs {class_b}.")
        return

    prefix = f"comparacion_{class_a}_vs_{class_b}"
    write_summary(compare_df, output_dir, prefix, ["true_label", "plane", "plane_name"], metrics)

    long_df = summarize_long(compare_df, ["true_label", "plane", "plane_name"], metrics)
    run_cols = ["source_csv", "cam_method", "target_layer", "cam_target_mode", "plane", "plane_name", "metric"]
    rows = []
    for group_values, group in long_df.groupby(run_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        by_class = group.set_index("true_label")
        if class_a not in by_class.index or class_b not in by_class.index:
            continue
        row = dict(zip(run_cols, group_values))
        for field in (
            "n",
            "mask_area_frac_mean",
            "mask_area_frac_median",
            "mask_area_frac_min",
            "mask_area_frac_max",
            "multi_mean",
            "ova_mean",
            "diff_ova_minus_multi",
        ):
            row[f"{class_a}_{field}"] = by_class.loc[class_a, field]
            row[f"{class_b}_{field}"] = by_class.loc[class_b, field]
            if field not in {"n"}:
                row[f"{class_b}_minus_{class_a}_{field}"] = (
                    by_class.loc[class_b, field] - by_class.loc[class_a, field]
                )
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_diferencias_por_plano.csv", index=False)


def write_prediction_summary(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    run_cols = ["source_csv", "cam_method", "target_layer", "cam_target_mode"]
    group_cols = run_cols + ["true_label", "plane", "plane_name"]
    for group_values, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        row["n"] = int(len(group))
        row["multi_accuracy"] = float((group["multi_pred"] == group["true_label"]).mean())
        row["ova_accuracy"] = float((group["ova_pred"] == group["true_label"]).mean())
        row["diff_ova_minus_multi_accuracy"] = row["ova_accuracy"] - row["multi_accuracy"]
        row["both_correct_frac"] = float((group["outcome"] == "both_correct").mean())
        row["multi_correct_ova_wrong_frac"] = float((group["outcome"] == "multi_correct_ova_wrong").mean())
        row["multi_wrong_ova_correct_frac"] = float((group["outcome"] == "multi_wrong_ova_correct").mean())
        row["both_wrong_frac"] = float((group["outcome"] == "both_wrong").mean())
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "rendimiento_por_clase_plano.csv", index=False)


def write_outcome_summary(df: pd.DataFrame, output_dir: Path, metrics: list[str]) -> None:
    if "outcome" not in df.columns:
        return
    write_summary(df, output_dir, "resumen_por_outcome", ["outcome"], metrics)
    write_summary(df, output_dir, "resumen_por_clase_outcome", ["true_label", "outcome"], metrics)


def main() -> None:
    args = build_parser().parse_args()
    df = read_inputs(args.csv, include_no_tumor=args.include_no_tumor)
    metrics = available_metrics(df, args.metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = (
        df.groupby(["true_label", "plane", "plane_name"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["true_label", "plane"])
    )
    counts.to_csv(args.output_dir / "conteo_por_clase_plano.csv", index=False)

    write_summary(df, args.output_dir, "resumen_global", [], metrics)
    write_summary(df, args.output_dir, "resumen_por_clase", ["true_label"], metrics)
    write_summary(df, args.output_dir, "resumen_por_plano", ["plane", "plane_name"], metrics)
    write_summary(df, args.output_dir, "resumen_por_clase_plano", ["true_label", "plane", "plane_name"], metrics)
    write_class_comparison(df, args.output_dir, args.compare_classes[0], args.compare_classes[1], metrics)
    write_prediction_summary(df, args.output_dir)
    write_outcome_summary(df, args.output_dir, metrics)

    print(f"Filas analizadas: {len(df)}")
    print(f"Metricas usadas: {len(metrics)}")
    print(f"Resultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()
