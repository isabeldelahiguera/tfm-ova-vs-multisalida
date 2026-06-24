from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "tumor_area_frac",
    "mask_centroid_x_norm",
    "mask_centroid_y_norm",
    "mask_bbox_width_frac",
    "mask_bbox_height_frac",
    "mask_bbox_area_frac",
    "tumor_intensity_mean",
    "tumor_vs_outside_mean_diff",
    "tumor_vs_peritumor_mean_diff",
    "image_nonzero_area_frac",
    "image_nonzero_centroid_x_norm",
    "image_nonzero_centroid_y_norm",
]

PLANE_NAMES = {
    "ax": "axial",
    "co": "coronal",
    "sa": "sagittal",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cruza descriptores del dataset BRISC con predicciones Multi/OVA para "
            "comparar aciertos y errores por clase/plano."
        )
    )
    parser.add_argument(
        "--dataset-csv",
        default="resultados_actualizados/analisis_dataset/brisc/brisc_dataset_por_imagen.csv",
    )
    parser.add_argument(
        "--prediction-csv",
        nargs="+",
        default=[
            "resultados_actualizados/explicabilidad/brisc/seed_1_lrp/gradcam_index.csv",
            "resultados_actualizados/explicabilidad/brisc/seed_2_lrp/gradcam_index.csv",
            "resultados_actualizados/explicabilidad/brisc/seed_3_lrp/gradcam_index.csv",
        ],
        help="Uno o varios gradcam_index.csv con true_label, multi_pred y ova_pred.",
    )
    parser.add_argument(
        "--output-dir",
        default="resultados_actualizados/analisis_explicabilidad/brisc/errores_dataset",
    )
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    return parser


def seed_from_path(path: Path) -> str:
    match = re.search(r"seed_(\d+)", str(path))
    return match.group(1) if match else "unknown"


def plane_from_name(path: str) -> str:
    match = re.search(r"_(ax|co|sa)_", Path(path).stem.lower())
    return match.group(1) if match else "unknown"


def load_predictions(paths: list[str]) -> pd.DataFrame:
    frames = []
    for csv_path in paths:
        path = Path(csv_path)
        df = pd.read_csv(path)
        needed = ["image_path", "true_label", "multi_pred", "ova_pred"]
        missing = [col for col in needed if col not in df.columns]
        if missing:
            raise ValueError(f"{path} no contiene columnas requeridas: {missing}")
        df = df.copy()
        df["source_csv"] = str(path)
        if "seed" not in df.columns:
            df["seed"] = seed_from_path(path)
        if "plane" not in df.columns:
            df["plane"] = df["image_path"].map(plane_from_name)
        df["plane_name"] = df["plane"].map(PLANE_NAMES).fillna("unknown")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_train_reference_zscores(
    merged: pd.DataFrame,
    dataset_df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    train = dataset_df[dataset_df["split"] == "train"].copy()
    ref = train.groupby(["true_label", "plane"], dropna=False)[features].agg(["mean", "std"])

    rows = []
    for _, row in merged.iterrows():
        label = row["true_label"]
        plane = row["plane"]
        out = row.to_dict()
        z_values = []
        for feature in features:
            if feature not in merged.columns:
                continue
            try:
                mean = float(ref.loc[(label, plane), (feature, "mean")])
                std = float(ref.loc[(label, plane), (feature, "std")])
            except KeyError:
                mean = np.nan
                std = np.nan
            value = pd.to_numeric(row.get(feature), errors="coerce")
            if pd.isna(value) or pd.isna(mean) or pd.isna(std) or std <= 1e-12:
                z = np.nan
            else:
                z = float((value - mean) / std)
                z_values.append(abs(z))
            out[f"{feature}_train_z"] = z
            out[f"{feature}_abs_train_z"] = abs(z) if not pd.isna(z) else np.nan
        out["train_typicality_abs_z_mean"] = float(np.mean(z_values)) if z_values else np.nan
        out["train_typicality_abs_z_max"] = float(np.max(z_values)) if z_values else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def long_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, pred_col in [("multi", "multi_pred"), ("ova", "ova_pred")]:
        tmp = df.copy()
        tmp["model"] = model_name
        tmp["pred_label"] = tmp[pred_col]
        tmp["is_correct"] = tmp["pred_label"] == tmp["true_label"]
        tmp["result"] = np.where(tmp["is_correct"], "correct", "wrong")
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def summarize_correct_vs_wrong(df_long: pd.DataFrame, group_cols: list[str], features: list[str]) -> pd.DataFrame:
    rows = []
    numeric = [
        feature
        for feature in features
        + ["train_typicality_abs_z_mean", "train_typicality_abs_z_max"]
        if feature in df_long.columns
    ]
    for keys, group in df_long.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n"] = int(len(group))
        row["accuracy"] = float(group["is_correct"].mean())
        row["wrong_n"] = int((~group["is_correct"]).sum())
        row["correct_n"] = int(group["is_correct"].sum())
        for feature in numeric:
            correct_values = pd.to_numeric(group.loc[group["is_correct"], feature], errors="coerce")
            wrong_values = pd.to_numeric(group.loc[~group["is_correct"], feature], errors="coerce")
            row[f"correct_{feature}_mean"] = (
                float(correct_values.mean()) if correct_values.notna().any() else np.nan
            )
            row[f"wrong_{feature}_mean"] = (
                float(wrong_values.mean()) if wrong_values.notna().any() else np.nan
            )
            row[f"diff_wrong_minus_correct_{feature}"] = (
                row[f"wrong_{feature}_mean"] - row[f"correct_{feature}_mean"]
                if not pd.isna(row[f"wrong_{feature}_mean"])
                and not pd.isna(row[f"correct_{feature}_mean"])
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wrong = df_long[~df_long["is_correct"]].copy()
    for keys, group in wrong.groupby(["model", "true_label", "pred_label"], dropna=False):
        model, true_label, pred_label = keys
        rows.append(
            {
                "model": model,
                "true_label": true_label,
                "pred_label": pred_label,
                "n": int(len(group)),
                "frac_of_model_errors": float(len(group) / max(1, len(wrong[wrong["model"] == model]))),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "n"], ascending=[True, False])


def write_pituitary_extremes(df_long: pd.DataFrame, output_dir: Path) -> None:
    pit = df_long[(df_long["true_label"] == "pituitary") & (~df_long["is_correct"])].copy()
    if pit.empty:
        pd.DataFrame().to_csv(output_dir / "pituitary_errores_extremos.csv", index=False)
        return
    sort_cols = [
        "train_typicality_abs_z_max",
        "tumor_area_frac_abs_train_z",
        "mask_centroid_x_norm_abs_train_z",
        "mask_centroid_y_norm_abs_train_z",
    ]
    available = [col for col in sort_cols if col in pit.columns]
    cols = [
        "seed",
        "model",
        "true_label",
        "pred_label",
        "plane",
        "image_path",
        "tumor_area_frac",
        "mask_centroid_x_norm",
        "mask_centroid_y_norm",
        "tumor_intensity_mean",
        "train_typicality_abs_z_mean",
        "train_typicality_abs_z_max",
    ]
    cols = [col for col in cols if col in pit.columns]
    pit.sort_values(available, ascending=False).head(50)[cols].to_csv(
        output_dir / "pituitary_errores_extremos.csv",
        index=False,
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = pd.read_csv(args.dataset_csv)
    predictions = load_predictions(args.prediction_csv)
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()

    features = [feature for feature in args.features if feature in dataset_df.columns]
    merged = predictions.merge(
        test_dataset,
        on=["image_path", "true_label", "plane", "plane_name"],
        how="left",
        suffixes=("", "_dataset"),
    )
    merged = add_train_reference_zscores(merged, dataset_df, features)
    df_long = long_by_model(merged)

    merged.to_csv(output_dir / "predicciones_con_descriptores.csv", index=False)
    df_long.to_csv(output_dir / "predicciones_con_descriptores_long_modelo.csv", index=False)

    summary_specs = {
        "correctos_vs_errores_por_modelo_clase.csv": ["model", "true_label"],
        "correctos_vs_errores_por_modelo_clase_plano.csv": [
            "model",
            "true_label",
            "plane",
            "plane_name",
        ],
        "correctos_vs_errores_por_modelo.csv": ["model"],
    }
    for filename, group_cols in summary_specs.items():
        summarize_correct_vs_wrong(df_long, group_cols, features).to_csv(
            output_dir / filename,
            index=False,
        )

    confusion_summary(df_long).to_csv(output_dir / "confusiones_errores.csv", index=False)
    write_pituitary_extremes(df_long, output_dir)

    print(f"Predicciones cruzadas: {len(merged)}", flush=True)
    print(f"Filas long modelo: {len(df_long)}", flush=True)
    print(f"Resultados guardados en: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
