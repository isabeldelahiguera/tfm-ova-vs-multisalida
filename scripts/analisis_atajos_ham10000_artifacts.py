from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, feature, filters


DEFAULT_INPUTS = {
    "gradcam": "resultados_actualizados/explicabilidad/ham10000/seed_1_vgg16_block5_sampler_balanced_gradcam_alltest/gradcam_index.csv",
    "gradcampp": "resultados_actualizados/explicabilidad/ham10000/seed_1_vgg16_block5_sampler_balanced_gradcampp_alltest_gradcampp/gradcam_index.csv",
    "lrp": "resultados_actualizados/explicabilidad/ham10000/seed_1_vgg16_block5_sampler_balanced_lrp_alltest/gradcam_index.csv",
}


SPATIAL_METRICS = [
    "inside_frac",
    "outside_frac",
    "outside_peritumor_r5_activation_frac",
    "pointing_game_hit",
    "top_mask_area_dice",
    "top_mask_area_iou",
    "mask_centroid_distance_norm",
]


ARTIFACT_METRICS = [
    "artifact_dark_border_frac",
    "artifact_dark_corner_frac",
    "artifact_hair_line_score",
    "artifact_specular_frac",
    "artifact_high_saturation_frac",
    "artifact_edge_density_outside_lesion",
]


def read_index(path: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = method
    return df


def image_id_from_path(path: str) -> str:
    return Path(path).stem


def load_rgb(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: str | float | None, image_size: int) -> np.ndarray:
    if not isinstance(path, str) or not path:
        return np.zeros((image_size, image_size), dtype=bool)
    mask_path = Path(path)
    if not mask_path.exists():
        return np.zeros((image_size, image_size), dtype=bool)
    with Image.open(mask_path) as image:
        mask = image.convert("L").resize((image_size, image_size), Image.NEAREST)
    return np.asarray(mask) > 0


def border_mask(shape: tuple[int, int], width_frac: float = 0.06) -> np.ndarray:
    h, w = shape
    border = max(1, int(round(min(h, w) * width_frac)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return mask


def corner_mask(shape: tuple[int, int], size_frac: float = 0.16) -> np.ndarray:
    h, w = shape
    size = max(1, int(round(min(h, w) * size_frac)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:size, :size] = True
    mask[:size, -size:] = True
    mask[-size:, :size] = True
    mask[-size:, -size:] = True
    return mask


def safe_frac(mask: np.ndarray, region: np.ndarray) -> float:
    denom = int(region.sum())
    if denom == 0:
        return 0.0
    return float(np.logical_and(mask, region).sum() / denom)


def artifact_features(image_path: str, mask_path: str | float | None, image_size: int) -> dict[str, float | str]:
    rgb = load_rgb(image_path, image_size)
    lesion = load_mask(mask_path, image_size)
    outside_lesion = ~lesion
    gray = color.rgb2gray(rgb)
    hsv = color.rgb2hsv(rgb)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    border = border_mask(gray.shape)
    corners = corner_mask(gray.shape)

    dark_pixels = gray < 0.12
    specular_pixels = (value > 0.92) & (saturation < 0.25)
    high_saturation_pixels = (saturation > 0.75) & (value > 0.30)

    # Frangi is a standard vessel/ridge filter; on inverted dermoscopy images
    # it gives a reproducible proxy for dark thin line artifacts such as hairs.
    inverted = 1.0 - gray
    hair_response = filters.frangi(inverted, sigmas=(1, 2, 3), black_ridges=False)
    hair_response = np.nan_to_num(hair_response, nan=0.0, posinf=0.0, neginf=0.0)
    dark_line_region = outside_lesion & (gray < 0.45)
    hair_line_score = float(hair_response[dark_line_region].mean()) if dark_line_region.any() else 0.0

    edges = feature.canny(gray, sigma=1.2)

    return {
        "image_id": image_id_from_path(image_path),
        "artifact_dark_border_frac": safe_frac(dark_pixels, border),
        "artifact_dark_corner_frac": safe_frac(dark_pixels, corners),
        "artifact_hair_line_score": hair_line_score,
        "artifact_specular_frac": safe_frac(specular_pixels, outside_lesion),
        "artifact_high_saturation_frac": safe_frac(high_saturation_pixels, outside_lesion),
        "artifact_edge_density_outside_lesion": safe_frac(edges, outside_lesion),
    }


def compute_artifacts(reference: pd.DataFrame, image_size: int) -> pd.DataFrame:
    rows = []
    seen = set()
    for row in reference.itertuples(index=False):
        image_id = image_id_from_path(row.image_path)
        if image_id in seen:
            continue
        seen.add(image_id)
        rows.append(artifact_features(row.image_path, row.mask_path, image_size))
    return pd.DataFrame(rows)


def add_metadata(df: pd.DataFrame, metadata_path: str | None) -> pd.DataFrame:
    if metadata_path is None:
        return df
    path = Path(metadata_path)
    if not path.exists():
        return df
    meta = pd.read_csv(path)
    if "image_id" not in meta.columns:
        return df
    keep = [column for column in ["image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization", "dataset"] if column in meta.columns]
    return df.merge(meta[keep], on="image_id", how="left")


def metric_column(model_type: str, metric: str) -> str:
    return f"{model_type}_cam_{metric}"


def summarize_spatial_by_class(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, true_label), group in all_df.groupby(["method", "true_label"], dropna=False):
        row = {"method": method, "true_label": true_label, "n": int(len(group))}
        for metric in SPATIAL_METRICS:
            multi_col = metric_column("multi", metric)
            ova_col = metric_column("ova", metric)
            if multi_col in group.columns and ova_col in group.columns:
                row[f"multi_{metric}"] = float(group[multi_col].mean())
                row[f"ova_{metric}"] = float(group[ova_col].mean())
                row[f"delta_multi_minus_ova_{metric}"] = float(group[multi_col].mean() - group[ova_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def artifact_levels(artifact_df: pd.DataFrame) -> pd.DataFrame:
    df = artifact_df.copy()
    for metric in ARTIFACT_METRICS:
        q75 = float(df[metric].quantile(0.75))
        q90 = float(df[metric].quantile(0.90))
        df[f"{metric}_high_q75"] = df[metric] > q75
        df[f"{metric}_very_high_q90"] = df[metric] > q90
    return df


def summarize_artifacts_by_class(artifact_df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    class_ref = reference[["image_path", "true_label"]].copy()
    class_ref["image_id"] = class_ref["image_path"].map(image_id_from_path)
    class_ref = class_ref.drop_duplicates("image_id")
    merged = artifact_df.merge(class_ref, on="image_id", how="left")

    rows = []
    for true_label, group in merged.groupby("true_label", dropna=False):
        row = {"true_label": true_label, "n": int(len(group))}
        for metric in ARTIFACT_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            high_col = f"{metric}_high_q75"
            very_high_col = f"{metric}_very_high_q90"
            if high_col in group.columns:
                row[f"{metric}_high_q75_frac"] = float(group[high_col].mean())
            if very_high_col in group.columns:
                row[f"{metric}_very_high_q90_frac"] = float(group[very_high_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_spatial_by_artifact(all_df: pd.DataFrame, artifact_df: pd.DataFrame) -> pd.DataFrame:
    merged = all_df.copy()
    merged["image_id"] = merged["image_path"].map(image_id_from_path)
    merged = merged.merge(artifact_df[["image_id"] + [f"{m}_high_q75" for m in ARTIFACT_METRICS]], on="image_id", how="left")

    rows = []
    for method, method_df in merged.groupby("method", dropna=False):
        for artifact in ARTIFACT_METRICS:
            flag = f"{artifact}_high_q75"
            for level_name, level_value in [("low_or_mid", False), ("high_q75", True)]:
                group = method_df[method_df[flag] == level_value]
                if group.empty:
                    continue
                row = {"method": method, "artifact_metric": artifact, "artifact_level": level_name, "n": int(len(group))}
                for metric in SPATIAL_METRICS:
                    multi_col = metric_column("multi", metric)
                    ova_col = metric_column("ova", metric)
                    if multi_col in group.columns and ova_col in group.columns:
                        row[f"multi_{metric}"] = float(group[multi_col].mean())
                        row[f"ova_{metric}"] = float(group[ova_col].mean())
                        row[f"delta_multi_minus_ova_{metric}"] = float(group[multi_col].mean() - group[ova_col].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_spatial_by_class_and_artifact(all_df: pd.DataFrame, artifact_df: pd.DataFrame) -> pd.DataFrame:
    merged = all_df.copy()
    merged["image_id"] = merged["image_path"].map(image_id_from_path)
    merged = merged.merge(artifact_df[["image_id"] + [f"{m}_high_q75" for m in ARTIFACT_METRICS]], on="image_id", how="left")

    rows = []
    for (method, true_label), class_df in merged.groupby(["method", "true_label"], dropna=False):
        for artifact in ARTIFACT_METRICS:
            flag = f"{artifact}_high_q75"
            high = class_df[class_df[flag] == True]
            low = class_df[class_df[flag] == False]
            if high.empty or low.empty:
                continue
            row = {
                "method": method,
                "true_label": true_label,
                "artifact_metric": artifact,
                "n_high_q75": int(len(high)),
                "n_low_or_mid": int(len(low)),
            }
            for metric in ["inside_frac", "outside_peritumor_r5_activation_frac", "top_mask_area_dice"]:
                multi_col = metric_column("multi", metric)
                ova_col = metric_column("ova", metric)
                if multi_col in class_df.columns and ova_col in class_df.columns:
                    row[f"multi_high_{metric}"] = float(high[multi_col].mean())
                    row[f"multi_low_mid_{metric}"] = float(low[multi_col].mean())
                    row[f"ova_high_{metric}"] = float(high[ova_col].mean())
                    row[f"ova_low_mid_{metric}"] = float(low[ova_col].mean())
                    row[f"ova_high_minus_low_mid_{metric}"] = float(high[ova_col].mean() - low[ova_col].mean())
                    row[f"multi_high_minus_low_mid_{metric}"] = float(high[multi_col].mean() - low[multi_col].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza atajos espaciales, proxies de artefactos y diferencias por clase en HAM10000."
    )
    parser.add_argument("--gradcam-index", default=DEFAULT_INPUTS["gradcam"])
    parser.add_argument("--gradcampp-index", default=DEFAULT_INPUTS["gradcampp"])
    parser.add_argument("--lrp-index", default=DEFAULT_INPUTS["lrp"])
    parser.add_argument("--metadata-csv", default="data/ham10000/raw/HAM10000_metadata")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        default="resultados_actualizados/explicabilidad/ham10000/analisis_atajos_artifacts_sampler_balanced",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "gradcam": read_index(args.gradcam_index, "gradcam"),
        "gradcampp": read_index(args.gradcampp_index, "gradcampp"),
        "lrp": read_index(args.lrp_index, "lrp"),
    }
    all_df = pd.concat(tables.values(), ignore_index=True)
    reference = tables["gradcampp"]

    artifacts = compute_artifacts(reference, args.image_size)
    artifacts = artifact_levels(add_metadata(artifacts, args.metadata_csv))
    artifacts.to_csv(output_dir / "artifact_metrics_by_image.csv", index=False)

    class_spatial = summarize_spatial_by_class(all_df)
    class_spatial.to_csv(output_dir / "spatial_shortcuts_by_class_contrast.csv", index=False)

    artifact_by_class = summarize_artifacts_by_class(artifacts, reference)
    artifact_by_class.to_csv(output_dir / "artifact_prevalence_by_class.csv", index=False)

    artifact_spatial = summarize_spatial_by_artifact(all_df, artifacts)
    artifact_spatial.to_csv(output_dir / "spatial_shortcuts_by_artifact_contrast.csv", index=False)

    class_artifact_spatial = summarize_spatial_by_class_and_artifact(all_df, artifacts)
    class_artifact_spatial.to_csv(output_dir / "spatial_shortcuts_by_class_and_artifact.csv", index=False)

    print(f"Saved artifact/spatial shortcut analysis to {output_dir}", flush=True)
    print("\nClass contrast, Grad-CAM++:")
    keep = [
        "method",
        "true_label",
        "n",
        "multi_inside_frac",
        "ova_inside_frac",
        "delta_multi_minus_ova_inside_frac",
        "multi_top_mask_area_dice",
        "ova_top_mask_area_dice",
    ]
    print(class_spatial[class_spatial["method"] == "gradcampp"][keep].round(3).to_string(index=False))

    print("\nArtifact contrast, Grad-CAM++:")
    keep = [
        "method",
        "artifact_metric",
        "artifact_level",
        "n",
        "multi_inside_frac",
        "ova_inside_frac",
        "delta_multi_minus_ova_inside_frac",
        "ova_outside_peritumor_r5_activation_frac",
    ]
    print(artifact_spatial[artifact_spatial["method"] == "gradcampp"][keep].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
