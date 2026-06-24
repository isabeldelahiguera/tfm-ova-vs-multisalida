from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, disk


CLASS_LABELS = ["glioma", "meningioma", "pituitary", "no_tumor"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PLANE_NAMES = {
    "ax": "axial",
    "co": "coronal",
    "sa": "sagittal",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Análisis descriptivo de BRISC por split, clase y plano para estudiar "
            "posibles sesgos o atajos usados por los modelos."
        )
    )
    parser.add_argument("--brisc-root", default="/mnt/homeGPU/imhiguera/data/brisc2025")
    parser.add_argument(
        "--brisc-segmentation-root",
        default="/mnt/homeGPU/imhiguera/data/brisc2025_segmentation",
    )
    parser.add_argument(
        "--output-dir",
        default="resultados_actualizados/analisis_dataset/brisc",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--peritumor-radius", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limita el numero de imagenes procesadas para pruebas rapidas.",
    )
    parser.add_argument(
        "--peritumor-radii",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Radios peritumorales a analizar. Si no se indica, se usa "
            "--peritumor-radius para mantener compatibilidad."
        ),
    )
    return parser


def plane_from_name(path: Path) -> str:
    match = re.search(r"_(ax|co|sa)_", path.stem.lower())
    return match.group(1) if match else "unknown"


def mask_path_for(segmentation_root: Path, split: str, image_path: Path) -> Path | None:
    mask_path = segmentation_root / split / "masks" / f"{image_path.stem}.png"
    return mask_path if mask_path.exists() else None


def iter_images(brisc_root: Path, splits: list[str]) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for split in splits:
        for class_name in CLASS_LABELS:
            class_dir = brisc_root / split / class_name
            if not class_dir.exists():
                continue
            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            rows.extend((split, class_name, path) for path in image_paths)
    return rows


def read_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def read_mask(path: Path | None, image_size: int) -> np.ndarray | None:
    if path is None:
        return None
    with Image.open(path) as image:
        image = image.convert("L").resize((image_size, image_size), Image.NEAREST)
        return np.asarray(image, dtype=np.float32) > 0


def dilate_bool_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    return binary_dilation(mask, footprint=disk(radius))


def bbox_metrics(mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    regions = regionprops(mask.astype(np.uint8))
    if not regions:
        return {}
    region = regions[0]
    height, width = mask.shape
    y_min, x_min, y_max_exclusive, x_max_exclusive = region.bbox
    x_max = int(x_max_exclusive - 1)
    y_max = int(y_max_exclusive - 1)
    bbox_height, bbox_width = region.image.shape
    bbox_area = float(region.area_bbox)
    area = float(region.area)
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    centroid_y, centroid_x = region.centroid
    centroid_x = float(centroid_x)
    centroid_y = float(centroid_y)
    diagonal = float(np.hypot(width, height))
    bbox_center_x = (x_min + x_max) / 2
    bbox_center_y = (y_min + y_max) / 2
    perimeter = float(region.perimeter)
    compactness = (4 * np.pi * area) / max(1.0, perimeter * perimeter)

    return {
        "mask_centroid_x_norm": float(centroid_x / max(1, width - 1)),
        "mask_centroid_y_norm": float(centroid_y / max(1, height - 1)),
        "mask_centroid_distance_center_norm": float(
            np.hypot(centroid_x - center_x, centroid_y - center_y) / max(1.0, diagonal)
        ),
        "mask_bbox_x_min_norm": float(x_min / max(1, width - 1)),
        "mask_bbox_y_min_norm": float(y_min / max(1, height - 1)),
        "mask_bbox_x_max_norm": float(x_max / max(1, width - 1)),
        "mask_bbox_y_max_norm": float(y_max / max(1, height - 1)),
        "mask_bbox_center_x_norm": float(bbox_center_x / max(1, width - 1)),
        "mask_bbox_center_y_norm": float(bbox_center_y / max(1, height - 1)),
        "mask_bbox_center_distance_center_norm": float(
            np.hypot(bbox_center_x - center_x, bbox_center_y - center_y) / max(1.0, diagonal)
        ),
        "mask_bbox_width_frac": float(bbox_width / width),
        "mask_bbox_height_frac": float(bbox_height / height),
        "mask_bbox_area_frac": float(bbox_area / (width * height)),
        "mask_bbox_aspect_width_height": float(bbox_width / max(1, bbox_height)),
        "mask_bbox_fill_frac": float(region.extent),
        "mask_equivalent_diameter_frac": float(region.equivalent_diameter_area / max(width, height)),
        "mask_perimeter_frac": float(perimeter / max(1, 2 * (width + height))),
        "mask_compactness": float(compactness),
    }


def region_stats(prefix: str, image: np.ndarray, region: np.ndarray) -> dict[str, float]:
    if not region.any():
        return {
            f"{prefix}_area_frac": 0.0,
            f"{prefix}_intensity_mean": np.nan,
            f"{prefix}_intensity_std": np.nan,
            f"{prefix}_intensity_p10": np.nan,
            f"{prefix}_intensity_p50": np.nan,
            f"{prefix}_intensity_p90": np.nan,
        }
    values = image[region]
    return {
        f"{prefix}_area_frac": float(region.mean()),
        f"{prefix}_intensity_mean": float(values.mean()),
        f"{prefix}_intensity_std": float(values.std()),
        f"{prefix}_intensity_p10": float(np.percentile(values, 10)),
        f"{prefix}_intensity_p50": float(np.percentile(values, 50)),
        f"{prefix}_intensity_p90": float(np.percentile(values, 90)),
    }


def image_metrics(image: np.ndarray) -> dict[str, float]:
    nonzero = image > 0.02
    metrics = {
        "image_intensity_mean": float(image.mean()),
        "image_intensity_std": float(image.std()),
        "image_intensity_p10": float(np.percentile(image, 10)),
        "image_intensity_p50": float(np.percentile(image, 50)),
        "image_intensity_p90": float(np.percentile(image, 90)),
        "image_nonzero_area_frac": float(nonzero.mean()),
    }
    if nonzero.any():
        values = image[nonzero]
        yy, xx = np.where(nonzero)
        metrics.update(
            {
                "image_nonzero_intensity_mean": float(values.mean()),
                "image_nonzero_intensity_std": float(values.std()),
                "image_nonzero_centroid_x_norm": float(xx.mean() / max(1, image.shape[1] - 1)),
                "image_nonzero_centroid_y_norm": float(yy.mean() / max(1, image.shape[0] - 1)),
            }
        )
    else:
        metrics.update(
            {
                "image_nonzero_intensity_mean": np.nan,
                "image_nonzero_intensity_std": np.nan,
                "image_nonzero_centroid_x_norm": np.nan,
                "image_nonzero_centroid_y_norm": np.nan,
            }
        )
    return metrics


def analyze_image(
    split: str,
    true_label: str,
    image_path: Path,
    segmentation_root: Path,
    image_size: int,
    peritumor_radii: list[int],
) -> dict[str, object]:
    plane = plane_from_name(image_path)
    mask_path = mask_path_for(segmentation_root, split, image_path)
    image = read_image(image_path, image_size)
    mask = read_mask(mask_path, image_size)

    row: dict[str, object] = {
        "split": split,
        "true_label": true_label,
        "plane": plane,
        "plane_name": PLANE_NAMES.get(plane, "unknown"),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path is not None else "",
        "has_mask": int(mask is not None and mask.any()),
    }
    row.update(image_metrics(image))

    if mask is None or not mask.any():
        return row

    row.update(bbox_metrics(mask))
    row.update(region_stats("tumor", image, mask))

    for peritumor_radius in peritumor_radii:
        peritumor = np.logical_and(dilate_bool_mask(mask, peritumor_radius), ~mask)
        outside = ~(mask | peritumor)
        peritumor_prefix = f"peritumor_r{peritumor_radius}"
        outside_prefix = f"outside_peritumor_r{peritumor_radius}"
        row.update(region_stats(peritumor_prefix, image, peritumor))
        row.update(region_stats(outside_prefix, image, outside))
        row[f"tumor_vs_outside_r{peritumor_radius}_mean_diff"] = (
            row["tumor_intensity_mean"] - row[f"{outside_prefix}_intensity_mean"]
        )
        row[f"tumor_vs_peritumor_r{peritumor_radius}_mean_diff"] = (
            row["tumor_intensity_mean"] - row[f"{peritumor_prefix}_intensity_mean"]
        )
        row[f"tumor_vs_outside_r{peritumor_radius}_mean_ratio"] = (
            row["tumor_intensity_mean"] / max(row[f"{outside_prefix}_intensity_mean"], 1e-8)
        )
        row[f"tumor_vs_peritumor_r{peritumor_radius}_mean_ratio"] = (
            row["tumor_intensity_mean"] / max(row[f"{peritumor_prefix}_intensity_mean"], 1e-8)
        )

    if 5 in peritumor_radii:
        row["tumor_vs_outside_mean_diff"] = row["tumor_vs_outside_r5_mean_diff"]
        row["tumor_vs_peritumor_mean_diff"] = row["tumor_vs_peritumor_r5_mean_diff"]
    return row


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in {"has_mask"}
    ]
    grouped = df.groupby(group_cols, dropna=False)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n"] = int(len(group))
        row["mask_available_frac"] = float(group["has_mask"].mean())
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{col}_median"] = float(values.median()) if values.notna().any() else np.nan
            row[f"{col}_min"] = float(values.min()) if values.notna().any() else np.nan
            row[f"{col}_max"] = float(values.max()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_train_test_diffs(summary: pd.DataFrame, output_path: Path) -> None:
    key_cols = ["true_label", "plane"]
    if not set(["split", *key_cols]).issubset(summary.columns):
        return
    metrics = [
        "tumor_area_frac_mean",
        "mask_centroid_x_norm_mean",
        "mask_centroid_y_norm_mean",
        "mask_centroid_distance_center_norm_mean",
        "mask_bbox_width_frac_mean",
        "mask_bbox_height_frac_mean",
        "mask_bbox_fill_frac_mean",
        "mask_compactness_mean",
        "image_nonzero_area_frac_mean",
        "tumor_intensity_mean_mean",
        "outside_peritumor_r5_intensity_mean_mean",
        "tumor_vs_outside_r5_mean_diff_mean",
        "tumor_vs_peritumor_r5_mean_diff_mean",
    ]
    available = [metric for metric in metrics if metric in summary.columns]
    if not available:
        return
    rows = []
    for keys, group in summary.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_split = {str(row["split"]): row for _, row in group.iterrows()}
        if "train" not in by_split or "test" not in by_split:
            continue
        row = {col: value for col, value in zip(key_cols, keys)}
        row["train_n"] = int(by_split["train"]["n"])
        row["test_n"] = int(by_split["test"]["n"])
        for metric in available:
            train_value = float(by_split["train"][metric])
            test_value = float(by_split["test"][metric])
            row[f"train_{metric}"] = train_value
            row[f"test_{metric}"] = test_value
            row[f"diff_test_minus_train_{metric}"] = test_value - train_value
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_feature_contrasts(
    df: pd.DataFrame,
    group_cols: list[str],
    output_path: Path,
    *,
    split: str = "train",
) -> None:
    feature_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and col not in {"has_mask"}
        and not col.endswith("_train_z")
        and not col.endswith("_abs_train_z")
    ]
    source = df[df["split"] == split].copy()
    rows = []
    for group_values, group in source.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_mask = np.ones(len(source), dtype=bool)
        for col, value in zip(group_cols, group_values):
            group_mask &= source[col].to_numpy() == value
        rest = source.loc[~group_mask]
        if rest.empty:
            continue

        base = {col: value for col, value in zip(group_cols, group_values)}
        base["split"] = split
        base["group_n"] = int(len(group))
        base["rest_n"] = int(len(rest))
        for feature in feature_cols:
            group_values_numeric = pd.to_numeric(group[feature], errors="coerce").dropna()
            rest_values_numeric = pd.to_numeric(rest[feature], errors="coerce").dropna()
            if len(group_values_numeric) < 2 or len(rest_values_numeric) < 2:
                continue
            group_mean = float(group_values_numeric.mean())
            rest_mean = float(rest_values_numeric.mean())
            group_std = float(group_values_numeric.std(ddof=1))
            rest_std = float(rest_values_numeric.std(ddof=1))
            pooled_std = float(np.sqrt((group_std**2 + rest_std**2) / 2))
            rows.append(
                {
                    **base,
                    "feature": feature,
                    "group_mean": group_mean,
                    "rest_mean": rest_mean,
                    "diff_group_minus_rest": group_mean - rest_mean,
                    "standardized_diff": (
                        (group_mean - rest_mean) / pooled_std if pooled_std > 0 else np.nan
                    ),
                    "abs_standardized_diff": abs((group_mean - rest_mean) / pooled_std)
                    if pooled_std > 0
                    else np.nan,
                }
            )

    contrasts = pd.DataFrame(rows)
    if not contrasts.empty:
        contrasts = contrasts.sort_values("abs_standardized_diff", ascending=False)
    contrasts.to_csv(output_path, index=False)


def main() -> None:
    args = build_parser().parse_args()
    brisc_root = Path(args.brisc_root).expanduser()
    segmentation_root = Path(args.brisc_segmentation_root).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    peritumor_radii = args.peritumor_radii or [args.peritumor_radius]
    peritumor_radii = sorted(set(peritumor_radii))

    image_rows = iter_images(brisc_root, args.splits)
    if args.max_images is not None and args.max_images > 0:
        image_rows = image_rows[: args.max_images]
    rows = []
    for idx, (split, true_label, image_path) in enumerate(image_rows, start=1):
        rows.append(
            analyze_image(
                split,
                true_label,
                image_path,
                segmentation_root,
                args.image_size,
                peritumor_radii,
            )
        )
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"Analizadas {idx}/{len(image_rows)} imagenes", flush=True)

    per_image_path = output_dir / "brisc_dataset_por_imagen.csv"
    with per_image_path.open("w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        ordered = [
            "split",
            "true_label",
            "plane",
            "plane_name",
            "has_mask",
            "image_path",
            "mask_path",
        ]
        fieldnames = ordered + [key for key in fieldnames if key not in ordered]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    summary_specs = {
        "conteo_por_split_clase_plano.csv": ["split", "true_label", "plane", "plane_name"],
        "resumen_por_split_clase.csv": ["split", "true_label"],
        "resumen_por_split_plano.csv": ["split", "plane", "plane_name"],
        "resumen_por_split_clase_plano.csv": ["split", "true_label", "plane", "plane_name"],
        "resumen_por_clase_plano.csv": ["true_label", "plane", "plane_name"],
    }
    summaries: dict[str, pd.DataFrame] = {}
    for filename, group_cols in summary_specs.items():
        summary = summarize(df, group_cols)
        summaries[filename] = summary
        summary.to_csv(output_dir / filename, index=False)

    train_df = df[df["split"] == "train"].copy()
    summarize(train_df, ["true_label"]).to_csv(output_dir / "resumen_train_por_clase.csv", index=False)
    summarize(train_df, ["plane", "plane_name"]).to_csv(output_dir / "resumen_train_por_plano.csv", index=False)
    summarize(train_df, ["true_label", "plane", "plane_name"]).to_csv(
        output_dir / "resumen_train_por_clase_plano.csv",
        index=False,
    )

    write_feature_contrasts(
        df,
        ["true_label"],
        output_dir / "contrastes_train_por_clase.csv",
    )
    write_feature_contrasts(
        df,
        ["plane", "plane_name"],
        output_dir / "contrastes_train_por_plano.csv",
    )
    write_feature_contrasts(
        df,
        ["true_label", "plane", "plane_name"],
        output_dir / "contrastes_train_por_clase_plano.csv",
    )

    write_train_test_diffs(
        summaries["resumen_por_split_clase_plano.csv"],
        output_dir / "comparacion_train_test_por_clase_plano.csv",
    )

    print(f"Imagenes analizadas: {len(df)}", flush=True)
    print(f"Resultados guardados en: {output_dir}", flush=True)
    print(f"Radios peritumorales: {peritumor_radii}", flush=True)


if __name__ == "__main__":
    main()
