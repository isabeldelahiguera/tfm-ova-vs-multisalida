from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


DEFAULT_INPUTS = {
    "gradcam": "resultados_actualizados/explicabilidad/ham10000/seed_1_vgg16_block5_sampler_balanced_gradcam_alltest/gradcam_index.csv",
    "gradcampp": "resultados_actualizados/explicabilidad/ham10000/seed_1_vgg16_block5_sampler_balanced_gradcampp_alltest_gradcampp/gradcam_index.csv",
    "lrp": "resultados_actualizados/explicabilidad/ham10000/seed_1_vgg16_block5_sampler_balanced_lrp_alltest/gradcam_index.csv",
}


MODEL_PREFIXES = {"multi": "multi_cam", "ova": "ova_cam"}


SPATIAL_METRICS = [
    "inside_frac",
    "outside_frac",
    "outside_peritumor_r5_activation_frac",
    "outside_peritumor_r5_mean",
    "mask_centroid_distance_norm",
    "active_area_frac_50",
    "gini",
    "pointing_game_hit",
    "top_mask_area_dice",
    "top_mask_area_iou",
]


def read_method_tables(inputs: dict[str, str]) -> dict[str, pd.DataFrame]:
    tables = {}
    for method, path in inputs.items():
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {method} index: {csv_path}")
        df = pd.read_csv(csv_path)
        df["method"] = method
        tables[method] = df
    return tables


def metric_column(model_type: str, metric: str) -> str:
    return f"{MODEL_PREFIXES[model_type]}_{metric}"


def summarize_group(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for group_values, group_df in df.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base = dict(zip(group_columns, group_values))
        for model_type in MODEL_PREFIXES:
            row = {**base, "model_type": model_type, "n": len(group_df)}
            for metric in SPATIAL_METRICS:
                column = metric_column(model_type, metric)
                if column in group_df.columns:
                    row[metric] = float(group_df[column].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def build_spatial_summaries(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat(tables.values(), ignore_index=True)
    global_summary = summarize_group(all_df, ["method"])
    class_summary = summarize_group(all_df, ["method", "true_label"])
    outcome_summary = summarize_group(all_df, ["method", "outcome"])
    return global_summary, class_summary, outcome_summary


def shortcut_candidates(reference: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = reference.copy()
    df["inside_delta_multi_minus_ova"] = df["multi_cam_inside_frac"] - df["ova_cam_inside_frac"]
    df["outside_delta_ova_minus_multi"] = df["ova_cam_outside_frac"] - df["multi_cam_outside_frac"]
    df["peritumor_delta_ova_minus_multi"] = (
        df["ova_cam_outside_peritumor_r5_activation_frac"]
        - df["multi_cam_outside_peritumor_r5_activation_frac"]
    )
    df["ova_low_inside_rank"] = df["ova_cam_inside_frac"].rank(method="first", ascending=True)

    columns = [
        "test_index",
        "true_label",
        "multi_pred",
        "ova_pred",
        "outcome",
        "multi_confidence",
        "ova_confidence",
        "multi_cam_inside_frac",
        "ova_cam_inside_frac",
        "inside_delta_multi_minus_ova",
        "multi_cam_outside_frac",
        "ova_cam_outside_frac",
        "outside_delta_ova_minus_multi",
        "multi_cam_outside_peritumor_r5_activation_frac",
        "ova_cam_outside_peritumor_r5_activation_frac",
        "peritumor_delta_ova_minus_multi",
        "multi_cam_mask_centroid_distance_norm",
        "ova_cam_mask_centroid_distance_norm",
        "image_path",
        "mask_path",
    ]
    return (
        df.sort_values(
            ["ova_cam_inside_frac", "inside_delta_multi_minus_ova", "ova_cam_outside_frac"],
            ascending=[True, False, False],
        )[columns]
        .head(top_n)
        .reset_index(drop=True)
    )


def overlay_mask(image: Image.Image, mask_path: str | float | None) -> Image.Image:
    image = image.convert("RGB")
    if not isinstance(mask_path, str) or not mask_path:
        return image
    path = Path(mask_path)
    if not path.exists():
        return image
    mask = Image.open(path).convert("L").resize(image.size, Image.NEAREST)
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    alpha = mask.point(lambda value: 80 if value > 0 else 0)
    overlay.putalpha(alpha)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def make_contact_sheet(candidates: pd.DataFrame, output_path: Path, top_n: int, thumb_size: int = 180) -> None:
    selected = candidates.head(top_n)
    if selected.empty:
        return

    cols = 5
    label_height = 54
    gap = 8
    rows = int(np.ceil(len(selected) / cols))
    sheet = Image.new(
        "RGB",
        (cols * thumb_size + (cols + 1) * gap, rows * (thumb_size + label_height) + (rows + 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for offset, row in enumerate(selected.itertuples(index=False)):
        image_path = Path(row.image_path)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        image = overlay_mask(image, row.mask_path)
        image.thumbnail((thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
        canvas.paste(image, ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2))

        col = offset % cols
        grid_row = offset // cols
        x = gap + col * (thumb_size + gap)
        y = gap + grid_row * (thumb_size + label_height + gap)
        sheet.paste(canvas, (x, y))

        label = (
            f"idx {row.test_index} {row.true_label}\n"
            f"M:{row.multi_pred} O:{row.ova_pred}\n"
            f"O in={row.ova_cam_inside_frac:.2f} d={row.inside_delta_multi_minus_ova:.2f}"
        )
        draw.text((x, y + thumb_size + 3), label, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis de posibles atajos en HAM10000.")
    parser.add_argument("--gradcam-index", default=DEFAULT_INPUTS["gradcam"])
    parser.add_argument("--gradcampp-index", default=DEFAULT_INPUTS["gradcampp"])
    parser.add_argument("--lrp-index", default=DEFAULT_INPUTS["lrp"])
    parser.add_argument(
        "--output-dir",
        default="resultados_actualizados/explicabilidad/ham10000/analisis_atajos_sampler_balanced",
    )
    parser.add_argument("--top-candidates", type=int, default=50)
    parser.add_argument("--contact-sheet-cases", type=int, default=25)
    args = parser.parse_args()

    inputs = {
        "gradcam": args.gradcam_index,
        "gradcampp": args.gradcampp_index,
        "lrp": args.lrp_index,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = read_method_tables(inputs)
    global_summary, class_summary, outcome_summary = build_spatial_summaries(tables)
    global_summary.to_csv(output_dir / "spatial_shortcut_global_summary.csv", index=False)
    class_summary.to_csv(output_dir / "spatial_shortcut_by_class.csv", index=False)
    outcome_summary.to_csv(output_dir / "spatial_shortcut_by_outcome.csv", index=False)

    reference = tables["gradcampp"]
    candidates = shortcut_candidates(reference, args.top_candidates)
    candidates.to_csv(output_dir / "ova_shortcut_candidate_cases_gradcampp.csv", index=False)
    make_contact_sheet(
        candidates,
        output_dir / "ova_shortcut_candidate_contact_sheet_top25.png",
        args.contact_sheet_cases,
    )

    print(f"Saved shortcut analysis to {output_dir}")
    print("\nGlobal summary:")
    keep = ["method", "model_type", "n", "inside_frac", "outside_frac", "outside_peritumor_r5_activation_frac", "pointing_game_hit", "top_mask_area_dice", "top_mask_area_iou"]
    print(global_summary[keep].round(3).to_string(index=False))
    print("\nTop candidate cases:")
    print(
        candidates[
            [
                "test_index",
                "true_label",
                "multi_pred",
                "ova_pred",
                "outcome",
                "multi_cam_inside_frac",
                "ova_cam_inside_frac",
                "inside_delta_multi_minus_ova",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
