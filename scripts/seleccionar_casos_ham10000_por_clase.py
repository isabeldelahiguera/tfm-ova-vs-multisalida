from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INDEX = (
    "resultados_actualizados/explicabilidad/ham10000/"
    "seed_1_vgg16_block5_sampler_balanced_gradcampp_alltest_gradcampp/gradcam_index.csv"
)


CASE_GROUPS = {
    "ova_correct_low_inside": lambda df: df[(df["ova_pred"] == df["true_label"])],
    "both_correct_low_inside": lambda df: df[(df["ova_pred"] == df["true_label"]) & (df["multi_pred"] == df["true_label"])],
    "ova_wrong_low_inside": lambda df: df[(df["ova_pred"] != df["true_label"])],
    "ova_predicts_class_low_inside": lambda df: df.copy(),
    "multi_correct_ova_wrong": lambda df: df[(df["multi_pred"] == df["true_label"]) & (df["ova_pred"] != df["true_label"])],
}


METRIC_COLUMNS = [
    "test_index",
    "true_label",
    "multi_pred",
    "ova_pred",
    "outcome",
    "multi_confidence",
    "ova_confidence",
    "multi_cam_inside_frac",
    "ova_cam_inside_frac",
    "multi_cam_outside_frac",
    "ova_cam_outside_frac",
    "multi_cam_peritumor_r5_activation_frac",
    "ova_cam_peritumor_r5_activation_frac",
    "multi_cam_outside_peritumor_r5_activation_frac",
    "ova_cam_outside_peritumor_r5_activation_frac",
    "multi_cam_top_mask_area_dice",
    "ova_cam_top_mask_area_dice",
    "multi_cam_mask_centroid_distance_norm",
    "ova_cam_mask_centroid_distance_norm",
    "image_path",
    "mask_path",
]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["inside_delta_multi_minus_ova"] = df["multi_cam_inside_frac"] - df["ova_cam_inside_frac"]
    df["outside_delta_ova_minus_multi"] = df["ova_cam_outside_frac"] - df["multi_cam_outside_frac"]
    df["peritumor_delta_ova_minus_multi"] = (
        df["ova_cam_peritumor_r5_activation_frac"] - df["multi_cam_peritumor_r5_activation_frac"]
    )
    df["outside_peritumor_delta_ova_minus_multi"] = (
        df["ova_cam_outside_peritumor_r5_activation_frac"]
        - df["multi_cam_outside_peritumor_r5_activation_frac"]
    )
    df["dice_delta_multi_minus_ova"] = df["multi_cam_top_mask_area_dice"] - df["ova_cam_top_mask_area_dice"]
    return df


def select_true_class_cases(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    rows = []
    for true_label, class_df in df.groupby("true_label", dropna=False):
        for group_name, selector in CASE_GROUPS.items():
            if group_name == "ova_predicts_class_low_inside":
                candidates = df[df["ova_pred"] == true_label].copy()
            else:
                candidates = selector(class_df).copy()
            if candidates.empty:
                continue
            candidates["selection_group"] = group_name
            candidates["selection_class"] = true_label
            candidates = candidates.sort_values(
                [
                    "ova_cam_inside_frac",
                    "ova_cam_outside_peritumor_r5_activation_frac",
                    "inside_delta_multi_minus_ova",
                ],
                ascending=[True, False, False],
            )
            rows.append(candidates.head(per_class))
    if not rows:
        return pd.DataFrame()
    selected = pd.concat(rows, ignore_index=True)
    selected = add_derived_columns(selected)
    columns = ["selection_class", "selection_group"] + [
        column for column in METRIC_COLUMNS if column in selected.columns
    ] + [
        "inside_delta_multi_minus_ova",
        "outside_delta_ova_minus_multi",
        "peritumor_delta_ova_minus_multi",
        "outside_peritumor_delta_ova_minus_multi",
        "dice_delta_multi_minus_ova",
    ]
    return selected[columns]


def summarize_predicted_class_focus(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predicted_class, group in df.groupby("ova_pred", dropna=False):
        row = {
            "ova_predicted_class": predicted_class,
            "n": int(len(group)),
            "true_class_frac": float((group["true_label"] == predicted_class).mean()),
            "ova_inside_frac": float(group["ova_cam_inside_frac"].mean()),
            "ova_peritumor_r5_activation_frac": float(group["ova_cam_peritumor_r5_activation_frac"].mean()),
            "ova_outside_peritumor_r5_activation_frac": float(
                group["ova_cam_outside_peritumor_r5_activation_frac"].mean()
            ),
            "ova_top_mask_area_dice": float(group["ova_cam_top_mask_area_dice"].mean()),
            "ova_mask_centroid_distance_norm": float(group["ova_cam_mask_centroid_distance_norm"].mean()),
            "multi_inside_frac": float(group["multi_cam_inside_frac"].mean()),
            "multi_top_mask_area_dice": float(group["multi_cam_top_mask_area_dice"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ova_predicted_class")


def build_image_index_argument(cases: pd.DataFrame, selection_class: str, selection_group: str, max_cases: int) -> str:
    subset = cases[
        (cases["selection_class"] == selection_class)
        & (cases["selection_group"] == selection_group)
    ].head(max_cases)
    return " ".join(str(int(idx)) for idx in subset["test_index"].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selecciona casos por clase para inspeccionar en que se centra OVA en HAM10000."
    )
    parser.add_argument("--gradcampp-index", default=DEFAULT_INDEX)
    parser.add_argument(
        "--output-dir",
        default="resultados_actualizados/explicabilidad/ham10000/analisis_por_clase_ova_sampler_balanced",
    )
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--command-group", default="ova_correct_low_inside")
    parser.add_argument("--command-max-cases", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.gradcampp_index)
    df = add_derived_columns(df)

    focus = summarize_predicted_class_focus(df)
    focus.to_csv(output_dir / "ova_focus_by_predicted_class.csv", index=False)

    cases = select_true_class_cases(df, args.per_class)
    cases.to_csv(output_dir / "ova_class_case_selection.csv", index=False)

    command_rows = []
    for selection_class in sorted(cases["selection_class"].dropna().unique()):
        indices = build_image_index_argument(cases, selection_class, args.command_group, args.command_max_cases)
        if not indices:
            continue
        command_rows.append(
            {
                "selection_class": selection_class,
                "selection_group": args.command_group,
                "image_indices": indices,
                "command": (
                    "python scripts/explicabilidad_gradcam_vgg.py "
                    "--dataset ham10000 "
                    "--model-arch vgg16-pretrained "
                    "--pretrained-finetune block5 "
                    "--train-sampler balanced "
                    "--class-weighting none "
                    "--image-size 224 "
                    "--batch-size 16 "
                    "--cam-method gradcam++ "
                    "--cam-target predicted "
                    "--selection ordered "
                    "--require-mask "
                    f"--image-indices {indices}"
                ),
            }
        )
    pd.DataFrame(command_rows).to_csv(output_dir / "commands_to_render_selected_cases.csv", index=False)

    print(f"Saved class OVA focus analysis to {output_dir}", flush=True)
    print("\nOVA focus by predicted class:")
    print(focus.round(3).to_string(index=False))
    print("\nSelected cases:")
    show_cols = [
        "selection_class",
        "selection_group",
        "test_index",
        "true_label",
        "multi_pred",
        "ova_pred",
        "outcome",
        "ova_cam_inside_frac",
        "ova_cam_peritumor_r5_activation_frac",
        "ova_cam_outside_peritumor_r5_activation_frac",
        "inside_delta_multi_minus_ova",
    ]
    print(cases[show_cols].head(40).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
