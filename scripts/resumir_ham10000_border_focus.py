from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("resultados_actualizados/explicabilidad/ham10000")

METHOD_PATTERNS = {
    "gradcam": "seed_*_vgg16_block5_sampler_balanced_briscmetrics_gradcam/gradcam_index.csv",
    "gradcampp": "seed_*_vgg16_block5_sampler_balanced_briscmetrics_gradcampp_gradcampp/gradcam_index.csv",
    "lrp": "seed_*_vgg16_block5_sampler_balanced_briscmetrics_lrp/gradcam_index.csv",
}

RADII = [5, 10, 15]


def seed_from_path(path: Path) -> int:
    return int(path.parent.name.split("_")[1])


def summarize() -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows = []
    class_rows = []
    for method, pattern in METHOD_PATTERNS.items():
        for path in sorted(BASE.glob(pattern)):
            seed = seed_from_path(path)
            df = pd.read_csv(path)
            for model in ["multi", "ova"]:
                row = {
                    "method": method,
                    "seed": seed,
                    "model_type": model,
                    "n": int(len(df)),
                    "inside_frac": float(df[f"{model}_cam_inside_frac"].mean()),
                    "outside_frac": float(df[f"{model}_cam_outside_frac"].mean()),
                    "top_mask_area_dice": float(df[f"{model}_cam_top_mask_area_dice"].mean()),
                }
                for radius in RADII:
                    peritumor_col = f"{model}_cam_peritumor_r{radius}_activation_frac"
                    outside_col = f"{model}_cam_outside_frac"
                    outside_peritumor_col = f"{model}_cam_outside_peritumor_r{radius}_activation_frac"
                    row[f"border_r{radius}_activation_frac"] = float(df[peritumor_col].mean())
                    row[f"border_r{radius}_outside_share"] = float(
                        (df[peritumor_col] / (df[outside_col] + 1e-8)).mean()
                    )
                    row[f"lesion_plus_border_r{radius}_activation_frac"] = float(
                        (df[f"{model}_cam_inside_frac"] + df[peritumor_col]).mean()
                    )
                    row[f"outside_peritumor_r{radius}_activation_frac"] = float(
                        df[outside_peritumor_col].mean()
                    )
                global_rows.append(row)

                for true_label, group in df.groupby("true_label", dropna=False):
                    class_row = {
                        "method": method,
                        "seed": seed,
                        "model_type": model,
                        "true_label": true_label,
                        "n": int(len(group)),
                        "inside_frac": float(group[f"{model}_cam_inside_frac"].mean()),
                        "outside_frac": float(group[f"{model}_cam_outside_frac"].mean()),
                        "top_mask_area_dice": float(group[f"{model}_cam_top_mask_area_dice"].mean()),
                    }
                    for radius in RADII:
                        peritumor_col = f"{model}_cam_peritumor_r{radius}_activation_frac"
                        outside_col = f"{model}_cam_outside_frac"
                        outside_peritumor_col = f"{model}_cam_outside_peritumor_r{radius}_activation_frac"
                        class_row[f"border_r{radius}_activation_frac"] = float(group[peritumor_col].mean())
                        class_row[f"border_r{radius}_outside_share"] = float(
                            (group[peritumor_col] / (group[outside_col] + 1e-8)).mean()
                        )
                        class_row[f"lesion_plus_border_r{radius}_activation_frac"] = float(
                            (group[f"{model}_cam_inside_frac"] + group[peritumor_col]).mean()
                        )
                        class_row[f"outside_peritumor_r{radius}_activation_frac"] = float(
                            group[outside_peritumor_col].mean()
                        )
                    class_rows.append(class_row)
    return pd.DataFrame(global_rows), pd.DataFrame(class_rows)


def main() -> None:
    output_dir = BASE / "analisis_border_focus_sampler_balanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    global_seed, class_seed = summarize()
    global_mean = global_seed.groupby(["method", "model_type"], as_index=False).mean(numeric_only=True)
    class_mean = class_seed.groupby(["method", "model_type", "true_label"], as_index=False).mean(numeric_only=True)

    global_seed.to_csv(output_dir / "border_focus_by_seed.csv", index=False)
    global_mean.to_csv(output_dir / "border_focus_global_mean.csv", index=False)
    class_seed.to_csv(output_dir / "border_focus_by_class_seed.csv", index=False)
    class_mean.to_csv(output_dir / "border_focus_by_class_mean.csv", index=False)

    keep = [
        "method",
        "model_type",
        "inside_frac",
        "border_r5_activation_frac",
        "border_r5_outside_share",
        "outside_peritumor_r5_activation_frac",
        "border_r10_activation_frac",
        "border_r10_outside_share",
        "outside_peritumor_r10_activation_frac",
        "border_r15_activation_frac",
        "border_r15_outside_share",
        "outside_peritumor_r15_activation_frac",
    ]
    print(f"Saved border focus summaries to {output_dir}")
    print(global_mean[keep].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
