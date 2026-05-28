from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explicabilidad_gradcam_vgg import (
    CLASS_LABELS,
    brisc_mask_path,
    experiment_args,
    load_mask,
    predict_multi,
    predict_ova,
    prediction_outcome,
    test_image_paths,
)
from tfm.data import load_experiment_data
from tfm.experiment import train_multiclass_model, train_ova_models
from tfm.training import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BRISC tumor occlusion sensitivity for multi-output vs OVA.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="resultados_actualizados/explicabilidad/brisc/occlusion")
    parser.add_argument("--vgg-channels", type=int, nargs=3, default=[32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--brisc-root", default="/mnt/homeGPU/imhiguera/data/brisc2025")
    parser.add_argument("--brisc-segmentation-root", default="/mnt/homeGPU/imhiguera/data/brisc2025_segmentation")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--ring-radius",
        type=int,
        default=15,
        help="Radius in pixels used to build the surrounding ring for local mean filling.",
    )
    parser.add_argument(
        "--occlusion-radii",
        type=int,
        nargs="+",
        default=[0],
        help="Mask dilation radii to occlude. 0 occludes only the original tumor mask.",
    )
    return parser


def to_explicabilidad_args(parsed: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="brisc",
        seed=parsed.seed,
        vgg_channels=parsed.vgg_channels,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        early_stopping_patience=parsed.early_stopping_patience,
        early_stopping_min_delta=parsed.early_stopping_min_delta,
        learning_rate=parsed.learning_rate,
        image_size=parsed.image_size,
        brisc_root=parsed.brisc_root,
        brisc_segmentation_root=parsed.brisc_segmentation_root,
        tb_root="/mnt/homeGPU/imhiguera/data/tb_chest_xray",
        max_train=parsed.max_train,
        max_test=parsed.max_test,
    )


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    size = max(3, radius * 2 + 1)
    if size % 2 == 0:
        size += 1
    mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L")
    dilated = mask_image.filter(ImageFilter.MaxFilter(size=size))
    return np.asarray(dilated) > 0


def occlude_with_local_ring_mean(
    image: np.ndarray,
    mask: np.ndarray,
    ring_radius: int,
    occlusion_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    occluded = image.copy()
    occlusion_mask = dilate_mask(mask, occlusion_radius) if occlusion_radius > 0 else mask
    ring = np.logical_and(dilate_mask(occlusion_mask, ring_radius), ~occlusion_mask)

    for channel in range(image.shape[0]):
        channel_values = image[channel]
        if ring.any():
            fill_value = float(channel_values[ring].mean())
        else:
            fill_value = float(channel_values.mean())
        occluded[channel, occlusion_mask] = fill_value
    return occluded, occlusion_mask


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    groups = [
        ("all", lambda _row: "all"),
        ("occlusion_radius", lambda row: str(row["occlusion_radius"])),
        ("true_label", lambda row: str(row["true_label"])),
        (
            "true_label_occlusion_radius",
            lambda row: f"{row['true_label']}|r={row['occlusion_radius']}",
        ),
        ("outcome", lambda row: str(row["outcome"])),
        (
            "outcome_occlusion_radius",
            lambda row: f"{row['outcome']}|r={row['occlusion_radius']}",
        ),
    ]
    metrics = [
        "multi_drop_true_class",
        "ova_drop_true_class",
        "multi_delta_no_tumor",
        "ova_delta_no_tumor",
        "multi_pred_changed",
        "ova_pred_changed",
        "multi_correct_after_occlusion",
        "ova_correct_after_occlusion",
    ]

    summary_rows = []
    for group_name, key_fn in groups:
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault(key_fn(row), []).append(row)
        for group_value, group_rows in grouped.items():
            summary = {"group": group_name, "value": group_value, "n": len(group_rows)}
            for metric in metrics:
                summary[metric] = float(np.mean([float(row[metric]) for row in group_rows]))
            summary_rows.append(summary)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parsed = build_parser().parse_args()
    args = experiment_args(to_explicabilidad_args(parsed))
    set_seed(parsed.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(parsed.output_dir) / f"seed_{parsed.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading BRISC with seed={parsed.seed}", flush=True)
    experiment_data = load_experiment_data(args, parsed.seed)
    print(
        f"Training models on device={device}, train={len(experiment_data.y_train)}, "
        f"val={len(experiment_data.y_val)}, test={len(experiment_data.y_test)}",
        flush=True,
    )
    multi_model, _, _ = train_multiclass_model(experiment_data, args, device, parsed.seed)
    ova_models, _, _ = train_ova_models(experiment_data, args, device, parsed.seed)

    X_test = experiment_data.X_test
    y_true = experiment_data.y_test
    image_paths = test_image_paths(to_explicabilidad_args(parsed), len(y_true))
    mask_paths = [brisc_mask_path(to_explicabilidad_args(parsed), path) for path in image_paths]
    selected = [idx for idx, mask_path in enumerate(mask_paths) if mask_path is not None]
    labels = CLASS_LABELS["brisc"]
    no_tumor_idx = labels.index("no_tumor")

    print(f"Predicting original test set ({len(y_true)} images)", flush=True)
    multi_probs = predict_multi(multi_model, X_test, device)
    ova_probs = predict_ova(ova_models, X_test, device)
    multi_pred = multi_probs.argmax(axis=1)
    ova_pred = ova_probs.argmax(axis=1)

    print(
        f"Creating occluded images for {len(selected)} masked cases "
        f"and radii={parsed.occlusion_radii}",
        flush=True,
    )
    occluded_images = []
    occluded_meta: list[dict[str, object]] = []
    for idx in selected:
        width, height = X_test[idx].shape[-1], X_test[idx].shape[-2]
        mask = load_mask(mask_paths[idx], width, height)
        if mask is None:
            continue
        for occlusion_radius in parsed.occlusion_radii:
            occluded_image, occlusion_mask = occlude_with_local_ring_mean(
                X_test[idx],
                mask,
                parsed.ring_radius,
                occlusion_radius,
            )
            occluded_images.append(occluded_image)
            occluded_meta.append(
                {
                    "idx": idx,
                    "occlusion_radius": occlusion_radius,
                    "mask_area_frac": float(mask.mean()),
                    "occlusion_area_frac": float(occlusion_mask.mean()),
                }
            )
    X_occluded = np.asarray(occluded_images, dtype=np.float32)

    print("Predicting occluded images", flush=True)
    multi_probs_occ = predict_multi(multi_model, X_occluded, device)
    ova_probs_occ = predict_ova(ova_models, X_occluded, device)
    multi_pred_occ = multi_probs_occ.argmax(axis=1)
    ova_pred_occ = ova_probs_occ.argmax(axis=1)

    rows = []
    for row_idx, meta in enumerate(occluded_meta):
        idx = int(meta["idx"])
        target = int(y_true[idx])
        multi_pred_before = int(multi_pred[idx])
        ova_pred_before = int(ova_pred[idx])
        multi_pred_after = int(multi_pred_occ[row_idx])
        ova_pred_after = int(ova_pred_occ[row_idx])

        multi_true_original = float(multi_probs[idx, target])
        multi_true_occluded = float(multi_probs_occ[row_idx, target])
        ova_true_original = float(ova_probs[idx, target])
        ova_true_occluded = float(ova_probs_occ[row_idx, target])

        multi_no_tumor_original = float(multi_probs[idx, no_tumor_idx])
        multi_no_tumor_occluded = float(multi_probs_occ[row_idx, no_tumor_idx])
        ova_no_tumor_original = float(ova_probs[idx, no_tumor_idx])
        ova_no_tumor_occluded = float(ova_probs_occ[row_idx, no_tumor_idx])

        rows.append(
            {
                "test_index": idx,
                "occlusion_radius": int(meta["occlusion_radius"]),
                "mask_area_frac": float(meta["mask_area_frac"]),
                "occlusion_area_frac": float(meta["occlusion_area_frac"]),
                "true_label": labels[target],
                "multi_pred_original": labels[multi_pred_before],
                "ova_pred_original": labels[ova_pred_before],
                "multi_pred_occluded": labels[multi_pred_after],
                "ova_pred_occluded": labels[ova_pred_after],
                "outcome": prediction_outcome(target, multi_pred_before, ova_pred_before),
                "multi_true_prob_original": multi_true_original,
                "multi_true_prob_occluded": multi_true_occluded,
                "multi_drop_true_class": multi_true_original - multi_true_occluded,
                "ova_true_prob_original": ova_true_original,
                "ova_true_prob_occluded": ova_true_occluded,
                "ova_drop_true_class": ova_true_original - ova_true_occluded,
                "multi_no_tumor_prob_original": multi_no_tumor_original,
                "multi_no_tumor_prob_occluded": multi_no_tumor_occluded,
                "multi_delta_no_tumor": multi_no_tumor_occluded - multi_no_tumor_original,
                "ova_no_tumor_prob_original": ova_no_tumor_original,
                "ova_no_tumor_prob_occluded": ova_no_tumor_occluded,
                "ova_delta_no_tumor": ova_no_tumor_occluded - ova_no_tumor_original,
                "multi_pred_changed": float(multi_pred_before != multi_pred_after),
                "ova_pred_changed": float(ova_pred_before != ova_pred_after),
                "multi_correct_after_occlusion": float(multi_pred_after == target),
                "ova_correct_after_occlusion": float(ova_pred_after == target),
                "image_path": str(image_paths[idx]) if image_paths[idx] is not None else "",
                "mask_path": str(mask_paths[idx]) if mask_paths[idx] is not None else "",
            }
        )

    index_path = output_dir / "occlusion_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "occlusion_summary.csv"
    write_summary(rows, summary_path)

    print(f"Saved occlusion index to {index_path}", flush=True)
    print(f"Saved occlusion summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
