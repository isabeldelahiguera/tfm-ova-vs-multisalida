from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox

from checkpoint_utils import checkpoints_exist, load_model_checkpoints, save_model_checkpoints
from explicabilidad_gradcam_vgg import (
    CLASS_LABELS,
    balanced_indices,
    cam_mask_metrics,
    contrast_indices,
    experiment_args,
    image_to_pil,
    load_mask,
    make_panel,
    ordered_indices,
    outcome_indices,
    overlay_heatmap,
    predict_multi,
    predict_ova,
    prefixed_metrics,
    segmentation_mask_path,
    test_image_paths,
    write_metrics_summary,
)
from tfm.data import load_experiment_data
from tfm.experiment import train_multiclass_model, train_ova_models
from tfm.training import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Zennit LRP explanations for VGG multi-output and OVA."
    )
    parser.add_argument("--dataset", choices=["brisc", "tb_chest_xray", "ham10000"], required=True)
    parser.add_argument(
        "--model-arch",
        choices=["vgg", "vgg16-pretrained", "vit-b-16-pretrained"],
        default="vgg",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--image-indices", type=int, nargs="*", default=None)
    parser.add_argument("--selection", choices=["all", "balanced", "ordered", "contrast", "outcomes"], default="balanced")
    parser.add_argument("--lrp-target", choices=["predicted", "true"], default="predicted")
    parser.add_argument(
        "--lrp-epsilon",
        type=float,
        default=1e-6,
        help="Epsilon stabilizer used by Zennit's EpsilonGammaBox composite.",
    )
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--output-dir", default="resultados_actualizados/explicabilidad")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--vgg-channels", type=int, nargs=3, default=[32, 64, 128])
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    parser.add_argument("--data-augmentation", choices=["none", "ham10000-basic"], default="none")
    parser.add_argument("--train-sampler", choices=["none", "balanced"], default="none")
    parser.add_argument(
        "--pretrained-finetune",
        choices=["frozen", "block5", "last-block", "full", "all"],
        default="frozen",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--cam-top-percents", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--cam-mask-area-factors", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--peritumor-radii", type=int, nargs="+", default=[5])
    parser.add_argument("--brisc-root", default="/mnt/homeGPU/imhiguera/data/brisc2025")
    parser.add_argument("--brisc-segmentation-root", default="/mnt/homeGPU/imhiguera/data/brisc2025_segmentation")
    parser.add_argument("--tb-root", default="/mnt/homeGPU/imhiguera/data/tb_chest_xray")
    parser.add_argument("--ham10000-root", default="/mnt/homeGPU/imhiguera/data/ham10000")
    parser.add_argument("--ham10000-test", choices=["internal", "official"], default="internal")
    parser.add_argument("--ham10000-split-csv", default=None)
    parser.add_argument("--ham10000-split-seed", type=int, default=2000)
    parser.add_argument("--ham10000-exclude-classes", type=str, nargs="*", default=[])
    parser.add_argument(
        "--ham10000-label-mode",
        choices=["original", "malignant_binary"],
        default="original",
        help=(
            "HAM10000 label formulation. 'malignant_binary' evaluates the "
            "first level of the nested experiment: malignant vs non_malignant."
        ),
    )
    parser.add_argument(
        "--ham10000-mask-root",
        default="/mnt/homeGPU/imhiguera/data/ham10000/masks/HAM10000_segmentations_lesion_tschandl",
    )
    parser.add_argument("--require-mask", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    return parser


def disable_inplace_relu(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def target_relevance(target_score_index: int):
    def relevance(output: torch.Tensor) -> torch.Tensor:
        selected = torch.zeros_like(output)
        if output.ndim == 1:
            selected[target_score_index] = output[target_score_index]
        else:
            selected[:, target_score_index] = output[:, target_score_index]
        return selected

    return relevance


def lrp_heatmap(
    model: nn.Module,
    image: np.ndarray,
    target_score_index: int,
    device: torch.device,
    epsilon: float,
) -> np.ndarray:
    model.eval()
    disable_inplace_relu(model)
    tensor = torch.tensor(image[None, ...], dtype=torch.float32, device=device)
    try:
        composite = EpsilonGammaBox(low=0.0, high=1.0, epsilon=epsilon)
    except TypeError:
        composite = EpsilonGammaBox(low=0.0, high=1.0)
    with Gradient(model, composite=composite) as attributor:
        _output, relevance = attributor(tensor, target_relevance(target_score_index))

    heatmap = relevance.detach().sum(dim=1).squeeze(0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap - heatmap.min()
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap.astype(np.float32)


def lrp_experiment_args(parsed: argparse.Namespace) -> SimpleNamespace:
    return experiment_args(parsed)


def main() -> None:
    parsed = build_parser().parse_args()
    if any(percent <= 0 or percent >= 100 for percent in parsed.cam_top_percents):
        raise ValueError("--cam-top-percents values must be between 1 and 99.")
    if any(factor <= 0 for factor in parsed.cam_mask_area_factors):
        raise ValueError("--cam-mask-area-factors values must be positive.")

    args = lrp_experiment_args(parsed)
    set_seed(parsed.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {parsed.dataset} with seed={parsed.seed}", flush=True)
    experiment_data = load_experiment_data(args, parsed.seed)
    print(f"Training/loading models on device={device}", flush=True)
    if parsed.reuse_checkpoints and checkpoints_exist(parsed, experiment_data.target_dim):
        multi_model, ova_models = load_model_checkpoints(parsed, experiment_data, device, experiment_args)
    else:
        multi_model, _, multi_training_info = train_multiclass_model(experiment_data, args, device, parsed.seed)
        ova_models, _, ova_training_info = train_ova_models(experiment_data, args, device, parsed.seed)
        if parsed.save_checkpoints or parsed.reuse_checkpoints:
            save_model_checkpoints(
                parsed,
                experiment_data,
                multi_model,
                ova_models,
                multi_training_info,
                ova_training_info,
                device,
                args,
            )

    if parsed.checkpoint_only:
        print("CHECKPOINT_ONLY=1, skipping LRP generation.", flush=True)
        return

    lrp_multi_model = copy.deepcopy(multi_model).to(device)
    lrp_ova_models = [copy.deepcopy(model).to(device) for model in ova_models]
    disable_inplace_relu(lrp_multi_model)
    for model in lrp_ova_models:
        disable_inplace_relu(model)

    X_test = experiment_data.X_test
    y_true = experiment_data.y_test.astype(int)
    image_paths = test_image_paths(parsed, len(y_true), experiment_data)
    mask_paths = [segmentation_mask_path(parsed, path) for path in image_paths]
    multi_probs = predict_multi(multi_model, X_test, device)
    ova_probs = predict_ova(ova_models, X_test, device)
    multi_pred = multi_probs.argmax(axis=1)
    ova_pred = ova_probs.argmax(axis=1)
    multi_confidence = multi_probs[np.arange(len(multi_pred)), multi_pred]
    ova_confidence = ova_probs[np.arange(len(ova_pred)), ova_pred]

    allowed_indices = None
    if parsed.require_mask:
        allowed_indices = {idx for idx, mask_path in enumerate(mask_paths) if mask_path is not None}

    if parsed.image_indices:
        selected = parsed.image_indices[: parsed.num_images]
    elif parsed.selection == "all":
        selected = sorted(allowed_indices) if allowed_indices is not None else list(range(len(y_true)))
    elif parsed.selection == "balanced":
        selected = balanced_indices(y_true, multi_pred, ova_pred, parsed.num_images, allowed_indices)
    elif parsed.selection == "contrast":
        selected = contrast_indices(y_true, multi_pred, ova_pred, multi_confidence, ova_confidence, parsed.num_images, allowed_indices)
    elif parsed.selection == "outcomes":
        selected = outcome_indices(y_true, multi_pred, ova_pred, parsed.num_images, allowed_indices)
    else:
        selected = ordered_indices(y_true, multi_pred, ova_pred, parsed.num_images)

    labels = list(getattr(experiment_data, "class_names", None) or CLASS_LABELS.get(parsed.dataset, [str(i) for i in range(experiment_data.target_dim)]))
    labels = labels[: experiment_data.target_dim]
    seed_dir = f"seed_{parsed.seed}_lrp"
    if parsed.run_tag:
        seed_dir = f"seed_{parsed.seed}_{parsed.run_tag}_lrp"
    if parsed.lrp_target == "true":
        seed_dir = f"{seed_dir}_true_target"
    output_dir = Path(parsed.output_dir) / parsed.dataset / seed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in selected:
        image = X_test[idx]
        target_class = int(y_true[idx])
        multi_pred_class = int(multi_pred[idx])
        ova_pred_class = int(ova_pred[idx])
        if parsed.lrp_target == "true":
            multi_target = target_class
            ova_target = target_class
        else:
            multi_target = multi_pred_class
            ova_target = ova_pred_class

        multi_lrp = lrp_heatmap(lrp_multi_model, image, multi_target, device, parsed.lrp_epsilon)
        ova_lrp = lrp_heatmap(lrp_ova_models[ova_target], image, 0, device, parsed.lrp_epsilon)
        original = image_to_pil(image)
        mask = load_mask(mask_paths[idx], original.width, original.height)
        multi_metrics = prefixed_metrics(
            "multi",
            cam_mask_metrics(
                multi_lrp,
                mask,
                top_percents=parsed.cam_top_percents,
                mask_area_factors=parsed.cam_mask_area_factors,
                peritumor_radii=parsed.peritumor_radii,
            ),
        )
        ova_metrics = prefixed_metrics(
            "ova",
            cam_mask_metrics(
                ova_lrp,
                mask,
                top_percents=parsed.cam_top_percents,
                mask_area_factors=parsed.cam_mask_area_factors,
                peritumor_radii=parsed.peritumor_radii,
            ),
        )

        filename = (
            f"idx_{idx:04d}_true_{labels[target_class]}_"
            f"multi_{labels[multi_pred_class]}_ova_{labels[ova_pred_class]}.png"
        )
        if parsed.no_save_images:
            filename = ""
        else:
            panel = make_panel(
                original,
                overlay_heatmap(image, multi_lrp),
                overlay_heatmap(image, ova_lrp),
                true_label=labels[target_class],
                multi_label=labels[multi_pred_class],
                multi_confidence=float(multi_probs[idx, multi_pred_class]),
                ova_label=labels[ova_pred_class],
                ova_confidence=float(ova_probs[idx, ova_pred_class]),
                mask=mask,
            )
            panel.save(output_dir / filename)

        rows.append(
            {
                "test_index": idx,
                "cam_method": "zennit_epsilon_gamma_box",
                "target_layer": "input",
                "true_label": labels[target_class],
                "multi_pred": labels[multi_pred_class],
                "ova_pred": labels[ova_pred_class],
                "cam_target_mode": parsed.lrp_target,
                "multi_gradcam_target": labels[multi_target],
                "ova_gradcam_target": labels[ova_target],
                "outcome": (
                    "both_correct"
                    if multi_pred_class == target_class and ova_pred_class == target_class
                    else "multi_correct_ova_wrong"
                    if multi_pred_class == target_class
                    else "multi_wrong_ova_correct"
                    if ova_pred_class == target_class
                    else "both_wrong"
                ),
                "multi_confidence": float(multi_confidence[idx]),
                "ova_confidence": float(ova_confidence[idx]),
                "image_path": str(image_paths[idx]) if image_paths[idx] is not None else "",
                "mask_path": str(mask_paths[idx]) if mask_paths[idx] is not None else "",
                "image_file": filename,
                "all_ova_image_file": "",
                **multi_metrics,
                **ova_metrics,
            }
        )

    index_path = output_dir / "gradcam_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    write_metrics_summary(rows, output_dir / "gradcam_metrics_summary.csv")
    print(f"Saved LRP index to {index_path}", flush=True)


if __name__ == "__main__":
    main()
