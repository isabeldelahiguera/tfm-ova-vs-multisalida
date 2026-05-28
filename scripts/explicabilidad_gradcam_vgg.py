from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.model_selection import train_test_split
from torch import nn

from tfm.data import load_experiment_data
from tfm.experiment import train_multiclass_model, train_ova_models
from tfm.training import set_seed


CLASS_LABELS = {
    "brisc": ["glioma", "meningioma", "pituitary", "no_tumor"],
    "tb_chest_xray": ["normal", "tuberculosis"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM comparisons for VGG multi-output and OVA models."
    )
    parser.add_argument("--dataset", choices=["brisc", "tb_chest_xray"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--image-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--selection",
        choices=["all", "balanced", "ordered", "contrast", "outcomes"],
        default="balanced",
    )
    parser.add_argument(
        "--cam-target",
        choices=["predicted", "true"],
        default="predicted",
        help=(
            "Class used as Grad-CAM target. 'predicted' explains each model decision; "
            "'true' compares attention on the ground-truth class."
        ),
    )
    parser.add_argument(
        "--cam-method",
        choices=["gradcam", "gradcam++", "rise"],
        default="gradcam",
        help="Class activation mapping method used to generate heatmaps.",
    )
    parser.add_argument(
        "--target-layer",
        default="last",
        help=(
            "Layer used for CAM. Use 'last', a Conv2d index such as -1/3, "
            "or a module name such as features.12."
        ),
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Only write CSV metrics; skip Grad-CAM PNG panels.",
    )
    parser.add_argument(
        "--include-all-ova-cams",
        action="store_true",
        help="Also save a panel with Grad-CAM maps for every OVA binary model.",
    )
    parser.add_argument(
        "--include-all-class-metrics",
        action="store_true",
        help="Also compute Grad-CAM metrics for every class/logit in multi-output and OVA.",
    )
    parser.add_argument("--output-dir", default="resultados_actualizados/explicabilidad")
    parser.add_argument("--vgg-channels", type=int, nargs=3, default=[32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--rise-samples", type=int, default=200)
    parser.add_argument("--rise-mask-size", type=int, default=8)
    parser.add_argument("--rise-prob", type=float, default=0.5)
    parser.add_argument("--rise-batch-size", type=int, default=64)
    parser.add_argument("--brisc-root", default="/mnt/homeGPU/imhiguera/data/brisc2025")
    parser.add_argument("--brisc-segmentation-root", default="/mnt/homeGPU/imhiguera/data/brisc2025_segmentation")
    parser.add_argument("--tb-root", default="/mnt/homeGPU/imhiguera/data/tb_chest_xray")
    parser.add_argument("--require-mask", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    return parser


def experiment_args(parsed: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        task="classification",
        dataset=parsed.dataset,
        model_arch="vgg",
        hidden_layers=[32, 16],
        vgg_channels=parsed.vgg_channels,
        batch_normalization=False,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        early_stopping_patience=parsed.early_stopping_patience,
        early_stopping_min_delta=parsed.early_stopping_min_delta,
        learning_rate=parsed.learning_rate,
        seed=parsed.seed,
        seeds=[parsed.seed],
        coupling_modes=["multi-output", "ova"],
        synthetic_samples=600,
        synthetic_features=20,
        synthetic_classes=4,
        synthetic_targets=3,
        dependency_strength=0.3,
        max_train=parsed.max_train if parsed.max_train and parsed.max_train > 0 else None,
        max_test=parsed.max_test if parsed.max_test and parsed.max_test > 0 else None,
        brisc_root=parsed.brisc_root,
        tb_root=parsed.tb_root,
        image_size=parsed.image_size,
    )


def conv_layers_with_names(model: nn.Module) -> list[tuple[str, nn.Module]]:
    conv_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if not conv_layers:
        raise ValueError("Grad-CAM requires at least one Conv2d layer")
    return conv_layers


def target_conv_layer(model: nn.Module, layer_spec: str) -> tuple[str, nn.Module]:
    conv_layers = conv_layers_with_names(model)
    if layer_spec == "last":
        return conv_layers[-1]

    modules = dict(model.named_modules())
    if layer_spec in modules:
        module = modules[layer_spec]
        if not isinstance(module, nn.Conv2d):
            raise ValueError(f"Target layer {layer_spec!r} is not a Conv2d layer")
        return layer_spec, module

    try:
        layer_index = int(layer_spec)
    except ValueError as exc:
        available = ", ".join(name for name, _module in conv_layers)
        raise ValueError(
            f"Unknown target layer {layer_spec!r}. Available Conv2d layers: {available}"
        ) from exc

    try:
        return conv_layers[layer_index]
    except IndexError as exc:
        available = ", ".join(f"{idx}:{name}" for idx, (name, _module) in enumerate(conv_layers))
        raise ValueError(
            f"Conv2d layer index {layer_index} is out of range. Available: {available}"
        ) from exc


def disable_inplace_relu(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def gradcam(
    model: nn.Module,
    image: np.ndarray,
    target_score_index: int,
    device: torch.device,
    method: str = "gradcam",
    target_layer: str = "last",
) -> np.ndarray:
    model.eval()
    _layer_name, layer = target_conv_layer(model, target_layer)
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    try:
        tensor = torch.tensor(image[None, ...], dtype=torch.float32, device=device)
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        score = output[:, target_score_index].sum()
        score.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Could not collect activations/gradients for Grad-CAM")

    if method == "gradcam++":
        gradients_pow2 = gradients.pow(2)
        gradients_pow3 = gradients_pow2 * gradients
        denominator = 2.0 * gradients_pow2 + (
            activations * gradients_pow3
        ).sum(dim=(2, 3), keepdim=True)
        alphas = gradients_pow2 / (denominator + 1e-8)
        weights = (alphas * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)
    else:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1)).squeeze(0)
    cam = cam.cpu().numpy()
    cam = resize_array(cam, image.shape[-1], image.shape[-2])
    cam = cam - cam.min()
    max_value = cam.max()
    if max_value > 0:
        cam = cam / max_value
    return cam


def resize_array(array: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.uint8(np.clip(array, 0, None) / (array.max() or 1.0) * 255))
    image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def random_rise_masks(
    num_masks: int,
    mask_size: int,
    image_height: int,
    image_width: int,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    masks = rng.random((num_masks, mask_size, mask_size)) < probability
    resized_masks = []
    for mask in masks:
        mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L")
        mask_image = mask_image.resize((image_width, image_height), Image.BILINEAR)
        resized_masks.append(np.asarray(mask_image, dtype=np.float32) / 255.0)
    return np.stack(resized_masks, axis=0)


def rise_cam(
    model: nn.Module,
    image: np.ndarray,
    target_score_index: int,
    device: torch.device,
    *,
    output_kind: str,
    num_masks: int,
    mask_size: int,
    probability: float,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    model.eval()
    image_height, image_width = image.shape[-2], image.shape[-1]
    rng = np.random.default_rng(seed)
    masks = random_rise_masks(
        num_masks,
        mask_size,
        image_height,
        image_width,
        probability,
        rng,
    )
    saliency = np.zeros((image_height, image_width), dtype=np.float32)
    image_batch = image[None, ...]
    eps = 1e-8

    with torch.no_grad():
        for start in range(0, num_masks, batch_size):
            batch_masks = masks[start : start + batch_size]
            masked_images = image_batch * batch_masks[:, None, :, :]
            tensor = torch.tensor(masked_images, dtype=torch.float32, device=device)
            output = model(tensor)
            if output_kind == "multi":
                scores = torch.softmax(output, dim=1)[:, target_score_index]
            else:
                scores = torch.sigmoid(output.reshape(output.shape[0], -1)[:, target_score_index])
            scores_np = scores.detach().cpu().numpy().astype(np.float32)
            saliency += np.sum(batch_masks * scores_np[:, None, None], axis=0)

    saliency = saliency / (num_masks * probability + eps)
    saliency = saliency - saliency.min()
    max_value = saliency.max()
    if max_value > 0:
        saliency = saliency / max_value
    return saliency


def compute_cam(
    model: nn.Module,
    image: np.ndarray,
    target_score_index: int,
    device: torch.device,
    parsed: argparse.Namespace,
    *,
    output_kind: str,
    sample_seed: int,
) -> np.ndarray:
    if parsed.cam_method == "rise":
        return rise_cam(
            model,
            image,
            target_score_index,
            device,
            output_kind=output_kind,
            num_masks=parsed.rise_samples,
            mask_size=parsed.rise_mask_size,
            probability=parsed.rise_prob,
            batch_size=parsed.rise_batch_size,
            seed=sample_seed,
        )
    return gradcam(
        model,
        image,
        target_score_index,
        device,
        parsed.cam_method,
        parsed.target_layer,
    )


def image_to_pil(image: np.ndarray) -> Image.Image:
    if image.shape[0] == 1:
        array = image[0]
        gray = np.uint8(np.clip(array, 0, 1) * 255)
        return Image.fromarray(gray, mode="L").convert("RGB")

    array = np.transpose(image, (1, 2, 0))
    rgb = np.uint8(np.clip(array, 0, 1) * 255)
    return Image.fromarray(rgb, mode="RGB")


def overlay_heatmap(image: np.ndarray, cam: np.ndarray) -> Image.Image:
    base = image_to_pil(image).convert("RGBA")
    heat = np.zeros((cam.shape[0], cam.shape[1], 4), dtype=np.uint8)
    heat[..., 0] = 255
    heat[..., 1] = np.uint8(120 * cam)
    heat[..., 3] = np.uint8(170 * cam)
    overlay = Image.fromarray(heat, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def load_mask(mask_path: Path | None, width: int, height: int) -> np.ndarray | None:
    if mask_path is None:
        return None
    with Image.open(mask_path) as mask:
        mask = mask.convert("L").resize((width, height), Image.NEAREST)
        array = np.asarray(mask, dtype=np.float32)
        return array > 0


def dilate_bool_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    size = radius * 2 + 1
    mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L")
    dilated = mask_image.filter(ImageFilter.MaxFilter(size=size))
    return np.asarray(dilated) > 0


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    array = np.zeros((*mask.shape, 3), dtype=np.uint8)
    array[mask] = (0, 180, 80)
    return Image.fromarray(array, mode="RGB")


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: int = 120) -> Image.Image:
    base = image.convert("RGBA")
    mask_layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_layer[mask] = (0, 220, 90, alpha)
    overlay = Image.fromarray(mask_layer, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def cam_global_metrics(cam: np.ndarray) -> dict[str, float]:
    cam = np.asarray(cam, dtype=np.float32)
    eps = 1e-8
    flat = cam.reshape(-1)
    flat_sum = float(flat.sum())
    if flat_sum > eps:
        sorted_cam = np.sort(flat)
        n_values = float(sorted_cam.size)
        gini = float((2 * np.sum((np.arange(sorted_cam.size) + 1) * sorted_cam)) / (n_values * flat_sum))
        gini -= float((n_values + 1) / n_values)
    else:
        gini = 0.0

    return {
        "cam_active_area_frac_50": float((cam >= 0.50).mean()),
        "cam_active_area_frac_75": float((cam >= 0.75).mean()),
        "cam_gini": gini,
    }


def cam_mask_metrics(cam: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
    cam = np.asarray(cam, dtype=np.float32)
    metrics = cam_global_metrics(cam)
    if mask is None or not mask.any():
        return metrics

    mask = np.asarray(mask, dtype=bool)
    peritumor = np.logical_and(dilate_bool_mask(mask, radius=5), ~mask)
    outside_peritumor = ~(mask | peritumor)
    total_activation = float(cam.sum())
    inside_activation = float(cam[mask].sum())
    outside_activation = float(cam[~mask].sum())
    mask_area = float(mask.mean())
    inside_mean = float(cam[mask].mean())
    outside_mean = float(cam[~mask].mean()) if (~mask).any() else 0.0
    eps = 1e-8

    metrics.update(
        {
            "cam_inside_frac": inside_activation / (total_activation + eps),
            "cam_outside_frac": outside_activation / (total_activation + eps),
            "cam_inside_mean": inside_mean,
            "cam_outside_mean": outside_mean,
            "cam_inside_outside_ratio": inside_mean / (outside_mean + eps),
            "mask_area_frac": mask_area,
        }
    )
    zone_specs = [
        ("tumor", mask),
        ("peritumor_r5", peritumor),
        ("outside_peritumor_r5", outside_peritumor),
    ]
    for zone_name, zone_mask in zone_specs:
        if zone_mask.any():
            zone_activation = float(cam[zone_mask].sum())
            zone_mean = float(cam[zone_mask].mean())
            zone_area = float(zone_mask.mean())
        else:
            zone_activation = 0.0
            zone_mean = 0.0
            zone_area = 0.0
        metrics[f"cam_{zone_name}_activation_frac"] = zone_activation / (total_activation + eps)
        metrics[f"cam_{zone_name}_mean"] = zone_mean
        metrics[f"cam_{zone_name}_area_frac"] = zone_area

    for threshold_label, threshold in (("50", 0.50), ("75", 0.75)):
        active = cam >= threshold
        intersection = float(np.logical_and(active, mask).sum())
        union = float(np.logical_or(active, mask).sum())
        active_area = float(active.sum())
        mask_area_pixels = float(mask.sum())
        metrics[f"cam_thr{threshold_label}_iou"] = intersection / (union + eps)
        metrics[f"cam_thr{threshold_label}_dice"] = (2.0 * intersection) / (
            active_area + mask_area_pixels + eps
        )
        metrics[f"cam_thr{threshold_label}_precision"] = intersection / (active_area + eps)
        metrics[f"cam_thr{threshold_label}_recall"] = intersection / (mask_area_pixels + eps)
        metrics[f"cam_thr{threshold_label}_outside_precision"] = (
            1.0 - metrics[f"cam_thr{threshold_label}_precision"]
        )

    yy, xx = np.indices(cam.shape)
    cam_mass = total_activation + eps
    cam_cy = float((yy * cam).sum() / cam_mass)
    cam_cx = float((xx * cam).sum() / cam_mass)
    mask_y = float(yy[mask].mean())
    mask_x = float(xx[mask].mean())
    diagonal = float(np.hypot(cam.shape[0], cam.shape[1]))
    metrics["cam_mask_centroid_distance_norm"] = float(
        np.hypot(cam_cy - mask_y, cam_cx - mask_x) / (diagonal + eps)
    )
    max_y, max_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    metrics["cam_pointing_game_hit"] = float(mask[max_y, max_x])

    for percentile in (70, 80, 90):
        threshold = float(np.percentile(cam, percentile))
        hot = cam >= threshold
        intersection = float(np.logical_and(hot, mask).sum())
        union = float(np.logical_or(hot, mask).sum())
        hot_area = float(hot.sum())
        mask_area_pixels = float(mask.sum())
        metrics[f"cam_top{100 - percentile}_iou"] = intersection / (union + eps)
        metrics[f"cam_top{100 - percentile}_dice"] = (2.0 * intersection) / (
            hot_area + mask_area_pixels + eps
        )
        metrics[f"cam_top{100 - percentile}_precision"] = intersection / (hot_area + eps)
        metrics[f"cam_top{100 - percentile}_recall"] = intersection / (mask_area_pixels + eps)
        metrics[f"cam_top{100 - percentile}_outside_precision"] = 1.0 - metrics[
            f"cam_top{100 - percentile}_precision"
        ]

    mask_area_pixels = int(mask.sum())
    hot = np.zeros(cam.size, dtype=bool)
    if mask_area_pixels > 0:
        flat_order = np.argsort(cam.reshape(-1))[::-1]
        hot[flat_order[:mask_area_pixels]] = True
    hot = hot.reshape(cam.shape)
    intersection = float(np.logical_and(hot, mask).sum())
    union = float(np.logical_or(hot, mask).sum())
    hot_area = float(hot.sum())
    mask_area_pixels_float = float(mask.sum())
    metrics["cam_top_mask_area_iou"] = intersection / (union + eps)
    metrics["cam_top_mask_area_dice"] = (2.0 * intersection) / (
        hot_area + mask_area_pixels_float + eps
    )
    metrics["cam_top_mask_area_precision"] = intersection / (hot_area + eps)
    metrics["cam_top_mask_area_recall"] = intersection / (mask_area_pixels_float + eps)
    metrics["cam_top_mask_area_outside_precision"] = (
        1.0 - metrics["cam_top_mask_area_precision"]
    )

    return metrics


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float | str]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def layer_tag(layer_spec: str, resolved_layer_name: str) -> str:
    if layer_spec == "last":
        return "last"
    tag_source = resolved_layer_name or layer_spec
    return tag_source.replace(".", "_").replace("-", "neg")


def cam_output_dir_name(parsed: argparse.Namespace, base_seed_dir: str, layer_name: str) -> str:
    seed_dir = base_seed_dir
    if parsed.cam_method != "gradcam":
        seed_dir = f"{seed_dir}_{parsed.cam_method.replace('+', 'p')}"
    if parsed.cam_method in {"gradcam", "gradcam++"}:
        resolved_layer_tag = layer_tag(parsed.target_layer, layer_name)
        if resolved_layer_tag != "last":
            seed_dir = f"{seed_dir}_layer_{resolved_layer_tag}"
    return seed_dir


def write_metrics_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.startswith(("multi_cam_", "ova_cam_"))
            and isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    if not metric_names:
        return

    summary_rows: list[dict[str, object]] = []
    group_specs = [
        ("all", lambda _row: "all"),
        ("true_label", lambda row: str(row["true_label"])),
        ("outcome", lambda row: str(row["outcome"])),
    ]
    for group_name, key_fn in group_specs:
        groups: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            groups.setdefault(key_fn(row), []).append(row)
        for group_value, group_rows in groups.items():
            summary: dict[str, object] = {
                "group": group_name,
                "value": group_value,
                "n": len(group_rows),
            }
            for metric_name in metric_names:
                values = [
                    float(row[metric_name])
                    for row in group_rows
                    if row.get(metric_name, "") != ""
                ]
                summary[metric_name] = float(np.mean(values)) if values else ""
            summary_rows.append(summary)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_all_class_metrics_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.startswith("cam_") and isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    if not metric_names:
        return

    group_specs = [
        ("model_type", ["model_type"]),
        ("model_type_relation", ["model_type", "cam_class_relation"]),
        ("model_type_class", ["model_type", "cam_class"]),
        ("model_type_true_label_relation", ["model_type", "true_label", "cam_class_relation"]),
    ]
    summary_rows: list[dict[str, object]] = []
    for group_name, group_keys in group_specs:
        groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in rows:
            groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
        for group_values, group_rows in groups.items():
            summary: dict[str, object] = {"group": group_name, "n": len(group_rows)}
            for key, value in zip(group_keys, group_values):
                summary[key] = value
            for metric_name in metric_names:
                values = [
                    float(row[metric_name])
                    for row in group_rows
                    if row.get(metric_name, "") != ""
                ]
                summary[metric_name] = float(np.mean(values)) if values else ""
            summary_rows.append(summary)

    with output_path.open("w", newline="") as handle:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        ordered_prefix = [
            "group",
            "model_type",
            "cam_class_relation",
            "cam_class",
            "true_label",
            "n",
        ]
        fieldnames = ordered_prefix + [key for key in fieldnames if key not in ordered_prefix]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def add_title(image: Image.Image, title: str, height: int = 34) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), title, fill="black")
    return canvas


def add_header(image: Image.Image, header_lines: list[str], line_height: int = 18) -> Image.Image:
    header_height = max(1, len(header_lines)) * line_height + 8
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    for line_idx, line in enumerate(header_lines):
        draw.text((4, 5 + line_idx * line_height), line, fill="black")
    return canvas


def make_panel(
    original: Image.Image,
    multi: Image.Image,
    ova: Image.Image,
    true_label: str,
    multi_label: str,
    multi_confidence: float,
    ova_label: str,
    ova_confidence: float,
    mask: np.ndarray | None = None,
) -> Image.Image:
    panels = [
        add_title(original, "Original"),
    ]
    if mask is not None:
        panels.append(add_title(overlay_mask(original, mask), "Mask real"))
        multi = overlay_mask(multi, mask, alpha=95)
        ova = overlay_mask(ova, mask, alpha=95)
    panels.extend(
        [
            add_title(multi, f"Multi: {multi_label} ({multi_confidence:.3f})"),
            add_title(ova, f"OVA: {ova_label} ({ova_confidence:.3f})"),
        ]
    )
    gap = 8
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    header_lines = [f"Real: {true_label}"]
    if mask is not None:
        header_lines.append("Verde: mascara real | Rojo: activacion Grad-CAM")
    else:
        header_lines.append("Rojo: activacion Grad-CAM")
    return add_header(canvas, header_lines)


def make_all_ova_panel(
    original: Image.Image,
    multi: Image.Image,
    ova_overlays: list[Image.Image],
    true_label: str,
    multi_label: str,
    multi_confidence: float,
    ova_labels: list[str],
    ova_probs: np.ndarray,
    mask: np.ndarray | None = None,
) -> Image.Image:
    panels = [add_title(original, "Original")]
    if mask is not None:
        panels.append(add_title(overlay_mask(original, mask), "Mask real"))
        multi = overlay_mask(multi, mask, alpha=95)
        ova_overlays = [overlay_mask(overlay, mask, alpha=95) for overlay in ova_overlays]

    panels.append(add_title(multi, f"Multi: {multi_label} ({multi_confidence:.3f})"))
    for label, prob, overlay in zip(ova_labels, ova_probs, ova_overlays):
        panels.append(add_title(overlay, f"OVA {label}: {float(prob):.3f}"))

    gap = 8
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap

    header_lines = [f"Real: {true_label}"]
    if mask is not None:
        header_lines.append("Verde: mascara real | Rojo: activacion Grad-CAM")
    else:
        header_lines.append("Rojo: activacion Grad-CAM")
    return add_header(canvas, header_lines)


def predict_multi(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for image in images:
            tensor = torch.tensor(image[None, ...], dtype=torch.float32, device=device)
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            probabilities.append(probs)
    return np.asarray(probabilities)


def predict_ova(models: list[nn.Module], images: np.ndarray, device: torch.device) -> np.ndarray:
    probabilities = []
    with torch.no_grad():
        for image in images:
            class_probs = []
            tensor = torch.tensor(image[None, ...], dtype=torch.float32, device=device)
            for model in models:
                model.eval()
                prob = torch.sigmoid(model(tensor).squeeze()).item()
                class_probs.append(prob)
            probabilities.append(class_probs)
    return np.asarray(probabilities)


def ordered_indices(y_true: np.ndarray, multi_pred: np.ndarray, ova_pred: np.ndarray, limit: int) -> list[int]:
    groups = [
        np.where((multi_pred == y_true) & (ova_pred == y_true))[0],
        np.where((multi_pred != y_true) & (ova_pred == y_true))[0],
        np.where((multi_pred == y_true) & (ova_pred != y_true))[0],
        np.where((multi_pred != y_true) & (ova_pred != y_true))[0],
    ]
    selected: list[int] = []
    for group in groups:
        for idx in group:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= limit:
                return selected
    return selected


def indices_by_prediction_quality(
    y_true: np.ndarray,
    multi_pred: np.ndarray,
    ova_pred: np.ndarray,
    candidates: np.ndarray,
) -> list[int]:
    groups = [
        candidates[(multi_pred[candidates] == y_true[candidates]) & (ova_pred[candidates] == y_true[candidates])],
        candidates[(multi_pred[candidates] != y_true[candidates]) & (ova_pred[candidates] == y_true[candidates])],
        candidates[(multi_pred[candidates] == y_true[candidates]) & (ova_pred[candidates] != y_true[candidates])],
        candidates[(multi_pred[candidates] != y_true[candidates]) & (ova_pred[candidates] != y_true[candidates])],
    ]
    ordered: list[int] = []
    for group in groups:
        ordered.extend(int(idx) for idx in group)
    return ordered


def balanced_indices(
    y_true: np.ndarray,
    multi_pred: np.ndarray,
    ova_pred: np.ndarray,
    limit: int,
    allowed_indices: set[int] | None = None,
) -> list[int]:
    labels = sorted(int(label) for label in np.unique(y_true))
    per_label: dict[int, list[int]] = {}
    for label in labels:
        candidates = np.where(y_true == label)[0]
        if allowed_indices is not None:
            candidates = np.asarray(
                [idx for idx in candidates if int(idx) in allowed_indices],
                dtype=np.int64,
            )
        per_label[label] = indices_by_prediction_quality(y_true, multi_pred, ova_pred, candidates)

    selected: list[int] = []
    while len(selected) < limit:
        added = False
        for label in labels:
            if per_label[label]:
                selected.append(per_label[label].pop(0))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def indices_by_contrast(
    y_true: np.ndarray,
    multi_pred: np.ndarray,
    ova_pred: np.ndarray,
    multi_confidence: np.ndarray,
    ova_confidence: np.ndarray,
    candidates: np.ndarray,
) -> list[int]:
    groups = [
        candidates[multi_pred[candidates] != ova_pred[candidates]],
        candidates[(multi_pred[candidates] != y_true[candidates]) | (ova_pred[candidates] != y_true[candidates])],
    ]

    selected: list[int] = []
    for group in groups:
        for idx in group:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)

    remaining = [int(idx) for idx in candidates if int(idx) not in selected]
    remaining.sort(key=lambda idx: min(float(multi_confidence[idx]), float(ova_confidence[idx])))
    selected.extend(remaining)
    return selected


def contrast_indices(
    y_true: np.ndarray,
    multi_pred: np.ndarray,
    ova_pred: np.ndarray,
    multi_confidence: np.ndarray,
    ova_confidence: np.ndarray,
    limit: int,
    allowed_indices: set[int] | None = None,
) -> list[int]:
    labels = sorted(int(label) for label in np.unique(y_true))
    per_label: dict[int, list[int]] = {}
    for label in labels:
        candidates = np.where(y_true == label)[0]
        if allowed_indices is not None:
            candidates = np.asarray(
                [idx for idx in candidates if int(idx) in allowed_indices],
                dtype=np.int64,
            )
        per_label[label] = indices_by_contrast(
            y_true,
            multi_pred,
            ova_pred,
            multi_confidence,
            ova_confidence,
            candidates,
        )

    selected: list[int] = []
    while len(selected) < limit:
        added = False
        for label in labels:
            if per_label[label]:
                selected.append(per_label[label].pop(0))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def prediction_outcome(y_true: int, multi_pred: int, ova_pred: int) -> str:
    multi_correct = multi_pred == y_true
    ova_correct = ova_pred == y_true
    if multi_correct and ova_correct:
        return "both_correct"
    if multi_correct and not ova_correct:
        return "multi_correct_ova_wrong"
    if not multi_correct and ova_correct:
        return "multi_wrong_ova_correct"
    return "both_wrong"


def balanced_from_candidates(y_true: np.ndarray, candidates: np.ndarray) -> list[int]:
    labels = sorted(int(label) for label in np.unique(y_true[candidates])) if len(candidates) else []
    per_label = {label: [int(idx) for idx in candidates if int(y_true[idx]) == label] for label in labels}

    selected: list[int] = []
    while True:
        added = False
        for label in labels:
            if per_label[label]:
                selected.append(per_label[label].pop(0))
                added = True
        if not added:
            break
    return selected


def outcome_indices(
    y_true: np.ndarray,
    multi_pred: np.ndarray,
    ova_pred: np.ndarray,
    limit: int,
    allowed_indices: set[int] | None = None,
) -> list[int]:
    outcome_order = [
        "both_correct",
        "multi_correct_ova_wrong",
        "multi_wrong_ova_correct",
        "both_wrong",
    ]
    per_outcome: dict[str, list[int]] = {}
    for outcome in outcome_order:
        candidates = [
            idx
            for idx in range(len(y_true))
            if prediction_outcome(int(y_true[idx]), int(multi_pred[idx]), int(ova_pred[idx])) == outcome
        ]
        if allowed_indices is not None:
            candidates = [idx for idx in candidates if idx in allowed_indices]
        per_outcome[outcome] = balanced_from_candidates(y_true, np.asarray(candidates, dtype=np.int64))

    selected: list[int] = []
    while len(selected) < limit:
        added = False
        for outcome in outcome_order:
            if per_outcome[outcome]:
                selected.append(per_outcome[outcome].pop(0))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def brisc_test_paths(brisc_root: str | Path, max_test: int | None) -> list[Path]:
    root = Path(brisc_root).expanduser()
    class_names = CLASS_LABELS["brisc"]
    per_class_limit = None
    extra_samples = 0
    if max_test is not None:
        per_class_limit, extra_samples = divmod(max_test, len(class_names))

    paths: list[Path] = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = root / "test" / class_name
        image_paths = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if per_class_limit is not None:
            class_limit = per_class_limit + int(class_idx < extra_samples)
            image_paths = image_paths[:class_limit]
        paths.extend(image_paths)
    return paths


def tb_test_paths(tb_root: str | Path, seed: int) -> list[Path]:
    root = Path(tb_root).expanduser()
    class_names = ["Normal", "Tuberculosis"]
    paths: list[Path] = []
    labels: list[int] = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = root / class_name
        image_paths = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        paths.extend(image_paths)
        labels.extend([class_idx] * len(image_paths))

    _, test_paths, _, _ = train_test_split(
        paths,
        labels,
        test_size=0.15,
        random_state=seed,
        stratify=labels,
    )
    return list(test_paths)


def test_image_paths(parsed: argparse.Namespace, expected_len: int) -> list[Path | None]:
    if parsed.dataset == "brisc":
        paths = brisc_test_paths(parsed.brisc_root, parsed.max_test if parsed.max_test and parsed.max_test > 0 else None)
    elif parsed.dataset == "tb_chest_xray":
        paths = tb_test_paths(parsed.tb_root, parsed.seed)
    else:
        paths = []

    if len(paths) != expected_len:
        print(
            f"Warning: collected {len(paths)} image paths but test set has {expected_len} samples",
            flush=True,
        )
        return [None] * expected_len
    return paths


def brisc_mask_path(parsed: argparse.Namespace, image_path: Path | None) -> Path | None:
    if parsed.dataset != "brisc" or image_path is None:
        return None
    mask_path = Path(parsed.brisc_segmentation_root).expanduser() / "test" / "masks" / f"{image_path.stem}.png"
    return mask_path if mask_path.exists() else None


def main() -> None:
    parsed = build_parser().parse_args()
    args = experiment_args(parsed)
    set_seed(parsed.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_seed_dir = f"seed_{parsed.seed}"
    if parsed.cam_target == "true":
        base_seed_dir = f"{base_seed_dir}_true_target"

    print(f"Loading {parsed.dataset} with seed={parsed.seed}", flush=True)
    experiment_data = load_experiment_data(args, parsed.seed)
    print(
        f"Training models on device={device}, train={len(experiment_data.y_train)}, "
        f"val={len(experiment_data.y_val)}, test={len(experiment_data.y_test)}",
        flush=True,
    )
    multi_model, _, _ = train_multiclass_model(experiment_data, args, device, parsed.seed)
    ova_models, _, _ = train_ova_models(experiment_data, args, device, parsed.seed)
    gradcam_multi_model = copy.deepcopy(multi_model).to(device)
    gradcam_ova_models = [copy.deepcopy(model).to(device) for model in ova_models]
    disable_inplace_relu(gradcam_multi_model)
    for model in gradcam_ova_models:
        disable_inplace_relu(model)

    if parsed.cam_method == "rise":
        resolved_layer_name = ""
    else:
        resolved_layer_name, _resolved_layer = target_conv_layer(
            gradcam_multi_model,
            parsed.target_layer,
        )
    seed_dir = cam_output_dir_name(parsed, base_seed_dir, resolved_layer_name)
    output_dir = Path(parsed.output_dir) / parsed.dataset / seed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"CAM method={parsed.cam_method}, target_layer={parsed.target_layer} "
        f"({resolved_layer_name}), save_images={not parsed.no_save_images}",
        flush=True,
    )
    if parsed.cam_method == "rise":
        print(
            f"RISE samples={parsed.rise_samples}, mask_size={parsed.rise_mask_size}, "
            f"prob={parsed.rise_prob}, batch_size={parsed.rise_batch_size}",
            flush=True,
        )

    X_test = experiment_data.X_test
    y_true = experiment_data.y_test
    image_paths = test_image_paths(parsed, len(y_true))
    mask_paths = [brisc_mask_path(parsed, path) for path in image_paths]
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
        if allowed_indices is None:
            selected = list(range(len(y_true)))
        else:
            selected = sorted(allowed_indices)
    elif parsed.selection == "balanced":
        selected = balanced_indices(y_true, multi_pred, ova_pred, parsed.num_images, allowed_indices)
    elif parsed.selection == "contrast":
        selected = contrast_indices(
            y_true,
            multi_pred,
            ova_pred,
            multi_confidence,
            ova_confidence,
            parsed.num_images,
            allowed_indices,
        )
    elif parsed.selection == "outcomes":
        selected = outcome_indices(y_true, multi_pred, ova_pred, parsed.num_images, allowed_indices)
    else:
        selected = ordered_indices(y_true, multi_pred, ova_pred, parsed.num_images)

    labels = CLASS_LABELS.get(parsed.dataset, [str(i) for i in range(experiment_data.target_dim)])
    labels = labels[: experiment_data.target_dim]
    rows = []
    all_class_rows: list[dict[str, object]] = []
    for idx in selected:
        image = X_test[idx]
        target_class = int(y_true[idx])
        multi_pred_class = int(multi_pred[idx])
        ova_pred_class = int(ova_pred[idx])
        if parsed.cam_target == "true":
            multi_target = target_class
            ova_target = target_class
        else:
            multi_target = multi_pred_class
            ova_target = ova_pred_class

        all_multi_cams = None
        if parsed.include_all_class_metrics:
            all_multi_cams = [
                compute_cam(
                    gradcam_multi_model,
                    image,
                    class_idx,
                    device,
                    parsed,
                    output_kind="multi",
                    sample_seed=parsed.seed * 1_000_000 + idx * 100 + class_idx,
                )
                for class_idx in range(experiment_data.target_dim)
            ]
            multi_cam = all_multi_cams[multi_target]
        else:
            multi_cam = compute_cam(
                gradcam_multi_model,
                image,
                multi_target,
                device,
                parsed,
                output_kind="multi",
                sample_seed=parsed.seed * 1_000_000 + idx * 100 + multi_target,
            )

        if parsed.include_all_ova_cams or parsed.include_all_class_metrics:
            all_ova_cams = [
                compute_cam(
                    model,
                    image,
                    0,
                    device,
                    parsed,
                    output_kind="ova",
                    sample_seed=parsed.seed * 1_000_000 + idx * 100 + class_idx,
                )
                for class_idx, model in enumerate(gradcam_ova_models)
            ]
            ova_cam = all_ova_cams[ova_target]
        else:
            all_ova_cams = None
            ova_cam = compute_cam(
                gradcam_ova_models[ova_target],
                image,
                0,
                device,
                parsed,
                output_kind="ova",
                sample_seed=parsed.seed * 1_000_000 + idx * 100 + ova_target,
            )
        original = image_to_pil(image)
        mask = load_mask(mask_paths[idx], original.width, original.height)
        multi_metrics = prefixed_metrics("multi", cam_mask_metrics(multi_cam, mask))
        ova_metrics = prefixed_metrics("ova", cam_mask_metrics(ova_cam, mask))

        filename = (
            f"idx_{idx:04d}_true_{labels[target_class]}_"
            f"multi_{labels[multi_pred_class]}_ova_{labels[ova_pred_class]}.png"
        )
        if parsed.no_save_images:
            filename = ""
        else:
            multi_overlay = overlay_heatmap(image, multi_cam)
            ova_overlay = overlay_heatmap(image, ova_cam)
            panel = make_panel(
                original,
                multi_overlay,
                ova_overlay,
                true_label=labels[target_class],
                multi_label=labels[multi_pred_class],
                multi_confidence=float(multi_probs[idx, multi_pred_class]),
                ova_label=labels[ova_pred_class],
                ova_confidence=float(ova_probs[idx, ova_pred_class]),
                mask=mask,
            )
            panel.save(output_dir / filename)
        all_ova_filename = ""
        if all_ova_cams is not None and not parsed.no_save_images:
            all_ova_filename = (
                f"idx_{idx:04d}_true_{labels[target_class]}_"
                f"multi_{labels[multi_pred_class]}_ova_all.png"
            )
            multi_overlay = overlay_heatmap(image, multi_cam)
            all_ova_panel = make_all_ova_panel(
                original,
                multi_overlay,
                [overlay_heatmap(image, cam) for cam in all_ova_cams],
                true_label=labels[target_class],
                multi_label=labels[multi_pred_class],
                multi_confidence=float(multi_probs[idx, multi_pred_class]),
                ova_labels=labels,
                ova_probs=ova_probs[idx],
                mask=mask,
            )
            all_ova_panel.save(output_dir / all_ova_filename)

        if parsed.include_all_class_metrics:
            assert all_multi_cams is not None
            assert all_ova_cams is not None
            for model_type, cams, pred_class in (
                ("multi", all_multi_cams, multi_pred_class),
                ("ova", all_ova_cams, ova_pred_class),
            ):
                for class_idx, class_cam in enumerate(cams):
                    class_label = labels[class_idx]
                    if class_idx == target_class:
                        relation = "true_class"
                    elif class_idx == pred_class:
                        relation = "predicted_non_true_class"
                    else:
                        relation = "other_non_true_class"
                    class_metrics = cam_mask_metrics(class_cam, mask)
                    all_class_rows.append(
                        {
                            "test_index": idx,
                            "cam_method": parsed.cam_method,
                            "target_layer": resolved_layer_name,
                            "true_label": labels[target_class],
                            "multi_pred": labels[multi_pred_class],
                            "ova_pred": labels[ova_pred_class],
                            "model_type": model_type,
                            "cam_class": class_label,
                            "cam_class_index": class_idx,
                            "cam_class_relation": relation,
                            "is_true_class": int(class_idx == target_class),
                            "is_model_pred_class": int(class_idx == pred_class),
                            **class_metrics,
                        }
                    )
        rows.append(
            {
                "test_index": idx,
                "cam_method": parsed.cam_method,
                "target_layer": resolved_layer_name,
                "true_label": labels[target_class],
                "multi_pred": labels[multi_pred_class],
                "ova_pred": labels[ova_pred_class],
                "cam_target_mode": parsed.cam_target,
                "multi_gradcam_target": labels[multi_target],
                "ova_gradcam_target": labels[ova_target],
                "outcome": prediction_outcome(target_class, multi_pred_class, ova_pred_class),
                "multi_confidence": float(multi_probs[idx, multi_pred_class]),
                "ova_confidence": float(ova_probs[idx, ova_pred_class]),
                "image_path": str(image_paths[idx]) if image_paths[idx] is not None else "",
                "mask_path": str(mask_paths[idx]) if mask_paths[idx] is not None else "",
                "image_file": filename,
                "all_ova_image_file": all_ova_filename,
                **multi_metrics,
                **ova_metrics,
            }
        )

    with (output_dir / "gradcam_index.csv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        ordered_prefix = [
            "test_index",
            "cam_method",
            "target_layer",
            "true_label",
            "multi_pred",
            "ova_pred",
            "cam_target_mode",
            "multi_gradcam_target",
            "ova_gradcam_target",
            "outcome",
            "multi_confidence",
            "ova_confidence",
            "image_path",
            "mask_path",
            "image_file",
            "all_ova_image_file",
        ]
        fieldnames = ordered_prefix + [key for key in fieldnames if key not in ordered_prefix]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_metrics_summary(rows, output_dir / "gradcam_metrics_summary.csv")

    if all_class_rows:
        all_class_path = output_dir / "gradcam_all_class_metrics.csv"
        with all_class_path.open("w", newline="") as handle:
            fieldnames = sorted({key for row in all_class_rows for key in row.keys()})
            ordered_prefix = [
                "test_index",
                "cam_method",
                "target_layer",
                "true_label",
                "multi_pred",
                "ova_pred",
                "model_type",
                "cam_class",
                "cam_class_index",
                "cam_class_relation",
                "is_true_class",
                "is_model_pred_class",
            ]
            fieldnames = ordered_prefix + [key for key in fieldnames if key not in ordered_prefix]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_class_rows)
        write_all_class_metrics_summary(
            all_class_rows,
            output_dir / "gradcam_all_class_metrics_summary.csv",
        )

    if parsed.no_save_images:
        print(f"Saved CSV metrics for {len(rows)} CAM samples to {output_dir}", flush=True)
    else:
        print(f"Saved {len(rows)} Grad-CAM panels to {output_dir}", flush=True)
    print(f"Index CSV: {output_dir / 'gradcam_index.csv'}", flush=True)
    print(f"Metrics summary CSV: {output_dir / 'gradcam_metrics_summary.csv'}", flush=True)
    if all_class_rows:
        print(f"All-class metrics CSV: {output_dir / 'gradcam_all_class_metrics.csv'}", flush=True)
        print(
            "All-class metrics summary CSV: "
            f"{output_dir / 'gradcam_all_class_metrics_summary.csv'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
