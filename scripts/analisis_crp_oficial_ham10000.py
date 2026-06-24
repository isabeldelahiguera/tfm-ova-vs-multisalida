from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from PIL import Image, ImageDraw
from zennit.composites import EpsilonGammaBox

from explicabilidad_gradcam_vgg import (
    CLASS_LABELS,
    cam_mask_metrics,
    disable_inplace_relu,
    experiment_args,
    image_to_pil,
    load_mask,
    overlay_heatmap,
    overlay_mask,
    predict_multi,
    predict_ova,
    prefixed_metrics,
    segmentation_mask_path,
    target_conv_layer,
    test_image_paths,
    write_metrics_summary,
)
from tfm.data import load_experiment_data
from tfm.experiment import train_multiclass_model, train_ova_models
from tfm.training import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CRP oficial con zennit-crp para analizar canales/conceptos en HAM10000."
    )
    parser.add_argument("--dataset", choices=["ham10000"], default="ham10000")
    parser.add_argument("--model-arch", choices=["vgg", "vgg16-pretrained"], default="vgg16-pretrained")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-tag", default="vgg16_block5_balanced")
    parser.add_argument("--reference-index-csv", default="")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=[
            "both_correct_ova_low_inside",
            "multi_correct_ova_wrong",
            "ova_correct_multi_wrong",
            "ova_low_inside",
            "minority_ova_low_inside",
        ],
    )
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--top-channels", type=int, default=5)
    parser.add_argument("--target-layer", default="last")
    parser.add_argument("--output-dir", default="resultados_actualizados/explicabilidad")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--vgg-channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--data-augmentation", choices=["none", "ham10000-basic"], default="none")
    parser.add_argument("--train-sampler", choices=["none", "balanced"], default="none")
    parser.add_argument("--pretrained-finetune", choices=["frozen", "block5", "full"], default="block5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cam-top-percents", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--cam-mask-area-factors", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--ham10000-root", default="/mnt/homeGPU/imhiguera/data/ham10000")
    parser.add_argument("--brisc-root", default="/mnt/homeGPU/imhiguera/data/brisc2025")
    parser.add_argument("--tb-root", default="/mnt/homeGPU/imhiguera/data/tb_chest_xray")
    parser.add_argument("--ham10000-test", choices=["internal", "official"], default="internal")
    parser.add_argument("--ham10000-split-csv", default=None)
    parser.add_argument("--ham10000-split-seed", type=int, default=2000)
    parser.add_argument(
        "--ham10000-mask-root",
        default="/mnt/homeGPU/imhiguera/data/ham10000/masks/HAM10000_segmentations_lesion_tschandl",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    return parser


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap - float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def resize_heatmap(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    heatmap = normalize_heatmap(heatmap)
    image = Image.fromarray(np.uint8(heatmap * 255))
    image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def select_group_rows(reference: pd.DataFrame, group: str, per_group: int) -> pd.DataFrame:
    df = reference.copy()
    if group == "both_correct_ova_low_inside":
        df = df[df["outcome"] == "both_correct"].sort_values("ova_cam_inside_frac", ascending=True)
    elif group == "multi_correct_ova_wrong":
        df = df[df["outcome"] == "multi_correct_ova_wrong"]
    elif group == "ova_correct_multi_wrong":
        df = df[df["outcome"] == "multi_wrong_ova_correct"]
    elif group == "ova_low_inside":
        df = df.sort_values("ova_cam_inside_frac", ascending=True)
    elif group == "minority_ova_low_inside":
        minority = {"akiec", "bcc", "df", "vasc"}
        df = df[df["true_label"].isin(minority)].sort_values("ova_cam_inside_frac", ascending=True)
    else:
        raise ValueError(f"Unknown group: {group}")
    return df.head(per_group)


def make_reference_table(y_true, multi_pred, ova_pred, labels):
    return pd.DataFrame(
        {
            "test_index": np.arange(len(y_true)),
            "true_label": [labels[i] for i in y_true],
            "multi_pred": [labels[i] for i in multi_pred],
            "ova_pred": [labels[i] for i in ova_pred],
            "outcome": [
                "both_correct"
                if multi_pred[i] == y_true[i] and ova_pred[i] == y_true[i]
                else "multi_correct_ova_wrong"
                if multi_pred[i] == y_true[i]
                else "multi_wrong_ova_correct"
                if ova_pred[i] == y_true[i]
                else "both_wrong"
                for i in range(len(y_true))
            ],
            "ova_cam_inside_frac": np.nan,
        }
    )


def top_channels(
    attribution: CondAttribution,
    tensor: torch.Tensor,
    output_idx: int,
    layer_name: str,
    composite: EpsilonGammaBox,
    concept: ChannelConcept,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    attr = attribution(
        tensor,
        conditions=[{"y": [output_idx]}],
        composite=composite,
        record_layer=[layer_name],
    )
    channel_relevance = concept.attribute(attr.relevances[layer_name], abs_norm=True)
    scores = channel_relevance.detach().cpu().numpy()[0]
    order = np.argsort(scores)[::-1][:top_k]
    return order.astype(int), scores


def conditional_channel_heatmap(
    attribution: CondAttribution,
    tensor: torch.Tensor,
    output_idx: int,
    layer_name: str,
    channel: int,
    composite: EpsilonGammaBox,
    width: int,
    height: int,
) -> np.ndarray:
    attr = attribution(
        tensor,
        conditions=[{"y": [output_idx], layer_name: [int(channel)]}],
        composite=composite,
        record_layer=[],
        mask_map=ChannelConcept.mask,
    )
    heatmap = attr.heatmap.detach().cpu().numpy()[0]
    return resize_heatmap(heatmap, width, height)


def add_label(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), text, fill="black")
    return canvas


def save_case_panel(
    output_path: Path,
    original: Image.Image,
    mask: np.ndarray | None,
    multi_heatmaps: list[tuple[int, np.ndarray]],
    ova_heatmaps: list[tuple[int, np.ndarray]],
    title: str,
):
    image_arr = np.asarray(original, dtype=np.float32).transpose(2, 0, 1) / 255.0
    panels = [add_label(original, "Original")]
    if mask is not None:
        panels.append(add_label(overlay_mask(original, mask), "Mask"))
    for channel, heatmap in multi_heatmaps:
        panel = overlay_heatmap(image_arr, heatmap)
        if mask is not None:
            panel = overlay_mask(panel, mask, alpha=95)
        panels.append(add_label(panel, f"Multi ch {channel}"))
    for channel, heatmap in ova_heatmaps:
        panel = overlay_heatmap(image_arr, heatmap)
        if mask is not None:
            panel = overlay_mask(panel, mask, alpha=95)
        panels.append(add_label(panel, f"OVA ch {channel}"))

    gap = 8
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels) + 32
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), title, fill="black")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 32))
        x += panel.width + gap
    canvas.save(output_path)


def main() -> None:
    parsed = build_parser().parse_args()
    set_seed(parsed.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args: SimpleNamespace = experiment_args(parsed)
    labels = CLASS_LABELS[parsed.dataset]

    print(f"Loading {parsed.dataset} seed={parsed.seed}", flush=True)
    experiment_data = load_experiment_data(args, parsed.seed)
    print(f"Training models on device={device}", flush=True)
    multi_model, _, _ = train_multiclass_model(experiment_data, args, device, parsed.seed)
    ova_models, _, _ = train_ova_models(experiment_data, args, device, parsed.seed)

    crp_multi = copy.deepcopy(multi_model).to(device).eval()
    crp_ova = [copy.deepcopy(model).to(device).eval() for model in ova_models]
    disable_inplace_relu(crp_multi)
    for model in crp_ova:
        disable_inplace_relu(model)

    multi_layer, _ = target_conv_layer(crp_multi, parsed.target_layer)
    ova_layer, _ = target_conv_layer(crp_ova[0], parsed.target_layer)
    print(f"CRP target layers: multi={multi_layer}, ova={ova_layer}", flush=True)

    X_test = experiment_data.X_test
    y_true = experiment_data.y_test.astype(int)
    image_paths = test_image_paths(parsed, len(y_true), experiment_data)
    mask_paths = [segmentation_mask_path(parsed, path) for path in image_paths]
    multi_probs = predict_multi(multi_model, X_test, device)
    ova_probs = predict_ova(ova_models, X_test, device)
    multi_pred = multi_probs.argmax(axis=1)
    ova_pred = ova_probs.argmax(axis=1)

    reference_path = Path(parsed.reference_index_csv).expanduser() if parsed.reference_index_csv else None
    reference = (
        pd.read_csv(reference_path)
        if reference_path is not None and reference_path.exists()
        else make_reference_table(y_true, multi_pred, ova_pred, labels)
    )

    selected = []
    seen = set()
    for group in parsed.groups:
        for row in select_group_rows(reference, group, parsed.per_group).to_dict("records"):
            key = (group, int(row["test_index"]))
            if key in seen:
                continue
            row["crp_group"] = group
            selected.append(row)
            seen.add(key)
    print(f"Selected {len(selected)} CRP cases", flush=True)

    output_dir = Path(parsed.output_dir) / parsed.dataset / f"seed_{parsed.seed}_{parsed.run_tag}_crp_official"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = output_dir / "panels"
    if parsed.save_images:
        panel_dir.mkdir(exist_ok=True)

    composite = EpsilonGammaBox(low=0.0, high=1.0)
    concept = ChannelConcept()
    multi_attr = CondAttribution(crp_multi, device=device)
    ova_attrs = [CondAttribution(model, device=device) for model in crp_ova]

    rows = []
    for item in selected:
        idx = int(item["test_index"])
        image = X_test[idx]
        original = image_to_pil(image)
        mask = load_mask(mask_paths[idx], original.width, original.height)
        target_class = int(y_true[idx])
        multi_target = int(multi_pred[idx])
        ova_target = int(ova_pred[idx])
        tensor = torch.tensor(image[None, ...], dtype=torch.float32, device=device, requires_grad=True)

        multi_channels, multi_scores = top_channels(
            multi_attr, tensor, multi_target, multi_layer, composite, concept, parsed.top_channels
        )
        ova_channels, ova_scores = top_channels(
            ova_attrs[ova_target], tensor, 0, ova_layer, composite, concept, parsed.top_channels
        )

        panel_multi = []
        panel_ova = []
        for model_type, pred_class, layer_name, channels, scores, attribution, output_idx in [
            ("multi", multi_target, multi_layer, multi_channels, multi_scores, multi_attr, multi_target),
            ("ova", ova_target, ova_layer, ova_channels, ova_scores, ova_attrs[ova_target], 0),
        ]:
            for rank, channel in enumerate(channels, start=1):
                heatmap = conditional_channel_heatmap(
                    attribution,
                    tensor,
                    output_idx,
                    layer_name,
                    int(channel),
                    composite,
                    original.width,
                    original.height,
                )
                if model_type == "multi" and len(panel_multi) < 2:
                    panel_multi.append((int(channel), heatmap))
                if model_type == "ova" and len(panel_ova) < 2:
                    panel_ova.append((int(channel), heatmap))
                metrics = prefixed_metrics(
                    "channel",
                    cam_mask_metrics(
                        heatmap,
                        mask,
                        top_percents=parsed.cam_top_percents,
                        mask_area_factors=parsed.cam_mask_area_factors,
                    ),
                )
                rows.append(
                    {
                        "crp_group": item["crp_group"],
                        "test_index": idx,
                        "true_label": labels[target_class],
                        "multi_pred": labels[multi_target],
                        "ova_pred": labels[ova_target],
                        "model_type": model_type,
                        "target_label": labels[pred_class],
                        "target_layer": layer_name,
                        "channel_rank": rank,
                        "channel": int(channel),
                        "channel_relevance": float(scores[channel]),
                        "image_path": str(image_paths[idx]) if image_paths[idx] is not None else "",
                        "mask_path": str(mask_paths[idx]) if mask_paths[idx] is not None else "",
                        **metrics,
                    }
                )

        if parsed.save_images:
            title = (
                f"{item['crp_group']} | idx={idx} | true={labels[target_class]} | "
                f"multi={labels[multi_target]} | ova={labels[ova_target]}"
            )
            save_case_panel(
                panel_dir / f"{item['crp_group']}_idx_{idx:04d}_{labels[target_class]}_multi_{labels[multi_target]}_ova_{labels[ova_target]}.png",
                original,
                mask,
                panel_multi,
                panel_ova,
                title,
            )

    index_path = output_dir / "crp_channel_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    summary = pd.DataFrame(rows).groupby(["crp_group", "model_type"], as_index=False).mean(numeric_only=True)
    summary.to_csv(output_dir / "crp_channel_summary.csv", index=False)
    write_metrics_summary(rows, output_dir / "crp_channel_metrics_summary.csv")
    print(f"Saved CRP channel index to {index_path}", flush=True)
    print(f"Saved CRP channel summary to {output_dir / 'crp_channel_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
