from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


OUTCOME_TITLES = {
    "both_correct": "Ambos modelos aciertan",
    "multi_correct_ova_wrong": "Acierta multi-salida",
    "multi_wrong_ova_correct": "Acierta One-vs-All",
    "both_wrong": "Ambos modelos fallan",
}


def parse_panel_name(path: Path) -> tuple[str, str, str]:
    match = re.search(
        r"true_(?P<true>.+?)_multi_(?P<multi>.+?)_ova_(?P<ova>.+)$",
        path.stem,
    )
    if not match:
        return "?", "?", "?"
    return match.group("true"), match.group("multi"), match.group("ova")


def infer_outcome(true_label: str, multi_label: str, ova_label: str) -> str:
    multi_ok = multi_label == true_label
    ova_ok = ova_label == true_label
    if multi_ok and ova_ok:
        return "both_correct"
    if multi_ok and not ova_ok:
        return "multi_correct_ova_wrong"
    if not multi_ok and ova_ok:
        return "multi_wrong_ova_correct"
    return "both_wrong"


def crop_panel_image(panel: Image.Image, with_mask: bool) -> list[np.ndarray]:
    """Extract visual columns from raw XAI panel, dropping its small text header."""
    arr = np.asarray(panel.convert("RGB"))
    height, width = arr.shape[:2]

    # Raw panels created by explicabilidad_* scripts have the image tiles below
    # a short textual header and per-tile labels. The x coordinates are stable:
    # 3 tiles without masks, 4 tiles with masks.
    y0 = 78 if height >= 180 else int(height * 0.38)
    ncols = 4 if with_mask else 3
    gap = 8
    tile_w = (width - gap * (ncols - 1)) // ncols

    crops: list[np.ndarray] = []
    for col in range(ncols):
        x0 = col * (tile_w + gap)
        crop = arr[y0:height, x0 : x0 + tile_w]
        crops.append(crop)
    return crops


def add_case_row(fig, grid, row_idx: int, panel_path: Path, with_mask: bool, method_label: str) -> None:
    panel = Image.open(panel_path)
    true_label, multi_label, ova_label = parse_panel_name(panel_path)
    outcome = infer_outcome(true_label, multi_label, ova_label)
    title = OUTCOME_TITLES[outcome]
    subtitle = f"Real: {true_label} | Multi: {multi_label} | OVA: {ova_label}"
    crops = crop_panel_image(panel, with_mask=with_mask)

    labels = ["Original", "Máscara", "Multi-salida", "One-vs-All"] if with_mask else [
        "Original",
        "Multi-salida",
        "One-vs-All",
    ]

    for col_idx, (crop, label) in enumerate(zip(crops, labels)):
        ax = fig.add_subplot(grid[row_idx * 2 + 1, col_idx])
        ax.imshow(crop, cmap="gray")
        ax.set_axis_off()
        if row_idx == 0:
            ax.set_title(label, fontsize=15, pad=8)

    ax_title = fig.add_subplot(grid[row_idx * 2, :])
    ax_title.set_axis_off()
    ax_title.text(
        0.5,
        0.64,
        f"{method_label}: {title}",
        ha="center",
        va="center",
        fontsize=16,
    )
    ax_title.text(
        0.5,
        0.18,
        subtitle,
        ha="center",
        va="center",
        fontsize=11,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose raw XAI panels into thesis-style figures."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Raw PNG panels.")
    parser.add_argument("--output", required=True, help="Output path without extension or with .png/.pdf.")
    parser.add_argument("--method-label", default="Grad-CAM++")
    parser.add_argument("--with-mask", action="store_true")
    parser.add_argument("--sort-outcomes", action="store_true")
    args = parser.parse_args()

    inputs = [Path(path) for path in args.inputs]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input panels: {missing}")
    if args.sort_outcomes:
        order = {name: idx for idx, name in enumerate(OUTCOME_TITLES)}
        inputs = sorted(
            inputs,
            key=lambda path: order[infer_outcome(*parse_panel_name(path))],
        )

    nrows = len(inputs)
    ncols = 4 if args.with_mask else 3
    fig_w = 12.6 if args.with_mask else 12.0
    fig_h = 2.65 * nrows + 0.25
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    grid = fig.add_gridspec(
        nrows=nrows * 2,
        ncols=ncols,
        height_ratios=sum(([0.27, 1.0] for _ in range(nrows)), []),
        hspace=0.12,
        wspace=0.04,
    )

    for row_idx, panel_path in enumerate(inputs):
        add_case_row(fig, grid, row_idx, panel_path, args.with_mask, args.method_label)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".png", ".pdf"}:
        base = output.with_suffix("")
    else:
        base = output
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
