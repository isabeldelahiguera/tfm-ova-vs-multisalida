from __future__ import annotations

"""Intervenciones contrafactuales para auditar shortcuts en BRISC.

El protocolo es:
1. Entrenar o cargar checkpoints de modelos BRISC entrenados con imagenes originales.
2. Predecir el test original.
3. Crear copias intervenidas de imagenes de test, moviendo una senal visual de una
   clase origen hacia el valor tipico de una clase destino.
4. Volver a predecir con el mismo modelo fijo.

No se modifican imagenes de train, no se cambian etiquetas y no se reentrena con
imagenes intervenidas. Las intervenciones son una auditoria de sensibilidad del
modelo, no una tecnica de aumento de datos.
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from skimage.morphology import binary_dilation, disk

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from explicabilidad_gradcam_vgg import (  # noqa: E402
    CLASS_LABELS,
    brisc_mask_path,
    checkpoints_exist,
    experiment_args,
    load_mask,
    load_model_checkpoints,
    predict_multi,
    predict_ova,
    save_model_checkpoints,
    test_image_paths,
)
from tfm.data import load_experiment_data  # noqa: E402
from tfm.experiment import train_multiclass_model, train_ova_models  # noqa: E402
from tfm.training import set_seed  # noqa: E402


CLASS_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS["brisc"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual shortcut interventions for BRISC. Each intervention moves "
            "one suspicious visual descriptor of a source class towards a target class "
            "and measures whether multi-output/OVA predictions follow that change."
        )
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--checkpoint-dir", default="resultados_actualizados/checkpoints")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Train/load models and write checkpoints, then skip interventions.",
    )
    parser.add_argument(
        "--output-dir",
        default="resultados_actualizados/analisis_shortcuts/brisc/intervenciones",
        help=(
            "Directory where shortcut_intervention_index.csv (per image) and "
            "shortcut_intervention_summary.csv (compact metrics) are written."
        ),
    )
    parser.add_argument(
        "--shortcut-stats-csv",
        default=(
            "resultados_actualizados/analisis_dataset/brisc_test/"
            "resumen_shortcuts_test/comparacion_top_train_vs_test_clase.csv"
        ),
    )
    parser.add_argument("--vgg-channels", type=int, nargs=3, default=[32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--brisc-root", default="/mnt/homeGPU/imhiguera/data/brisc2025")
    parser.add_argument(
        "--brisc-segmentation-root",
        default="/mnt/homeGPU/imhiguera/data/brisc2025_segmentation",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--max-cases-per-source",
        type=int,
        default=None,
        help="Optional cap per source class for fast exploratory runs.",
    )
    parser.add_argument(
        "--intervention-strength",
        type=float,
        default=1.0,
        help="Blend factor for intensity/contrast interventions. 1.0 matches the target class mean.",
    )
    parser.add_argument(
        "--position-strength",
        type=float,
        default=0.75,
        help="Fraction of the centroid-to-center vector used in geometric interventions.",
    )
    parser.add_argument("--save-examples", type=int, default=0)
    return parser


def ensure_checkpoint_attrs(parsed: argparse.Namespace) -> None:
    """Populate the fields expected by shared checkpoint helpers."""
    parsed.dataset = "brisc"
    parsed.model_arch = "vgg"
    parsed.class_weighting = "none"
    parsed.data_augmentation = "none"
    parsed.train_sampler = "none"
    parsed.pretrained_finetune = "frozen"
    parsed.tb_root = "/mnt/homeGPU/imhiguera/data/tb_chest_xray"
    parsed.ham10000_root = "/mnt/homeGPU/imhiguera/data/ham10000"
    parsed.ham10000_test = "internal"
    parsed.ham10000_split_csv = None
    parsed.ham10000_split_seed = 2000
    parsed.ham10000_exclude_classes = []
    parsed.ham10000_label_mode = "original"


def to_explicabilidad_args(parsed: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="brisc",
        seed=parsed.seed,
        run_tag=parsed.run_tag,
        checkpoint_dir=parsed.checkpoint_dir,
        save_checkpoints=parsed.save_checkpoints,
        reuse_checkpoints=parsed.reuse_checkpoints,
        checkpoint_only=parsed.checkpoint_only,
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
        model_arch="vgg",
        class_weighting="none",
        data_augmentation="none",
        train_sampler="none",
        pretrained_finetune="frozen",
        ham10000_root="/mnt/homeGPU/imhiguera/data/ham10000",
        ham10000_test="internal",
        ham10000_split_csv=None,
        ham10000_split_seed=2000,
        ham10000_exclude_classes=[],
        ham10000_label_mode="original",
    )


def load_train_means(path: Path) -> dict[tuple[str, str], float]:
    df = pd.read_csv(path)
    means: dict[tuple[str, str], float] = {}
    for row in df.itertuples(index=False):
        means[(str(row.true_label), str(row.feature))] = float(row.group_mean_train)
    return means


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    return binary_dilation(mask, footprint=disk(radius))


def region_mean(image: np.ndarray, region: np.ndarray) -> float:
    if not region.any():
        return float(np.nan)
    return float(image[:, region].mean())


def region_p50(image: np.ndarray, region: np.ndarray) -> float:
    if not region.any():
        return float(np.nan)
    return float(np.percentile(image[:, region], 50))


def add_to_region(image: np.ndarray, region: np.ndarray, delta: float) -> np.ndarray:
    """Add a constant only inside a boolean region and keep pixels in [0, 1]."""
    edited = image.copy()
    if region.any() and np.isfinite(delta):
        edited[:, region] = np.clip(edited[:, region] + delta, 0.0, 1.0)
    return edited


def match_region_mean(
    image: np.ndarray,
    region: np.ndarray,
    target_mean: float,
    strength: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Shift a region mean, for example tumor intensity, towards a target mean."""
    before = region_mean(image, region)
    delta = (target_mean - before) * strength
    edited = add_to_region(image, region, delta)
    return edited, {
        "region_mean_before": before,
        "region_mean_after": region_mean(edited, region),
        "target_region_mean": float(target_mean),
        "region_delta_applied": float(delta),
    }


def match_tumor_outside_contrast(
    image: np.ndarray,
    tumor: np.ndarray,
    outside: np.ndarray,
    target_diff: float,
    strength: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Shift tumor intensity so tumor-context contrast approaches target_diff."""
    tumor_mean = region_mean(image, tumor)
    outside_mean = region_mean(image, outside)
    current_diff = tumor_mean - outside_mean
    delta = (target_diff - current_diff) * strength
    edited = add_to_region(image, tumor, delta)
    new_diff = region_mean(edited, tumor) - region_mean(edited, outside)
    return edited, {
        "contrast_before": float(current_diff),
        "contrast_after": float(new_diff),
        "target_contrast": float(target_diff),
        "contrast_delta_applied_to_tumor": float(delta),
    }


def match_outside_p50(
    image: np.ndarray,
    outside: np.ndarray,
    target_p50: float,
    strength: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Shift contextual pixels so their median intensity approaches target_p50."""
    before = region_p50(image, outside)
    delta = (target_p50 - before) * strength
    edited = add_to_region(image, outside, delta)
    return edited, {
        "outside_p50_before": before,
        "outside_p50_after": region_p50(edited, outside),
        "target_outside_p50": float(target_p50),
        "outside_delta_applied": float(delta),
    }


def translate_image(image: np.ndarray, dy: int, dx: int, fill_value: float = 0.0) -> np.ndarray:
    edited = np.full_like(image, fill_value)
    height, width = image.shape[-2:]

    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)

    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)

    if src_y0 < src_y1 and src_x0 < src_x1:
        edited[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[:, src_y0:src_y1, src_x0:src_x1]
    return edited


def move_tumor_position(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    direction: str,
    strength: float,
    fill_mode: str = "black",
    max_shift: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Translate the whole image to change the apparent tumor centrality."""
    yy, xx = np.where(mask)
    if len(yy) == 0:
        return image.copy(), {"shift_y": 0.0, "shift_x": 0.0}

    height, width = mask.shape
    centroid_y = float(yy.mean())
    centroid_x = float(xx.mean())
    center_y = (height - 1) / 2
    center_x = (width - 1) / 2

    if direction == "towards_center":
        shift_y = (center_y - centroid_y) * strength
        shift_x = (center_x - centroid_x) * strength
    elif direction == "away_from_center":
        shift_y = (centroid_y - center_y) * strength
        shift_x = (centroid_x - center_x) * strength
        if abs(shift_y) < 1 and abs(shift_x) < 1:
            shift_y = 0.12 * height
            shift_x = 0.12 * width
    else:
        raise ValueError(f"Unknown direction: {direction}")

    dy = int(round(shift_y))
    dx = int(round(shift_x))
    if max_shift is not None:
        dy = int(np.clip(dy, -max_shift, max_shift))
        dx = int(np.clip(dx, -max_shift, max_shift))

    if fill_mode == "black":
        fill_value = 0.0
    elif fill_mode == "mean":
        fill_value = float(image.mean())
    else:
        raise ValueError(f"Unknown fill mode: {fill_mode}")

    return translate_image(image, dy, dx, fill_value), {
        "centroid_y_before": centroid_y,
        "centroid_x_before": centroid_x,
        "shift_y": float(dy),
        "shift_x": float(dx),
        "fill_value": float(fill_value),
    }


INTERVENTIONS = [
    {
        "source": "glioma",
        "target": "meningioma",
        "name": "glioma_to_meningioma_tumor_contrast",
        "analysis_group": "main",
        "description": "Aclarar/aumentar contraste tumoral de glioma hacia meningioma",
        "ops": [
            ("match_tumor_mean", "meningioma", "tumor_intensity_mean"),
            ("match_tumor_outside_contrast", "meningioma", "tumor_vs_outside_r10_mean_diff", 10),
        ],
    },
    {
        "source": "glioma",
        "target": "pituitary",
        "name": "glioma_to_pituitary_center_context",
        "analysis_group": "main",
        "description": "Centrar encuadre y ajustar contexto de glioma hacia pituitary",
        "ops": [
            ("move_position", "towards_center"),
            ("match_outside_p50", "pituitary", "outside_peritumor_r10_intensity_p50", 10),
        ],
    },
    {
        "source": "glioma",
        "target": "pituitary",
        "name": "glioma_to_pituitary_position_only_black_fill",
        "analysis_group": "control_glioma_pituitary",
        "description": "Control: centrar glioma hacia pituitary con relleno negro, sin cambiar contexto",
        "ops": [
            ("move_position", "towards_center", "black"),
        ],
    },
    {
        "source": "glioma",
        "target": "pituitary",
        "name": "glioma_to_pituitary_position_only_mean_fill",
        "analysis_group": "control_glioma_pituitary",
        "description": "Control: centrar glioma hacia pituitary con relleno medio, sin cambiar contexto",
        "ops": [
            ("move_position", "towards_center", "mean"),
        ],
    },
    {
        "source": "glioma",
        "target": "pituitary",
        "name": "glioma_to_pituitary_context_only",
        "analysis_group": "control_glioma_pituitary",
        "description": "Control: ajustar solo contexto de glioma hacia pituitary, sin mover imagen",
        "ops": [
            ("match_outside_p50", "pituitary", "outside_peritumor_r10_intensity_p50", 10),
        ],
    },
    {
        "source": "glioma",
        "target": "pituitary",
        "name": "glioma_to_pituitary_small_shift_mean_fill",
        "analysis_group": "control_glioma_pituitary",
        "description": "Control: desplazamiento pequeno hacia centro con relleno medio, sin cambiar contexto",
        "ops": [
            ("move_position", "towards_center", "mean", 8),
        ],
    },
    {
        "source": "meningioma",
        "target": "glioma",
        "name": "meningioma_to_glioma_dark_low_contrast",
        "analysis_group": "main",
        "description": "Oscurecer/reducir contraste tumoral de meningioma hacia glioma",
        "ops": [
            ("match_tumor_mean", "glioma", "tumor_intensity_mean"),
            ("match_tumor_peritumor_contrast", "glioma", "tumor_vs_peritumor_r5_mean_diff", 5),
        ],
    },
    {
        "source": "meningioma",
        "target": "pituitary",
        "name": "meningioma_to_pituitary_center_context",
        "analysis_group": "main",
        "description": "Centrar encuadre y ajustar contexto de meningioma hacia pituitary",
        "ops": [
            ("move_position", "towards_center"),
            ("match_outside_p50", "pituitary", "outside_peritumor_r10_intensity_p50", 10),
        ],
    },
    {
        "source": "meningioma",
        "target": "pituitary",
        "name": "meningioma_to_pituitary_position_only_black_fill",
        "analysis_group": "control_meningioma_pituitary",
        "description": "Control: centrar meningioma hacia pituitary con relleno negro, sin cambiar contexto",
        "ops": [
            ("move_position", "towards_center", "black"),
        ],
    },
    {
        "source": "meningioma",
        "target": "pituitary",
        "name": "meningioma_to_pituitary_position_only_mean_fill",
        "analysis_group": "control_meningioma_pituitary",
        "description": "Control: centrar meningioma hacia pituitary con relleno medio, sin cambiar contexto",
        "ops": [
            ("move_position", "towards_center", "mean"),
        ],
    },
    {
        "source": "meningioma",
        "target": "pituitary",
        "name": "meningioma_to_pituitary_context_only",
        "analysis_group": "control_meningioma_pituitary",
        "description": "Control: ajustar solo contexto de meningioma hacia pituitary, sin mover imagen",
        "ops": [
            ("match_outside_p50", "pituitary", "outside_peritumor_r10_intensity_p50", 10),
        ],
    },
    {
        "source": "meningioma",
        "target": "pituitary",
        "name": "meningioma_to_pituitary_small_shift_mean_fill",
        "analysis_group": "control_meningioma_pituitary",
        "description": "Control: desplazamiento pequeno hacia centro con relleno medio, sin cambiar contexto",
        "ops": [
            ("move_position", "towards_center", "mean", 8),
        ],
    },
    {
        "source": "pituitary",
        "target": "glioma",
        "name": "pituitary_to_glioma_decenter_dark_context",
        "analysis_group": "main",
        "description": "Descentrar y oscurecer pituitary hacia patron de glioma",
        "ops": [
            ("move_position", "away_from_center"),
            ("match_tumor_mean", "glioma", "tumor_intensity_mean"),
        ],
    },
    {
        "source": "pituitary",
        "target": "glioma",
        "name": "pituitary_to_glioma_position_only_black_fill",
        "analysis_group": "control_pituitary_glioma",
        "description": "Control: descentrar pituitary hacia glioma con relleno negro, sin oscurecer tumor",
        "ops": [
            ("move_position", "away_from_center", "black"),
        ],
    },
    {
        "source": "pituitary",
        "target": "glioma",
        "name": "pituitary_to_glioma_position_only_mean_fill",
        "analysis_group": "control_pituitary_glioma",
        "description": "Control: descentrar pituitary hacia glioma con relleno medio, sin oscurecer tumor",
        "ops": [
            ("move_position", "away_from_center", "mean"),
        ],
    },
    {
        "source": "pituitary",
        "target": "glioma",
        "name": "pituitary_to_glioma_tumor_intensity_only",
        "analysis_group": "control_pituitary_glioma",
        "description": "Control: oscurecer solo el tumor de pituitary hacia glioma, sin mover imagen",
        "ops": [
            ("match_tumor_mean", "glioma", "tumor_intensity_mean"),
        ],
    },
    {
        "source": "pituitary",
        "target": "glioma",
        "name": "pituitary_to_glioma_small_shift_mean_fill",
        "analysis_group": "control_pituitary_glioma",
        "description": "Control: desplazamiento pequeno fuera del centro con relleno medio, sin oscurecer tumor",
        "ops": [
            ("move_position", "away_from_center", "mean", 8),
        ],
    },
    {
        "source": "pituitary",
        "target": "glioma",
        "name": "pituitary_to_glioma_position_intensity_mean_fill",
        "analysis_group": "control_pituitary_glioma",
        "description": "Control: descentrar pituitary con relleno medio y oscurecer tumor hacia glioma",
        "ops": [
            ("move_position", "away_from_center", "mean"),
            ("match_tumor_mean", "glioma", "tumor_intensity_mean"),
        ],
    },
    {
        "source": "pituitary",
        "target": "meningioma",
        "name": "pituitary_to_meningioma_tumor_contrast",
        "analysis_group": "main",
        "description": "Aumentar intensidad/contraste tumoral de pituitary hacia meningioma",
        "ops": [
            ("match_tumor_mean", "meningioma", "tumor_intensity_mean"),
            ("match_tumor_outside_contrast", "meningioma", "tumor_vs_outside_r10_mean_diff", 10),
        ],
    },
]


def apply_intervention(
    image: np.ndarray,
    mask: np.ndarray,
    intervention: dict[str, object],
    train_means: dict[tuple[str, str], float],
    *,
    strength: float,
    position_strength: float,
) -> tuple[np.ndarray, dict[str, float]]:
    edited = image.copy()
    metadata: dict[str, float] = {}

    for op in intervention["ops"]:
        op_name = op[0]
        if op_name == "move_position":
            fill_mode = op[2] if len(op) >= 3 else "black"
            max_shift = int(op[3]) if len(op) >= 4 else None
            edited, op_meta = move_tumor_position(
                edited,
                mask,
                direction=op[1],
                strength=position_strength,
                fill_mode=fill_mode,
                max_shift=max_shift,
            )
        elif op_name == "match_tumor_mean":
            _op_name, target_class, feature = op
            edited, op_meta = match_region_mean(
                edited,
                mask,
                train_means[(target_class, feature)],
                strength,
            )
        elif op_name == "match_tumor_outside_contrast":
            _op_name, target_class, feature, radius = op
            outside = ~dilate_mask(mask, int(radius))
            edited, op_meta = match_tumor_outside_contrast(
                edited,
                mask,
                outside,
                train_means[(target_class, feature)],
                strength,
            )
        elif op_name == "match_tumor_peritumor_contrast":
            _op_name, target_class, feature, radius = op
            dilated = dilate_mask(mask, int(radius))
            peritumor = np.logical_and(dilated, ~mask)
            edited, op_meta = match_tumor_outside_contrast(
                edited,
                mask,
                peritumor,
                train_means[(target_class, feature)],
                strength,
            )
        elif op_name == "match_outside_p50":
            _op_name, target_class, feature, radius = op
            outside = ~dilate_mask(mask, int(radius))
            edited, op_meta = match_outside_p50(
                edited,
                outside,
                train_means[(target_class, feature)],
                strength,
            )
        else:
            raise ValueError(f"Unknown intervention op: {op_name}")

        for key, value in op_meta.items():
            metadata[f"{op_name}_{key}"] = value

    return np.clip(edited, 0.0, 1.0).astype(np.float32), metadata


def image_panel(original: np.ndarray, edited: np.ndarray, title: str) -> Image.Image:
    def to_rgb(array: np.ndarray) -> Image.Image:
        channel = array[0] if array.shape[0] == 1 else np.transpose(array, (1, 2, 0))
        if channel.ndim == 2:
            return Image.fromarray(np.uint8(np.clip(channel, 0, 1) * 255), mode="L").convert("RGB")
        return Image.fromarray(np.uint8(np.clip(channel, 0, 1) * 255), mode="RGB")

    original_pil = to_rgb(original)
    edited_pil = to_rgb(edited)
    diff = np.abs(edited - original).mean(axis=0)
    diff_pil = Image.fromarray(np.uint8(np.clip(diff / max(float(diff.max()), 1e-6), 0, 1) * 255), mode="L").convert("RGB")
    width, height = original_pil.size
    canvas = Image.new("RGB", (width * 3, height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), title, fill="black")
    canvas.paste(original_pil, (0, 28))
    canvas.paste(edited_pil, (width, 28))
    canvas.paste(diff_pil, (width * 2, 28))
    return canvas


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    summary_rows = []
    group_columns = [
        "analysis_group",
        "intervention",
        "intervention_description",
        "source",
        "target",
    ]
    for (analysis_group, intervention, description, source, target), group in df.groupby(group_columns):
        for model in ["multi", "ova"]:
            pred_original = group[f"{model}_pred_original"]
            pred_intervened = group[f"{model}_pred_intervened"]
            source_recall_original = float((pred_original == source).mean())
            source_recall_intervened = float((pred_intervened == source).mean())
            predicted_target_intervened = float((pred_intervened == target).mean())
            summary_rows.append(
                {
                    "intervention": intervention,
                    "analysis_group": analysis_group,
                    "intervention_description": description,
                    "source": source,
                    "target": target,
                    "model": model,
                    "n": int(len(group)),
                    "delta_source_prob_mean": float(group[f"{model}_delta_source_prob"].mean()),
                    "delta_target_prob_mean": float(group[f"{model}_delta_target_prob"].mean()),
                    "target_gain_rate": float((group[f"{model}_delta_target_prob"] > 0).mean()),
                    "flip_to_target_rate": float(group[f"{model}_flip_to_target"].mean()),
                    "source_recall_original": source_recall_original,
                    "source_recall_intervened": source_recall_intervened,
                    "source_fnr_intervened": 1.0 - source_recall_intervened,
                    "predicted_target_intervened_rate": predicted_target_intervened,
                }
            )

    pd.DataFrame(summary_rows).to_csv(output_path, index=False)


def write_class_delta_summary(rows: list[dict[str, object]], output_path: Path, labels: list[str]) -> None:
    df = pd.DataFrame(rows)
    summary_rows = []
    group_columns = [
        "analysis_group",
        "intervention",
        "intervention_description",
        "source",
        "target",
    ]
    for (analysis_group, intervention, description, source, target), group in df.groupby(group_columns):
        for model in ["multi", "ova"]:
            for label in labels:
                delta_col = f"{model}_{label}_delta_prob"
                summary_rows.append(
                    {
                        "intervention": intervention,
                        "analysis_group": analysis_group,
                        "intervention_description": description,
                        "source": source,
                        "target": target,
                        "model": model,
                        "class_name": label,
                        "n": int(len(group)),
                        "prob_original_mean": float(group[f"{model}_{label}_prob_original"].mean()),
                        "prob_intervened_mean": float(group[f"{model}_{label}_prob_intervened"].mean()),
                        "delta_prob_mean": float(group[delta_col].mean()),
                        "gain_rate": float((group[delta_col] > 0).mean()),
                    }
                )

    pd.DataFrame(summary_rows).to_csv(output_path, index=False)


def main() -> None:
    parsed = build_parser().parse_args()
    ensure_checkpoint_attrs(parsed)
    xai_args = to_explicabilidad_args(parsed)
    args = experiment_args(xai_args)
    set_seed(parsed.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(parsed.output_dir) / f"seed_{parsed.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    if parsed.save_examples > 0:
        examples_dir.mkdir(parents=True, exist_ok=True)

    train_means = load_train_means(Path(parsed.shortcut_stats_csv))

    print(f"Loading BRISC with seed={parsed.seed}", flush=True)
    experiment_data = load_experiment_data(args, parsed.seed)
    print(
        f"Preparing models on device={device}, train={len(experiment_data.y_train)}, "
        f"val={len(experiment_data.y_val)}, test={len(experiment_data.y_test)}",
        flush=True,
    )
    if parsed.reuse_checkpoints and checkpoints_exist(parsed, experiment_data.target_dim):
        multi_model, ova_models = load_model_checkpoints(parsed, experiment_data, device)
    else:
        multi_model, _, multi_training_info = train_multiclass_model(
            experiment_data,
            args,
            device,
            parsed.seed,
        )
        ova_models, _, ova_training_info = train_ova_models(
            experiment_data,
            args,
            device,
            parsed.seed,
        )
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
        print("CHECKPOINT_ONLY=1, skipping shortcut interventions.", flush=True)
        return

    X_test = experiment_data.X_test
    y_true = experiment_data.y_test.astype(int)
    labels = CLASS_LABELS["brisc"]
    image_paths = test_image_paths(xai_args, len(y_true))
    mask_paths = [brisc_mask_path(xai_args, path) for path in image_paths]

    print(f"Predicting original test set ({len(y_true)} images)", flush=True)
    multi_probs = predict_multi(multi_model, X_test, device)
    ova_probs = predict_ova(ova_models, X_test, device)
    multi_pred = multi_probs.argmax(axis=1)
    ova_pred = ova_probs.argmax(axis=1)

    selected_by_source: dict[str, list[int]] = {}
    for source in sorted({str(item["source"]) for item in INTERVENTIONS}):
        source_idx = CLASS_TO_IDX[source]
        selected = [
            idx
            for idx, target in enumerate(y_true)
            if int(target) == source_idx and mask_paths[idx] is not None
        ]
        if parsed.max_cases_per_source is not None:
            selected = selected[: parsed.max_cases_per_source]
        selected_by_source[source] = selected

    intervened_images: list[np.ndarray] = []
    intervened_meta: list[dict[str, object]] = []
    examples_saved: dict[str, int] = {}
    print("Creating shortcut interventions", flush=True)
    for intervention in INTERVENTIONS:
        source = str(intervention["source"])
        target = str(intervention["target"])
        for idx in selected_by_source[source]:
            width, height = X_test[idx].shape[-1], X_test[idx].shape[-2]
            mask = load_mask(mask_paths[idx], width, height)
            if mask is None:
                continue
            edited, op_meta = apply_intervention(
                X_test[idx],
                mask,
                intervention,
                train_means,
                strength=parsed.intervention_strength,
                position_strength=parsed.position_strength,
            )
            row = {
                "idx": idx,
                "intervention": str(intervention["name"]),
                "analysis_group": str(intervention["analysis_group"]),
                "intervention_description": str(intervention["description"]),
                "source": source,
                "target": target,
                "image_path": str(image_paths[idx]) if image_paths[idx] is not None else "",
                "mask_path": str(mask_paths[idx]) if mask_paths[idx] is not None else "",
                **op_meta,
            }
            intervened_images.append(edited)
            intervened_meta.append(row)

            saved = examples_saved.get(str(intervention["name"]), 0)
            if parsed.save_examples > 0 and saved < parsed.save_examples:
                panel = image_panel(
                    X_test[idx],
                    edited,
                    f"{intervention['name']} | idx={idx}",
                )
                panel.save(examples_dir / f"{intervention['name']}_idx_{idx:04d}.png")
                examples_saved[str(intervention["name"])] = saved + 1

    X_intervened = np.asarray(intervened_images, dtype=np.float32)
    print(f"Predicting intervened images ({len(X_intervened)} images)", flush=True)
    multi_probs_int = predict_multi(multi_model, X_intervened, device)
    ova_probs_int = predict_ova(ova_models, X_intervened, device)
    multi_pred_int = multi_probs_int.argmax(axis=1)
    ova_pred_int = ova_probs_int.argmax(axis=1)

    rows: list[dict[str, object]] = []
    for row_idx, meta in enumerate(intervened_meta):
        idx = int(meta["idx"])
        source_idx = CLASS_TO_IDX[str(meta["source"])]
        target_idx = CLASS_TO_IDX[str(meta["target"])]
        row = dict(meta)
        row.update(
            {
                "true_label": labels[int(y_true[idx])],
                "multi_pred_original": labels[int(multi_pred[idx])],
                "multi_pred_intervened": labels[int(multi_pred_int[row_idx])],
                "ova_pred_original": labels[int(ova_pred[idx])],
                "ova_pred_intervened": labels[int(ova_pred_int[row_idx])],
            }
        )
        for model, probs_before, probs_after, pred_before, pred_after in [
            ("multi", multi_probs[idx], multi_probs_int[row_idx], multi_pred[idx], multi_pred_int[row_idx]),
            ("ova", ova_probs[idx], ova_probs_int[row_idx], ova_pred[idx], ova_pred_int[row_idx]),
        ]:
            for class_idx, label in enumerate(labels):
                prob_before = float(probs_before[class_idx])
                prob_after = float(probs_after[class_idx])
                row.update(
                    {
                        f"{model}_{label}_prob_original": prob_before,
                        f"{model}_{label}_prob_intervened": prob_after,
                        f"{model}_{label}_delta_prob": prob_after - prob_before,
                    }
                )
            source_before = float(probs_before[source_idx])
            source_after = float(probs_after[source_idx])
            target_before = float(probs_before[target_idx])
            target_after = float(probs_after[target_idx])
            row.update(
                {
                    f"{model}_source_prob_original": source_before,
                    f"{model}_source_prob_intervened": source_after,
                    f"{model}_delta_source_prob": source_after - source_before,
                    f"{model}_target_prob_original": target_before,
                    f"{model}_target_prob_intervened": target_after,
                    f"{model}_delta_target_prob": target_after - target_before,
                    f"{model}_pred_changed": float(int(pred_before) != int(pred_after)),
                    f"{model}_flip_to_target": float(int(pred_after) == target_idx and int(pred_before) != target_idx),
                    f"{model}_correct_original": float(int(pred_before) == int(y_true[idx])),
                    f"{model}_correct_intervened": float(int(pred_after) == int(y_true[idx])),
                }
            )
        rows.append(row)

    index_path = output_dir / "shortcut_intervention_index.csv"
    with index_path.open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "shortcut_intervention_summary.csv"
    write_summary(rows, summary_path)
    class_delta_path = output_dir / "shortcut_intervention_class_delta_summary.csv"
    write_class_delta_summary(rows, class_delta_path, labels)
    print(f"Saved intervention index to {index_path}", flush=True)
    print(f"Saved intervention summary to {summary_path}", flush=True)
    print(f"Saved class delta summary to {class_delta_path}", flush=True)
    if parsed.save_examples > 0:
        print(f"Saved examples to {examples_dir}", flush=True)


if __name__ == "__main__":
    main()
