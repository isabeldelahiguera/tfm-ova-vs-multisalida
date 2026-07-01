from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from checkpoint_utils import load_model_checkpoints  # noqa: E402
from evaluate_checkpoints import experiment_args  # noqa: E402
from tfm.data import load_experiment_data  # noqa: E402
from tfm.experiment import predict_multiclass_probabilities, predict_ova_probabilities  # noqa: E402
from tfm.metrics import classification_metrics  # noqa: E402


FINAL_CLASSES = ["akiec", "bcc", "mel", "non_malignant"]
MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}


def parsed_args(seed: int, run_tag: str, label_mode: str, exclude_classes: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="ham10000",
        model_arch="vgg16-pretrained",
        seeds=[seed],
        seed=seed,
        run_tag=run_tag,
        checkpoint_run_tag=run_tag,
        checkpoint_dir=str(REPO_ROOT / "resultados_actualizados/checkpoints"),
        vgg_channels=[32, 64, 128],
        batch_size=16,
        epochs=50,
        early_stopping_patience=10,
        early_stopping_min_delta=1e-4,
        learning_rate=1e-4,
        class_weighting="none",
        data_augmentation="none",
        train_sampler="balanced",
        pretrained_finetune="block5",
        image_size=224,
        max_train=None,
        max_test=None,
        brisc_root=str(REPO_ROOT / "data/brisc2025"),
        tb_root=str(REPO_ROOT / "data/tb_chest_xray"),
        ham10000_root=str(REPO_ROOT / "data/ham10000"),
        ham10000_test="internal",
        ham10000_split_csv=None,
        ham10000_split_seed=2000,
        ham10000_exclude_classes=exclude_classes or [],
        ham10000_label_mode=label_mode,
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    row_sum = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum != 0)


def final_targets(class_names: list[str], y_test: np.ndarray) -> np.ndarray:
    targets = []
    for idx in y_test.astype(int):
        class_name = class_names[int(idx)]
        final_name = class_name if class_name in MALIGNANT_CLASSES else "non_malignant"
        targets.append(FINAL_CLASSES.index(final_name))
    return np.asarray(targets, dtype=int)


def grouped_7class_probabilities(probs_7: np.ndarray, class_names: list[str]) -> np.ndarray:
    class_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    grouped = np.zeros((len(probs_7), len(FINAL_CLASSES)), dtype=float)
    grouped[:, 0] = probs_7[:, class_idx["akiec"]]
    grouped[:, 1] = probs_7[:, class_idx["bcc"]]
    grouped[:, 2] = probs_7[:, class_idx["mel"]]
    grouped[:, 3] = (
        probs_7[:, class_idx["bkl"]]
        + probs_7[:, class_idx["df"]]
        + probs_7[:, class_idx["nv"]]
        + probs_7[:, class_idx["vasc"]]
    )
    return normalize_rows(grouped)


def nested_probabilities(binary_probs: np.ndarray, malignant_probs: np.ndarray) -> np.ndarray:
    final = np.zeros((len(binary_probs), len(FINAL_CLASSES)), dtype=float)
    final[:, 0] = binary_probs[:, 1] * malignant_probs[:, 0]
    final[:, 1] = binary_probs[:, 1] * malignant_probs[:, 1]
    final[:, 2] = binary_probs[:, 1] * malignant_probs[:, 2]
    final[:, 3] = binary_probs[:, 0]
    return normalize_rows(final)


def collect_7class(seed: int, device: torch.device):
    parsed = parsed_args(seed, "vgg16_block5_sampler_balanced_10seeds_checkpoints", "original")
    args = experiment_args(parsed, seed)
    data = load_experiment_data(args, seed)
    multi_model, ova_models = load_model_checkpoints(parsed, data, device, lambda _parsed: args)
    multi_probs = predict_multiclass_probabilities(multi_model, data, args, device)
    ova_probs = normalize_rows(predict_ova_probabilities(ova_models, data, args, device))
    return data, grouped_7class_probabilities(multi_probs, list(data.class_names)), grouped_7class_probabilities(ova_probs, list(data.class_names))


def collect_nested(seed: int, full_data, device: torch.device):
    binary_parsed = parsed_args(
        seed,
        "vgg16_block5_sampler_balanced_malignant_binary_10seeds_checkpoints",
        "malignant_binary",
    )
    binary_args = experiment_args(binary_parsed, seed)
    binary_data = load_experiment_data(binary_args, seed)
    binary_multi, binary_ova = load_model_checkpoints(binary_parsed, binary_data, device, lambda _parsed: binary_args)
    binary_multi_probs = predict_multiclass_probabilities(binary_multi, binary_data, binary_args, device)
    binary_ova_probs = normalize_rows(predict_ova_probabilities(binary_ova, binary_data, binary_args, device))

    malignant_parsed = parsed_args(
        seed,
        "vgg16_block5_sampler_balanced_true3classes_10seeds_checkpoints",
        "original",
        ["bkl", "df", "nv", "vasc"],
    )
    malignant_args = experiment_args(malignant_parsed, seed)
    malignant_data = load_experiment_data(malignant_args, seed)
    malignant_multi, malignant_ova = load_model_checkpoints(
        malignant_parsed, malignant_data, device, lambda _parsed: malignant_args
    )

    # The 3-class classifier is evaluated over the full HAM10000 test split so
    # each image receives conditional scores for akiec/bcc/mel.
    malignant_eval_data = SimpleNamespace(**malignant_data.__dict__)
    malignant_eval_data.X_test = full_data.X_test
    malignant_eval_data.y_test = np.zeros(len(full_data.y_test), dtype=int)
    malignant_multi_probs = predict_multiclass_probabilities(
        malignant_multi, malignant_eval_data, malignant_args, device
    )
    malignant_ova_probs = normalize_rows(
        predict_ova_probabilities(malignant_ova, malignant_eval_data, malignant_args, device)
    )

    return (
        nested_probabilities(binary_multi_probs, malignant_multi_probs),
        nested_probabilities(binary_ova_probs, malignant_ova_probs),
    )


def add_metrics(rows: list[dict[str, object]], seed: int, approach: str, model: str, y_true: np.ndarray, probs: np.ndarray):
    y_pred = probs.argmax(axis=1)
    rows.append({"seed": seed, "Enfoque": approach, "Modelo": model, **classification_metrics(y_true, y_pred)})
    return y_pred


def add_confusion(rows: list[dict[str, object]], seed: int, approach: str, model: str, y_true: np.ndarray, y_pred: np.ndarray):
    for true_idx, true_name in enumerate(FINAL_CLASSES):
        for pred_idx, pred_name in enumerate(FINAL_CLASSES):
            count = int(((y_true == true_idx) & (y_pred == pred_idx)).sum())
            rows.append(
                {
                    "seed": seed,
                    "Enfoque": approach,
                    "Modelo": model,
                    "true": true_name,
                    "pred": pred_name,
                    "count": count,
                }
            )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing HAM10000 nested probabilities on device={device}", flush=True)

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    for seed in range(1, 11):
        full_data, grouped_multi, grouped_ova = collect_7class(seed, device)
        nested_multi, nested_ova = collect_nested(seed, full_data, device)
        y_true = final_targets(list(full_data.class_names), full_data.y_test)

        for approach, model, probs in [
            ("7 clases agr. prob.", "Multi", grouped_multi),
            ("7 clases agr. prob.", "OVA", grouped_ova),
            ("Nested prob.", "Multi", nested_multi),
            ("Nested prob.", "OVA", nested_ova),
        ]:
            y_pred = add_metrics(metric_rows, seed, approach, model, y_true, probs)
            add_confusion(confusion_rows, seed, approach, model, y_true, y_pred)

        print(f"Finished seed {seed}", flush=True)

    out_dir = REPO_ROOT / "resultados_actualizados/secuencial"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "ham10000_nested_probability_metrics_10seeds.csv"
    metrics.to_csv(metrics_path, index=False)

    confusion = pd.DataFrame(confusion_rows)
    confusion_mean = (
        confusion.groupby(["Enfoque", "Modelo", "true", "pred"], as_index=False)["count"]
        .mean()
        .sort_values(["Enfoque", "Modelo", "true", "pred"])
    )
    confusion_path = out_dir / "ham10000_nested_probability_confusion_mean_10seeds.csv"
    confusion_mean.to_csv(confusion_path, index=False)

    summary = (
        metrics.groupby(["Enfoque", "Modelo"])[["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_path = out_dir / "ham10000_nested_probability_summary_10seeds.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved {metrics_path}", flush=True)
    print(f"Saved {confusion_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
