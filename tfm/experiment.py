from __future__ import annotations

import itertools
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from .config import ExperimentData
from .data import STANDARD_CLASSIFICATION_DATASETS, STANDARD_REGRESSION_DATASETS, load_experiment_data
from .evaluation import (
    evaluate_decoupled_regression,
    evaluate_multiclass_model,
    evaluate_multioutput_regression,
    evaluate_ova_ensemble,
    evaluate_ovo_ensemble,
)
from .metrics import classification_metrics
from .models import build_model
from .training import build_test_loader, collect_predictions, create_data_loaders, fit_model, set_seed


def fit_experiment_model(model, train_loader, val_loader, criterion, optimizer, device, problem_mode: str, args):
    return fit_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        problem_mode,
        args.epochs,
        args.early_stopping_patience,
        args.early_stopping_min_delta,
    )


def summarize_training_infos(training_infos: List[Dict[str, float]]) -> Dict[str, float]:
    best_val_losses = [info["best_val_loss"] for info in training_infos]
    epochs_trained = [info["epochs_trained"] for info in training_infos]
    stopped_early = [info["stopped_early"] for info in training_infos]
    return {
        "best_val_loss": float(np.mean(best_val_losses)),
        "epochs_trained": float(np.mean(epochs_trained)),
        "total_epochs_trained": int(np.sum(epochs_trained)),
        "stopped_early": bool(np.any(stopped_early)),
        "models_stopped_early": int(np.sum(stopped_early)),
    }


def model_input_metadata(experiment_data: ExperimentData):
    input_shape = tuple(experiment_data.X_train.shape[1:])
    input_dim = int(np.prod(input_shape))
    return input_dim, input_shape


def build_experiment_model(experiment_data: ExperimentData, args, output_dim: int):
    input_dim, input_shape = model_input_metadata(experiment_data)
    return build_model(
        input_dim=input_dim,
        hidden_layers=args.hidden_layers,
        output_dim=output_dim,
        batch_norm=args.batch_normalization,
        model_arch=args.model_arch,
        input_shape=input_shape,
        vgg_channels=args.vgg_channels,
        pretrained_finetune=getattr(args, "pretrained_finetune", "frozen"),
    )


def balanced_class_weights(y: np.ndarray, n_classes: int, device) -> torch.Tensor:
    classes = np.arange(n_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y.astype(int),
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)

def ova_pos_weight(y: np.ndarray, class_idx: int, device) -> torch.Tensor:
    positives = int(np.sum(y == class_idx))
    negatives = int(len(y) - positives)
    if positives == 0:
        raise ValueError(f"Cannot compute pos_weight for class {class_idx}: zero positive train samples")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal = (1.0 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = alpha_t * focal
        return focal.mean()


def focal_alpha(y: np.ndarray, class_idx: int, args) -> float | None:
    alpha = getattr(args, "focal_alpha", "balanced")
    if alpha == "none":
        return None
    if alpha == "balanced":
        positives = int(np.sum(y == class_idx))
        negatives = int(len(y) - positives)
        if positives == 0 or negatives == 0:
            raise ValueError(f"Cannot compute focal alpha for class {class_idx}: degenerate train split")
        return negatives / (positives + negatives)
    value = float(alpha)
    if value <= 0.0 or value >= 1.0:
        raise ValueError("focal_alpha as a numeric value must be in (0, 1)")
    return value


def train_multiclass_model(experiment_data: ExperimentData, args, device, seed: int):
    print(f"[seed={seed}] Training multi-output classification model", flush=True)
    train_loader, val_loader = create_data_loaders(
        experiment_data.X_train,
        experiment_data.y_train,
        experiment_data.X_val,
        experiment_data.y_val,
        args.batch_size,
        seed,
        torch.long,
        getattr(args, "data_augmentation", "none"),
        getattr(args, "train_sampler", "none"),
        experiment_data.y_train,
    )
    model = build_experiment_model(experiment_data, args, experiment_data.target_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    class_weighting = getattr(args, "class_weighting", "none")
    if class_weighting == "balanced":
        criterion = nn.CrossEntropyLoss(
            weight=balanced_class_weights(experiment_data.y_train, experiment_data.target_dim, device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    start = time.time()
    training_info = fit_experiment_model(model, train_loader, val_loader, criterion, optimizer, device, "multiclass", args)
    return model, time.time() - start, training_info


def train_ova_models(experiment_data: ExperimentData, args, device, seed: int):
    models = []
    training_infos = []
    start = time.time()

    for class_idx in range(experiment_data.target_dim):
        set_seed(seed)
        print(f"[seed={seed}] Training OVA model {class_idx + 1}/{experiment_data.target_dim}", flush=True)
        y_train_binary = (experiment_data.y_train == class_idx).astype(np.float32)
        y_val_binary = (experiment_data.y_val == class_idx).astype(np.float32)
        train_loader, val_loader = create_data_loaders(
            experiment_data.X_train,
            y_train_binary,
            experiment_data.X_val,
            y_val_binary,
            args.batch_size,
            seed,
            torch.float32,
            getattr(args, "data_augmentation", "none"),
            getattr(args, "train_sampler", "none"),
            experiment_data.y_train,
        )
        model = build_experiment_model(experiment_data, args, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        class_weighting = getattr(args, "class_weighting", "none")
        ova_loss = getattr(args, "ova_loss", "bce")
        if ova_loss == "focal":
            criterion = BinaryFocalLoss(
                gamma=float(getattr(args, "focal_gamma", 2.0)),
                alpha=focal_alpha(experiment_data.y_train, class_idx, args),
            )
        elif class_weighting == "balanced":
            criterion = nn.BCEWithLogitsLoss(pos_weight=ova_pos_weight(experiment_data.y_train, class_idx, device))
        else:
            criterion = nn.BCEWithLogitsLoss()
        training_infos.append(fit_experiment_model(model, train_loader, val_loader, criterion, optimizer, device, "binary", args))
        models.append(model)

    return models, time.time() - start, summarize_training_infos(training_infos)


def train_ovo_models(experiment_data: ExperimentData, args, device, seed: int):
    pair_models = {}
    training_infos = []
    start = time.time()
    pairs = list(itertools.combinations(range(experiment_data.target_dim), 2))

    for offset, (class_a, class_b) in enumerate(pairs):
        print(
            f"[seed={seed}] Training OVO model {offset + 1}/{len(pairs)} "
            f"(class {class_a} vs {class_b})",
            flush=True,
        )
        train_mask = np.isin(experiment_data.y_train, [class_a, class_b])
        val_mask = np.isin(experiment_data.y_val, [class_a, class_b])

        train_loader, val_loader = create_data_loaders(
            experiment_data.X_train[train_mask],
            (experiment_data.y_train[train_mask] == class_b).astype(np.float32),
            experiment_data.X_val[val_mask],
            (experiment_data.y_val[val_mask] == class_b).astype(np.float32),
            args.batch_size,
            seed + offset,
            torch.float32,
            getattr(args, "data_augmentation", "none"),
            getattr(args, "train_sampler", "none"),
            experiment_data.y_train[train_mask],
        )
        model = build_experiment_model(experiment_data, args, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        training_infos.append(fit_experiment_model(model, train_loader, val_loader, criterion, optimizer, device, "binary", args))
        pair_models[(class_a, class_b)] = model

    return pair_models, time.time() - start, summarize_training_infos(training_infos)


def prediction_outcome(y_true: int, multi_pred: int | None, ova_pred: int | None) -> str:
    if multi_pred is None or ova_pred is None:
        return "unavailable"
    multi_correct = multi_pred == y_true
    ova_correct = ova_pred == y_true
    if multi_correct and ova_correct:
        return "both_correct"
    if multi_correct and not ova_correct:
        return "multi_correct_ova_wrong"
    if not multi_correct and ova_correct:
        return "multi_wrong_ova_correct"
    return "both_wrong"


def plane_from_image_path(path: str) -> tuple[str, str]:
    match = re.search(r"_(ax|co|sa)_", Path(path).stem.lower())
    if not match:
        return "unknown", "unknown"
    plane = match.group(1)
    return plane, {"ax": "axial", "co": "coronal", "sa": "sagittal"}[plane]


def brisc_test_image_paths(args, class_names: list[str]) -> list[str]:
    root = Path(getattr(args, "brisc_root", "./data/brisc2025")).expanduser()
    max_test = getattr(args, "max_test", None)
    per_class_limit = None
    extra_samples = 0
    if max_test is not None:
        per_class_limit, extra_samples = divmod(max_test, len(class_names))

    paths: list[str] = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = root / "test" / class_name
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if per_class_limit is not None:
            class_limit = per_class_limit + int(class_idx < extra_samples)
            image_paths = image_paths[:class_limit]
        paths.extend(str(path) for path in image_paths)
    return paths


def test_image_paths(args, experiment_data: ExperimentData) -> list[str]:
    if experiment_data.test_image_paths and len(experiment_data.test_image_paths) == len(experiment_data.y_test):
        return experiment_data.test_image_paths
    if args.dataset == "brisc":
        paths = brisc_test_image_paths(args, experiment_data.class_names)
        if len(paths) == len(experiment_data.y_test):
            return paths
    return ["" for _ in range(len(experiment_data.y_test))]


def predict_multiclass_probabilities(model, experiment_data: ExperimentData, args, device) -> np.ndarray:
    test_loader = build_test_loader(experiment_data.X_test, experiment_data.y_test, args.batch_size, torch.long)
    _, logits = collect_predictions(model, test_loader, device)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def collect_ova_logits(models, X: np.ndarray, y: np.ndarray, args, device) -> tuple[np.ndarray, np.ndarray]:
    logits_by_class = []
    y_true = None
    for model in models:
        loader = build_test_loader(X, y, args.batch_size, torch.long)
        batch_targets, logits = collect_predictions(model, loader, device)
        if y_true is None:
            y_true = batch_targets.astype(int)
        logits_by_class.append(logits.squeeze(-1))
    if y_true is None:
        raise ValueError("Cannot collect OVA logits without trained OVA models")
    return y_true, np.column_stack(logits_by_class)


def predict_ova_probabilities(models, experiment_data: ExperimentData, args, device) -> np.ndarray:
    _, logits = collect_ova_logits(models, experiment_data.X_test, experiment_data.y_test, args, device)
    return expit(logits)


def fit_ova_platt_calibrators(models, experiment_data: ExperimentData, args, device) -> list[LogisticRegression]:
    y_val, val_logits = collect_ova_logits(models, experiment_data.X_val, experiment_data.y_val, args, device)
    calibrators = []
    for class_idx in range(experiment_data.target_dim):
        y_binary = (y_val == class_idx).astype(int)
        if len(np.unique(y_binary)) < 2:
            raise ValueError(f"Cannot calibrate OVA class {class_idx}: validation split has a single binary label")
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(val_logits[:, [class_idx]], y_binary)
        calibrators.append(calibrator)
    return calibrators


def predict_ova_platt_probabilities(models, experiment_data: ExperimentData, args, device) -> np.ndarray:
    calibrators = fit_ova_platt_calibrators(models, experiment_data, args, device)
    _, test_logits = collect_ova_logits(models, experiment_data.X_test, experiment_data.y_test, args, device)
    calibrated = [
        calibrator.predict_proba(test_logits[:, [class_idx]])[:, 1]
        for class_idx, calibrator in enumerate(calibrators)
    ]
    return np.column_stack(calibrated)


def fit_ova_thresholds(models, experiment_data: ExperimentData, args, device) -> np.ndarray:
    y_val, val_logits = collect_ova_logits(models, experiment_data.X_val, experiment_data.y_val, args, device)
    val_probs = expit(val_logits)
    candidates = np.linspace(0.01, 0.99, 99)
    thresholds = []

    for class_idx in range(experiment_data.target_dim):
        y_binary = (y_val == class_idx).astype(int)
        best_threshold = 0.5
        best_score = -np.inf

        for threshold in candidates:
            y_pred = (val_probs[:, class_idx] >= threshold).astype(int)
            score = balanced_accuracy_score(y_binary, y_pred)
            if score > best_score or (
                np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            ):
                best_score = score
                best_threshold = float(threshold)

        thresholds.append(best_threshold)

    return np.asarray(thresholds, dtype=np.float64)


def fit_ova_thresholds_f1_macro(models, experiment_data: ExperimentData, args, device) -> np.ndarray:
    y_val, val_logits = collect_ova_logits(models, experiment_data.X_val, experiment_data.y_val, args, device)
    val_probs = expit(val_logits)
    candidates = np.linspace(0.01, 0.99, 99)
    thresholds = np.full(experiment_data.target_dim, 0.5, dtype=np.float64)
    best_score = f1_score(y_val, (val_probs - thresholds[np.newaxis, :]).argmax(axis=1), average="macro", zero_division=0)

    for _round in range(3):
        improved = False
        for class_idx in range(experiment_data.target_dim):
            best_class_threshold = thresholds[class_idx]
            for threshold in candidates:
                candidate_thresholds = thresholds.copy()
                candidate_thresholds[class_idx] = threshold
                y_pred = (val_probs - candidate_thresholds[np.newaxis, :]).argmax(axis=1)
                score = f1_score(y_val, y_pred, average="macro", zero_division=0)
                if score > best_score or (
                    np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_class_threshold - 0.5)
                ):
                    best_score = score
                    best_class_threshold = float(threshold)

            if not np.isclose(best_class_threshold, thresholds[class_idx]):
                thresholds[class_idx] = best_class_threshold
                improved = True

        if not improved:
            break

    return thresholds


def predict_ova_threshold_scores(models, experiment_data: ExperimentData, args, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if getattr(args, "ova_calibration", "none") == "threshold-f1":
        thresholds = fit_ova_thresholds_f1_macro(models, experiment_data, args, device)
    else:
        thresholds = fit_ova_thresholds(models, experiment_data, args, device)
    _, test_logits = collect_ova_logits(models, experiment_data.X_test, experiment_data.y_test, args, device)
    test_probs = expit(test_logits)
    adjusted_scores = test_probs - thresholds[np.newaxis, :]
    return adjusted_scores, test_probs, thresholds


def evaluate_ova_platt_ensemble(models, experiment_data: ExperimentData, args, device) -> Dict[str, float]:
    calibrated_probs = predict_ova_platt_probabilities(models, experiment_data, args, device)
    y_pred = calibrated_probs.argmax(axis=1)
    return classification_metrics(experiment_data.y_test.astype(int), y_pred)


def evaluate_ova_threshold_ensemble(models, experiment_data: ExperimentData, args, device) -> Dict[str, float]:
    adjusted_scores, _test_probs, _thresholds = predict_ova_threshold_scores(models, experiment_data, args, device)
    y_pred = adjusted_scores.argmax(axis=1)
    return classification_metrics(experiment_data.y_test.astype(int), y_pred)


def save_classification_predictions(
    args,
    experiment_data: ExperimentData,
    device,
    seed: int,
    multi_model=None,
    ova_models=None,
) -> None:
    predictions_csv = getattr(args, "predictions_csv", None)
    if not predictions_csv:
        return

    class_names = list(experiment_data.class_names)
    y_true = experiment_data.y_test.astype(int)
    image_paths = test_image_paths(args, experiment_data)

    multi_probs = None
    multi_pred = None
    if multi_model is not None:
        multi_probs = predict_multiclass_probabilities(multi_model, experiment_data, args, device)
        multi_pred = multi_probs.argmax(axis=1)

    ova_probs = None
    ova_pred = None
    if ova_models is not None:
        ova_probs = predict_ova_probabilities(ova_models, experiment_data, args, device)
        ova_pred = ova_probs.argmax(axis=1)

    ova_platt_probs = None
    ova_platt_pred = None
    if ova_models is not None and getattr(args, "ova_calibration", "none") == "platt":
        ova_platt_probs = predict_ova_platt_probabilities(ova_models, experiment_data, args, device)
        ova_platt_pred = ova_platt_probs.argmax(axis=1)

    ova_threshold_scores = None
    ova_threshold_raw_probs = None
    ova_thresholds = None
    ova_threshold_pred = None
    if ova_models is not None and getattr(args, "ova_calibration", "none") in {"threshold", "threshold-f1"}:
        ova_threshold_scores, ova_threshold_raw_probs, ova_thresholds = predict_ova_threshold_scores(
            ova_models, experiment_data, args, device
        )
        ova_threshold_pred = ova_threshold_scores.argmax(axis=1)

    rows = []
    for idx, target in enumerate(y_true):
        image_path = image_paths[idx]
        plane, plane_name = plane_from_image_path(image_path) if image_path else ("unknown", "unknown")
        row: dict[str, object] = {
            "seed": seed,
            "test_index": idx,
            "image_path": image_path,
            "true_label": class_names[target],
            "plane": plane,
            "plane_name": plane_name,
        }
        multi_idx = int(multi_pred[idx]) if multi_pred is not None else None
        ova_idx = int(ova_pred[idx]) if ova_pred is not None else None
        ova_platt_idx = int(ova_platt_pred[idx]) if ova_platt_pred is not None else None
        ova_threshold_idx = int(ova_threshold_pred[idx]) if ova_threshold_pred is not None else None
        row["multi_pred"] = class_names[multi_idx] if multi_idx is not None else ""
        row["ova_pred"] = class_names[ova_idx] if ova_idx is not None else ""
        row["ova_platt_pred"] = class_names[ova_platt_idx] if ova_platt_idx is not None else ""
        row["ova_threshold_pred"] = class_names[ova_threshold_idx] if ova_threshold_idx is not None else ""
        row["multi_correct"] = bool(multi_idx == target) if multi_idx is not None else ""
        row["ova_correct"] = bool(ova_idx == target) if ova_idx is not None else ""
        row["ova_platt_correct"] = bool(ova_platt_idx == target) if ova_platt_idx is not None else ""
        row["ova_threshold_correct"] = bool(ova_threshold_idx == target) if ova_threshold_idx is not None else ""
        row["outcome"] = prediction_outcome(int(target), multi_idx, ova_idx)

        if multi_probs is not None:
            row["multi_confidence"] = float(multi_probs[idx, multi_idx])
            row["multi_true_prob"] = float(multi_probs[idx, target])
            for class_idx, class_name in enumerate(class_names):
                row[f"multi_prob_{class_name}"] = float(multi_probs[idx, class_idx])
        if ova_probs is not None:
            row["ova_confidence"] = float(ova_probs[idx, ova_idx])
            row["ova_true_prob"] = float(ova_probs[idx, target])
            for class_idx, class_name in enumerate(class_names):
                row[f"ova_prob_{class_name}"] = float(ova_probs[idx, class_idx])
        if ova_platt_probs is not None:
            row["ova_platt_confidence"] = float(ova_platt_probs[idx, ova_platt_idx])
            row["ova_platt_true_prob"] = float(ova_platt_probs[idx, target])
            for class_idx, class_name in enumerate(class_names):
                row[f"ova_platt_prob_{class_name}"] = float(ova_platt_probs[idx, class_idx])
        if ova_threshold_scores is not None and ova_threshold_raw_probs is not None and ova_thresholds is not None:
            row["ova_threshold_confidence"] = float(ova_threshold_scores[idx, ova_threshold_idx])
            row["ova_threshold_true_score"] = float(ova_threshold_scores[idx, target])
            row["ova_threshold_true_prob"] = float(ova_threshold_raw_probs[idx, target])
            for class_idx, class_name in enumerate(class_names):
                row[f"ova_threshold_{class_name}"] = float(ova_thresholds[class_idx])
                row[f"ova_threshold_score_{class_name}"] = float(ova_threshold_scores[idx, class_idx])
                row[f"ova_threshold_prob_{class_name}"] = float(ova_threshold_raw_probs[idx, class_idx])
        rows.append(row)

    output_path = Path(predictions_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(rows)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        output_df = pd.concat([existing, output_df], ignore_index=True)
    output_df.to_csv(output_path, index=False)
    print(f"[seed={seed}] Test predictions saved to {output_path}", flush=True)


def train_multioutput_regression(experiment_data: ExperimentData, args, device, seed: int):
    print(f"[seed={seed}] Training multi-output regression model", flush=True)
    train_loader, val_loader = create_data_loaders(
        experiment_data.X_train,
        experiment_data.y_train,
        experiment_data.X_val,
        experiment_data.y_val,
        args.batch_size,
        seed,
        torch.float32,
    )
    model = build_experiment_model(experiment_data, args, experiment_data.target_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    start = time.time()
    training_info = fit_experiment_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        "multioutput_regression",
        args,
    )
    return model, time.time() - start, training_info


def train_decoupled_regression(experiment_data: ExperimentData, args, device, seed: int):
    models = []
    training_infos = []
    start = time.time()

    for target_idx in range(experiment_data.target_dim):
        print(
            f"[seed={seed}] Training decoupled regression model "
            f"{target_idx + 1}/{experiment_data.target_dim}",
            flush=True,
        )
        train_loader, val_loader = create_data_loaders(
            experiment_data.X_train,
            experiment_data.y_train[:, target_idx].astype(np.float32),
            experiment_data.X_val,
            experiment_data.y_val[:, target_idx].astype(np.float32),
            args.batch_size,
            seed + target_idx,
            torch.float32,
        )
        model = build_experiment_model(experiment_data, args, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        training_infos.append(
            fit_experiment_model(model, train_loader, val_loader, criterion, optimizer, device, "singleoutput_regression", args)
        )
        models.append(model)

    return models, time.time() - start, summarize_training_infos(training_infos)


def append_shared_metadata(
    results: Dict[str, float],
    args,
    experiment_data: ExperimentData,
    seed: int,
    model_type: str,
    coupling_level: str,
    train_time: float,
    training_info: Dict[str, float],
) -> Dict[str, object]:
    hidden_layers_value = str(args.hidden_layers) if args.model_arch == "mlp" else "n/a"
    vgg_channels_value = str(args.vgg_channels) if args.model_arch == "vgg" else "n/a"
    return {
        "task": experiment_data.task_type,
        "dataset": experiment_data.dataset_name,
        "dependency_strength": experiment_data.dependency_strength,
        "target_dim": experiment_data.target_dim,
        "model_arch": args.model_arch,
        "hidden_layers": hidden_layers_value,
        "vgg_channels": vgg_channels_value,
        "batch_normalization": args.batch_normalization,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "learning_rate": args.learning_rate,
        "class_weighting": getattr(args, "class_weighting", "none"),
        "ova_loss": getattr(args, "ova_loss", "bce"),
        "focal_gamma": getattr(args, "focal_gamma", 2.0),
        "focal_alpha": getattr(args, "focal_alpha", "balanced"),
        "data_augmentation": getattr(args, "data_augmentation", "none"),
        "train_sampler": getattr(args, "train_sampler", "none"),
        "pretrained_finetune": getattr(args, "pretrained_finetune", "frozen"),
        "ova_calibration": getattr(args, "ova_calibration", "none"),
        "seed": seed,
        "model_type": model_type,
        "coupling_level": coupling_level,
        "train_time_seconds": train_time,
        **training_info,
        **results,
    }


def run_classification_experiment(args, experiment_data: ExperimentData, device, seed: int) -> List[Dict[str, object]]:
    results = []
    multi_model = None
    ova_models = None

    if "multi-output" in args.coupling_modes:
        multi_model, multi_time, multi_training_info = train_multiclass_model(experiment_data, args, device, seed)
        results.append(
            append_shared_metadata(
                evaluate_multiclass_model(multi_model, experiment_data, args, device),
                args,
                experiment_data,
                seed,
                "multi-output",
                "coupled_outputs",
                multi_time,
                multi_training_info,
            )
        )

    if "ova" in args.coupling_modes:
        ova_models, ova_time, ova_training_info = train_ova_models(experiment_data, args, device, seed)
        results.append(
            append_shared_metadata(
                evaluate_ova_ensemble(ova_models, experiment_data, args, device),
                args,
                experiment_data,
                seed,
                "OVA",
                "decoupled_outputs",
                ova_time,
                ova_training_info,
            )
        )
        if getattr(args, "ova_calibration", "none") == "platt":
            results.append(
                append_shared_metadata(
                    evaluate_ova_platt_ensemble(ova_models, experiment_data, args, device),
                    args,
                    experiment_data,
                    seed,
                    "OVA calibrated",
                    "decoupled_outputs_platt",
                    ova_time,
                    ova_training_info,
                )
            )
        if getattr(args, "ova_calibration", "none") in {"threshold", "threshold-f1"}:
            results.append(
                append_shared_metadata(
                    evaluate_ova_threshold_ensemble(ova_models, experiment_data, args, device),
                    args,
                    experiment_data,
                    seed,
                    "OVA thresholded",
                    "decoupled_outputs_threshold_f1"
                    if getattr(args, "ova_calibration", "none") == "threshold-f1"
                    else "decoupled_outputs_threshold",
                    ova_time,
                    ova_training_info,
                )
            )

    save_classification_predictions(
        args,
        experiment_data,
        device,
        seed,
        multi_model=multi_model,
        ova_models=ova_models,
    )

    if should_run_ovo(args, experiment_data):
        ovo_models, ovo_time, ovo_training_info = train_ovo_models(experiment_data, args, device, seed)
        results.append(
            append_shared_metadata(
                evaluate_ovo_ensemble(ovo_models, experiment_data, args, device),
                args,
                experiment_data,
                seed,
                "OVO",
                "pairwise_decomposition",
                ovo_time,
                ovo_training_info,
            )
        )

    return results


def run_regression_experiment(args, experiment_data: ExperimentData, device, seed: int) -> List[Dict[str, object]]:
    results = []

    coupled_model, coupled_time, coupled_training_info = train_multioutput_regression(experiment_data, args, device, seed)
    results.append(
        append_shared_metadata(
            evaluate_multioutput_regression(coupled_model, experiment_data, args, device),
            args,
            experiment_data,
            seed,
            "multi-output",
            "coupled_outputs",
            coupled_time,
            coupled_training_info,
        )
    )

    if "decoupled" in args.coupling_modes:
        decoupled_models, decoupled_time, decoupled_training_info = train_decoupled_regression(experiment_data, args, device, seed)
        results.append(
            append_shared_metadata(
                evaluate_decoupled_regression(decoupled_models, experiment_data, args, device),
                args,
                experiment_data,
                seed,
                "decoupled",
                "fully_decoupled_outputs",
                decoupled_time,
                decoupled_training_info,
            )
        )

    return results


def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in [
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "tpr_macro",
            "fpr_macro",
            "tnr_macro",
            "fnr_macro",
            "mse",
            "mae",
            "r2",
            "train_time_seconds",
            "parallel_train_time_seconds",
            "ova_model_train_time_seconds_mean",
            "best_val_loss",
            "epochs_trained",
            "total_epochs_trained",
            "models_stopped_early",
        ]
        if column in results_df.columns
    ]
    grouped = (
        results_df.groupby(
            [
                "task",
                "dataset",
                "dependency_strength",
                "target_dim",
                "model_arch",
                "hidden_layers",
                "vgg_channels",
                "batch_normalization",
                "batch_size",
                "epochs",
                "early_stopping_patience",
                "early_stopping_min_delta",
                "learning_rate",
                "class_weighting",
                "ova_loss",
                "focal_gamma",
                "focal_alpha",
                "data_augmentation",
                "train_sampler",
                "pretrained_finetune",
                "ova_calibration",
                "model_type",
                "coupling_level",
            ],
            as_index=False,
            dropna=False,
        )[metric_columns]
        .mean(numeric_only=True)
    )
    grouped["seed"] = "mean"
    return grouped


def parse_seeds(args) -> List[int]:
    return args.seeds if getattr(args, "seeds", None) else [args.seed]


def normalize_coupling_modes(args) -> None:
    args.coupling_modes = [mode.lower() for mode in args.coupling_modes]


def should_run_ovo(args, experiment_data: ExperimentData) -> bool:
    return "ovo" in args.coupling_modes and experiment_data.target_dim > 2


def validate_args(args) -> None:
    if args.task == "classification":
        valid_datasets = set(STANDARD_CLASSIFICATION_DATASETS.keys()) | {"synthetic_multiclass", "tb_chest_xray"}
        valid_modes = {"multi-output", "ova", "ovo"}
    else:
        valid_datasets = set(STANDARD_REGRESSION_DATASETS.keys()) | {"synthetic_multiregression"}
        valid_modes = {"multi-output", "decoupled"}

    if args.dataset not in valid_datasets:
        raise ValueError(f"Dataset '{args.dataset}' is not valid for task '{args.task}'")

    image_model_datasets = {"mnist", "cifar10", "brisc", "tb_chest_xray", "ham10000"}
    if args.model_arch in {"vgg", "vgg16-pretrained", "vit-b-16-pretrained"} and (
        args.task != "classification" or args.dataset not in image_model_datasets
    ):
        raise ValueError(
            "Image model architectures are only supported for classification with "
            "mnist, cifar10, brisc, tb_chest_xray or ham10000"
        )

    if args.model_arch == "vit-b-16-pretrained" and args.image_size != 224:
        raise ValueError("vit-b-16-pretrained expects --image-size 224")

    if any(channels <= 0 for channels in args.vgg_channels):
        raise ValueError("vgg_channels must be positive")

    if args.image_size <= 0:
        raise ValueError("image_size must be greater than 0")

    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be greater than or equal to 0")

    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be greater than or equal to 0")

    invalid_modes = set(args.coupling_modes) - valid_modes
    if invalid_modes:
        raise ValueError(f"Coupling modes {sorted(invalid_modes)} are not valid for task '{args.task}'")

    if getattr(args, "class_weighting", "none") not in {"none", "balanced"}:
        raise ValueError("class_weighting must be 'none' or 'balanced'")

    if getattr(args, "ova_loss", "bce") not in {"bce", "focal"}:
        raise ValueError("ova_loss must be 'bce' or 'focal'")

    if getattr(args, "train_sampler", "none") not in {"none", "balanced"}:
        raise ValueError("train_sampler must be 'none' or 'balanced'")

    if getattr(args, "ham10000_label_mode", "original") not in {"original", "malignant_binary"}:
        raise ValueError("ham10000_label_mode must be 'original' or 'malignant_binary'")

    if getattr(args, "focal_gamma", 2.0) < 0:
        raise ValueError("focal_gamma must be greater than or equal to 0")

    focal_alpha_value = getattr(args, "focal_alpha", "balanced")
    if focal_alpha_value not in {"balanced", "none"}:
        try:
            focal_alpha_float = float(focal_alpha_value)
        except ValueError as exc:
            raise ValueError("focal_alpha must be 'balanced', 'none' or a numeric value in (0, 1)") from exc
        if focal_alpha_float <= 0.0 or focal_alpha_float >= 1.0:
            raise ValueError("focal_alpha numeric value must be in (0, 1)")

    if getattr(args, "pretrained_finetune", "frozen") not in {"frozen", "block5", "last-block", "full"}:
        raise ValueError("pretrained_finetune must be one of: frozen, block5, last-block, full")

    if getattr(args, "ova_calibration", "none") not in {"none", "platt", "threshold", "threshold-f1"}:
        raise ValueError("ova_calibration must be one of: none, platt, threshold, threshold-f1")


def main(args):
    normalize_coupling_modes(args)
    validate_args(args)
    seeds = parse_seeds(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running task={args.task}, dataset={args.dataset}, "
        f"seeds={seeds}, coupling_modes={args.coupling_modes}, device={device}",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n=== Starting seed {seed} ===", flush=True)
        set_seed(seed)
        experiment_data = load_experiment_data(args, seed)
        print(
            f"[seed={seed}] Loaded dataset={experiment_data.dataset_name}, "
            f"target_dim={experiment_data.target_dim}, "
            f"train={experiment_data.X_train.shape[0]}, "
            f"val={experiment_data.X_val.shape[0]}, "
            f"test={experiment_data.X_test.shape[0]}",
            flush=True,
        )
        if args.task == "classification":
            all_results.extend(run_classification_experiment(args, experiment_data, device, seed))
        else:
            all_results.extend(run_regression_experiment(args, experiment_data, device, seed))
        print(f"=== Finished seed {seed} ===", flush=True)

    results_df = pd.DataFrame(all_results)
    summary_df = aggregate_results(results_df)

    results_df.to_csv(args.output_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    print(results_df.to_string(index=False), flush=True)
    print("\nAverage results across seeds:\n", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    return results_df, summary_df
