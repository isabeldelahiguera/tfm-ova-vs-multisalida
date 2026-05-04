from typing import Dict

import numpy as np
import torch

from .config import ExperimentData
from .metrics import classification_metrics, regression_metrics
from .training import build_test_loader, collect_predictions


def evaluate_multiclass_model(model, experiment_data: ExperimentData, args, device) -> Dict[str, float]:
    test_loader = build_test_loader(experiment_data.X_test, experiment_data.y_test, args.batch_size, torch.long)
    y_true, logits = collect_predictions(model, test_loader, device)
    y_pred = logits.argmax(axis=1)
    return classification_metrics(y_true, y_pred)


def evaluate_ova_ensemble(models, experiment_data: ExperimentData, args, device) -> Dict[str, float]:
    test_loader = build_test_loader(experiment_data.X_test, experiment_data.y_test, args.batch_size, torch.long)
    y_true = None
    probabilities = []

    for model in models:
        batch_targets, logits = collect_predictions(model, test_loader, device)
        if y_true is None:
            y_true = batch_targets
        probabilities.append(1.0 / (1.0 + np.exp(-logits.squeeze(-1))))

    y_pred = np.column_stack(probabilities).argmax(axis=1)
    return classification_metrics(y_true, y_pred)


def evaluate_ovo_ensemble(pair_models, experiment_data: ExperimentData, args, device) -> Dict[str, float]:
    test_loader = build_test_loader(experiment_data.X_test, experiment_data.y_test, args.batch_size, torch.long)
    y_true = None
    vote_matrix = np.zeros((experiment_data.X_test.shape[0], experiment_data.target_dim), dtype=np.float32)

    for (class_a, class_b), model in pair_models.items():
        batch_targets, logits = collect_predictions(model, test_loader, device)
        if y_true is None:
            y_true = batch_targets

        probabilities_b = 1.0 / (1.0 + np.exp(-logits.squeeze(-1)))
        vote_matrix[:, class_a] += 1.0 - probabilities_b
        vote_matrix[:, class_b] += probabilities_b

    y_pred = vote_matrix.argmax(axis=1)
    return classification_metrics(y_true, y_pred)


def evaluate_multioutput_regression(model, experiment_data: ExperimentData, args, device) -> Dict[str, float]:
    test_loader = build_test_loader(experiment_data.X_test, experiment_data.y_test, args.batch_size, torch.float32)
    y_true, y_pred = collect_predictions(model, test_loader, device)
    return regression_metrics(y_true, y_pred)


def evaluate_decoupled_regression(models, experiment_data: ExperimentData, args, device) -> Dict[str, float]:
    test_loader = build_test_loader(experiment_data.X_test, experiment_data.y_test, args.batch_size, torch.float32)
    y_true = None
    predictions = []

    for model in models:
        batch_targets, outputs = collect_predictions(model, test_loader, device)
        if y_true is None:
            y_true = batch_targets
        predictions.append(outputs.squeeze(-1))

    y_pred = np.column_stack(predictions)
    return regression_metrics(y_true, y_pred)

