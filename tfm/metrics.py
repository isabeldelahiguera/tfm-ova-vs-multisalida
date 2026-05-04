from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def macro_confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    labels = np.union1d(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = matrix.sum()

    tpr_values = []
    fpr_values = []
    tnr_values = []
    fnr_values = []

    for idx in range(len(labels)):
        tp = matrix[idx, idx]
        fn = matrix[idx, :].sum() - tp
        fp = matrix[:, idx].sum() - tp
        tn = total - tp - fn - fp

        tpr_values.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        fpr_values.append(fp / (fp + tn) if fp + tn > 0 else 0.0)
        tnr_values.append(tn / (tn + fp) if tn + fp > 0 else 0.0)
        fnr_values.append(fn / (fn + tp) if fn + tp > 0 else 0.0)

    return {
        "tpr_macro": float(np.mean(tpr_values)),
        "fpr_macro": float(np.mean(fpr_values)),
        "tnr_macro": float(np.mean(tnr_values)),
        "fnr_macro": float(np.mean(fnr_values)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        **macro_confusion_rates(y_true, y_pred),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred, multioutput="uniform_average"),
    }
