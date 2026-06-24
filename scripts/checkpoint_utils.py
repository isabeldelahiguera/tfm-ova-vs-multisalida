from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tfm.evaluation import sigmoid_estable
from tfm.experiment import build_experiment_model
from tfm.metrics import classification_metrics
from tfm.training import build_test_loader, collect_predictions


def checkpoint_base_dir(parsed: argparse.Namespace) -> Path:
    root = Path(parsed.checkpoint_dir or "resultados_actualizados/checkpoints").expanduser()
    tag = (
        getattr(parsed, "checkpoint_run_tag", "")
        or parsed.run_tag
        or f"{parsed.model_arch}_{parsed.pretrained_finetune}_{parsed.class_weighting}"
    )
    return root / parsed.dataset / f"seed_{parsed.seed}_{tag}"


def checkpoint_paths(parsed: argparse.Namespace, target_dim: int) -> dict[str, Path | list[Path]]:
    base = checkpoint_base_dir(parsed)
    return {
        "base": base,
        "multi": base / "multi_output.pt",
        "ova": [base / f"ova_class_{class_idx}.pt" for class_idx in range(target_dim)],
        "metadata": base / "metadata.json",
    }


def checkpoints_exist(parsed: argparse.Namespace, target_dim: int) -> bool:
    paths = checkpoint_paths(parsed, target_dim)
    return bool(paths["multi"].exists() and all(path.exists() for path in paths["ova"]))


def validation_metrics_multi(model: nn.Module, experiment_data, args, device) -> dict[str, float]:
    loader = build_test_loader(experiment_data.X_val, experiment_data.y_val, args.batch_size, torch.long)
    y_true, logits = collect_predictions(model, loader, device)
    return classification_metrics(y_true, logits.argmax(axis=1))


def validation_metrics_ova(models: list[nn.Module], experiment_data, args, device) -> dict[str, float]:
    loader = build_test_loader(experiment_data.X_val, experiment_data.y_val, args.batch_size, torch.long)
    y_true = None
    probabilities = []
    for model in models:
        batch_targets, logits = collect_predictions(model, loader, device)
        if y_true is None:
            y_true = batch_targets
        probabilities.append(sigmoid_estable(logits.squeeze(-1)))
    y_pred = np.column_stack(probabilities).argmax(axis=1)
    return classification_metrics(y_true, y_pred)


def checkpoint_metadata(parsed: argparse.Namespace, experiment_data, multi_info, ova_info, multi_val, ova_val) -> dict[str, object]:
    return {
        "dataset": parsed.dataset,
        "seed": parsed.seed,
        "run_tag": parsed.run_tag,
        "model_arch": parsed.model_arch,
        "pretrained_finetune": parsed.pretrained_finetune,
        "class_weighting": parsed.class_weighting,
        "data_augmentation": parsed.data_augmentation,
        "train_sampler": parsed.train_sampler,
        "image_size": parsed.image_size,
        "batch_size": parsed.batch_size,
        "learning_rate": parsed.learning_rate,
        "epochs": parsed.epochs,
        "early_stopping_patience": parsed.early_stopping_patience,
        "early_stopping_min_delta": parsed.early_stopping_min_delta,
        "ham10000_exclude_classes": parsed.ham10000_exclude_classes,
        "ham10000_label_mode": parsed.ham10000_label_mode,
        "target_dim": int(experiment_data.target_dim),
        "class_names": list(experiment_data.class_names),
        "train_size": int(len(experiment_data.y_train)),
        "val_size": int(len(experiment_data.y_val)),
        "test_size": int(len(experiment_data.y_test)),
        "multi_training_info": multi_info,
        "ova_training_info": ova_info,
        "multi_val_metrics": multi_val,
        "ova_val_metrics": ova_val,
    }


def save_model_checkpoints(
    parsed: argparse.Namespace,
    experiment_data,
    multi_model: nn.Module,
    ova_models: list[nn.Module],
    multi_info,
    ova_info,
    device,
    args,
) -> None:
    paths = checkpoint_paths(parsed, experiment_data.target_dim)
    paths["base"].mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": multi_model.state_dict()}, paths["multi"])
    for class_idx, model in enumerate(ova_models):
        torch.save({"state_dict": model.state_dict()}, paths["ova"][class_idx])
    multi_val = validation_metrics_multi(multi_model, experiment_data, args, device)
    ova_val = validation_metrics_ova(ova_models, experiment_data, args, device)
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(
            checkpoint_metadata(parsed, experiment_data, multi_info, ova_info, multi_val, ova_val),
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved checkpoints to {paths['base']}", flush=True)


def load_model_checkpoints(parsed: argparse.Namespace, experiment_data, device, experiment_args=None):
    if experiment_args is None:
        from explicabilidad_gradcam_vgg import experiment_args as experiment_args

    paths = checkpoint_paths(parsed, experiment_data.target_dim)
    args = experiment_args(parsed)
    multi_model = build_experiment_model(experiment_data, args, experiment_data.target_dim).to(device)
    multi_payload = torch.load(paths["multi"], map_location=device)
    multi_model.load_state_dict(multi_payload["state_dict"])
    ova_models = []
    for path in paths["ova"]:
        model = build_experiment_model(experiment_data, args, 1).to(device)
        payload = torch.load(path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        ova_models.append(model)
    print(f"Loaded checkpoints from {paths['base']}", flush=True)
    return multi_model, ova_models
