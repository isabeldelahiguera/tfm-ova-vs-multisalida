from __future__ import annotations

import argparse
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
from tfm.data import load_experiment_data  # noqa: E402
from tfm.evaluation import sigmoid_estable  # noqa: E402
from tfm.metrics import classification_metrics  # noqa: E402
from tfm.training import build_test_loader, collect_predictions  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved multi-output and OVA checkpoints on the test split.")
    parser.add_argument("--dataset", choices=["brisc", "tb_chest_xray", "ham10000"], required=True)
    parser.add_argument("--model-arch", choices=["vgg", "vgg16-pretrained", "vit-b-16-pretrained"], default="vgg")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--checkpoint-dir", default="resultados_actualizados/checkpoints")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--vgg-channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    parser.add_argument("--data-augmentation", choices=["none", "ham10000-basic"], default="none")
    parser.add_argument("--train-sampler", choices=["none", "balanced"], default="none")
    parser.add_argument("--pretrained-finetune", choices=["frozen", "block5", "last-block", "full"], default="frozen")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--brisc-root", default="/mnt/homeGPU/imhiguera/data/brisc2025")
    parser.add_argument("--tb-root", default="/mnt/homeGPU/imhiguera/data/tb_chest_xray")
    parser.add_argument("--ham10000-root", default="/mnt/homeGPU/imhiguera/data/ham10000")
    parser.add_argument("--ham10000-test", choices=["internal", "official"], default="internal")
    parser.add_argument("--ham10000-split-csv", default=None)
    parser.add_argument("--ham10000-split-seed", type=int, default=2000)
    parser.add_argument("--ham10000-exclude-classes", type=str, nargs="*", default=[])
    parser.add_argument("--ham10000-label-mode", choices=["original", "malignant_binary"], default="original")
    return parser


def experiment_args(parsed: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        task="classification",
        dataset=parsed.dataset,
        model_arch=parsed.model_arch,
        hidden_layers=[32, 16],
        vgg_channels=parsed.vgg_channels,
        batch_normalization=False,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        early_stopping_patience=parsed.early_stopping_patience,
        early_stopping_min_delta=parsed.early_stopping_min_delta,
        learning_rate=parsed.learning_rate,
        seed=seed,
        seeds=[seed],
        coupling_modes=["multi-output", "ova"],
        class_weighting=parsed.class_weighting,
        ova_loss="bce",
        focal_gamma=2.0,
        focal_alpha="balanced",
        data_augmentation=parsed.data_augmentation,
        train_sampler=parsed.train_sampler,
        pretrained_finetune=parsed.pretrained_finetune,
        ova_calibration="none",
        synthetic_samples=600,
        synthetic_features=20,
        synthetic_classes=4,
        synthetic_targets=3,
        dependency_strength=0.3,
        max_train=parsed.max_train if parsed.max_train and parsed.max_train > 0 else None,
        max_test=parsed.max_test if parsed.max_test and parsed.max_test > 0 else None,
        brisc_root=parsed.brisc_root,
        tb_root=parsed.tb_root,
        ham10000_root=parsed.ham10000_root,
        ham10000_test=parsed.ham10000_test,
        ham10000_split_csv=parsed.ham10000_split_csv,
        ham10000_split_seed=parsed.ham10000_split_seed,
        ham10000_exclude_classes=parsed.ham10000_exclude_classes,
        ham10000_label_mode=parsed.ham10000_label_mode,
        image_size=parsed.image_size,
    )


def parsed_for_seed(parsed: argparse.Namespace, seed: int) -> argparse.Namespace:
    seed_parsed = argparse.Namespace(**vars(parsed))
    seed_parsed.seed = seed
    return seed_parsed


def predict_multi(model, experiment_data, args, device) -> tuple[np.ndarray, np.ndarray]:
    loader = build_test_loader(experiment_data.X_test, experiment_data.y_test, args.batch_size, torch.long)
    y_true, logits = collect_predictions(model, loader, device)
    y_pred = logits.argmax(axis=1).astype(int)
    return y_true.astype(int), y_pred


def predict_ova(models, experiment_data, args, device) -> tuple[np.ndarray, np.ndarray]:
    loader = build_test_loader(experiment_data.X_test, experiment_data.y_test, args.batch_size, torch.long)
    y_true = None
    probabilities = []
    for model in models:
        targets, logits = collect_predictions(model, loader, device)
        if y_true is None:
            y_true = targets.astype(int)
        probabilities.append(sigmoid_estable(logits.squeeze(-1)))
    y_pred = np.column_stack(probabilities).argmax(axis=1).astype(int)
    return y_true, y_pred


def main() -> None:
    parsed = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating checkpoints on device={device}", flush=True)
    rows = []
    for seed in parsed.seeds:
        seed_parsed = parsed_for_seed(parsed, seed)
        args = experiment_args(seed_parsed, seed)
        experiment_data = load_experiment_data(args, seed)
        multi_model, ova_models = load_model_checkpoints(seed_parsed, experiment_data, device, lambda _parsed: args)
        for model_name, predictor in [("Multi", predict_multi), ("OVA", predict_ova)]:
            models = multi_model if model_name == "Multi" else ova_models
            y_true, y_pred = predictor(models, experiment_data, args, device)
            metrics = classification_metrics(y_true, y_pred)
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "dataset_name": experiment_data.dataset_name,
                    "target_dim": experiment_data.target_dim,
                    **metrics,
                }
            )
        print(f"Finished seed {seed}", flush=True)

    df = pd.DataFrame(rows)
    output_path = Path(parsed.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved test metrics to {output_path}", flush=True)
    summary = df.groupby("model")[
        ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]
    ].agg(["mean", "std"])
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"), flush=True)


if __name__ == "__main__":
    main()
