from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from .data import STANDARD_CLASSIFICATION_DATASETS, load_experiment_data
from .evaluation import sigmoid_estable
from .experiment import (
    aggregate_results,
    append_shared_metadata,
    build_experiment_model,
    fit_experiment_model,
    summarize_training_infos,
)
from .metrics import classification_metrics
from .training import build_test_loader, collect_predictions, create_data_loaders, set_seed


def artifact_dataset_dir(output_dir: str | Path, dataset: str) -> Path:
    return Path(output_dir) / dataset


def class_summary_path(output_dir: str | Path, dataset: str, class_idx: int) -> Path:
    return artifact_dataset_dir(output_dir, dataset) / f"class_{class_idx}_summaries.csv"


def class_predictions_path(output_dir: str | Path, dataset: str, class_idx: int) -> Path:
    return artifact_dataset_dir(output_dir, dataset) / f"class_{class_idx}_predictions.csv"


def artifact_metadata(
    args,
    experiment_data,
    seed: int,
    class_idx: int,
    train_time: float,
    training_info,
) -> Dict[str, object]:
    return {
        "dataset": experiment_data.dataset_name,
        "target_dim": experiment_data.target_dim,
        "model_arch": args.model_arch,
        "hidden_layers": list(args.hidden_layers),
        "vgg_channels": list(args.vgg_channels),
        "seed": seed,
        "ova_class_index": class_idx,
        "train_time_seconds": train_time,
        **training_info,
    }


def train_ova_class_for_seed(args, seed: int):
    if args.task != "classification":
        raise ValueError("Parallel OVA is only defined for classification")

    set_seed(seed)
    experiment_data = load_experiment_data(args, seed)
    if args.ova_class_index < 0 or args.ova_class_index >= experiment_data.target_dim:
        raise ValueError(
            f"OVA class index {args.ova_class_index} is outside [0, {experiment_data.target_dim - 1}]"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_idx = args.ova_class_index
    print(
        f"Training parallel OVA artifact dataset={args.dataset}, seed={seed}, "
        f"class={class_idx}/{experiment_data.target_dim - 1}, device={device}",
        flush=True,
    )

    y_train_binary = (experiment_data.y_train == class_idx).astype(np.float32)
    y_val_binary = (experiment_data.y_val == class_idx).astype(np.float32)
    set_seed(seed)
    train_loader, val_loader = create_data_loaders(
        experiment_data.X_train,
        y_train_binary,
        experiment_data.X_val,
        y_val_binary,
        args.batch_size,
        seed,
        torch.float32,
    )
    model = build_experiment_model(experiment_data, args, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    start = time.time()
    training_info = fit_experiment_model(model, train_loader, val_loader, criterion, optimizer, device, "binary", args)
    train_time = time.time() - start

    test_loader = build_test_loader(experiment_data.X_test, experiment_data.y_test, args.batch_size, torch.long)
    y_true, logits = collect_predictions(model, test_loader, device)
    probabilities = sigmoid_estable(logits.squeeze(-1))

    summary_row = artifact_metadata(args, experiment_data, seed, class_idx, train_time, training_info)
    prediction_rows = pd.DataFrame(
        {
            "seed": seed,
            "test_index": np.arange(len(y_true)),
            "y_true": y_true,
            "probability": probabilities,
        }
    )
    return summary_row, prediction_rows


def train_ova_class(args) -> None:
    seeds = args.seeds if args.seeds else [args.seed]
    class_summaries = []
    class_predictions = []
    for seed in seeds:
        summary_row, prediction_rows = train_ova_class_for_seed(args, seed)
        class_summaries.append(summary_row)
        class_predictions.append(prediction_rows)

    summary_path = class_summary_path(args.artifact_dir, args.dataset, args.ova_class_index)
    predictions_path = class_predictions_path(args.artifact_dir, args.dataset, args.ova_class_index)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(class_summaries).to_csv(summary_path, index=False)
    pd.concat(class_predictions, ignore_index=True).to_csv(predictions_path, index=False)
    print(f"Saved {summary_path}", flush=True)
    print(f"Saved {predictions_path}", flush=True)


def load_artifact(summary_path: Path, predictions_path: Path, seed: int):
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing parallel OVA class summary: {summary_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing parallel OVA class predictions: {predictions_path}")

    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df[summary_df["seed"] == seed]
    if len(summary_df) != 1:
        raise ValueError(f"Expected one metadata row for seed {seed} in {summary_path}")
    predictions_df = pd.read_csv(predictions_path)
    predictions_df = predictions_df[predictions_df["seed"] == seed].sort_values("test_index")
    metadata = summary_df.iloc[0].to_dict()
    y_true = predictions_df["y_true"].to_numpy()
    probabilities = predictions_df["probability"].to_numpy()
    return y_true, probabilities, metadata


def check_metadata(expected: Dict[str, object] | None, current: Dict[str, object], path: Path) -> Dict[str, object]:
    if expected is None:
        return current

    checked_fields = [
        "dataset",
        "target_dim",
        "model_arch",
        "hidden_layers",
        "vgg_channels",
        "seed",
    ]
    mismatches = [field for field in checked_fields if expected[field] != current[field]]
    if mismatches:
        raise ValueError(f"Artifact metadata mismatch in {path}: {mismatches}")
    return expected


def aggregate_seed(args, seed: int) -> Dict[str, object]:
    set_seed(seed)
    experiment_data = load_experiment_data(args, seed)
    y_true = None
    probabilities = []
    training_infos = []
    class_times = []
    reference_metadata = None

    for class_idx in range(experiment_data.target_dim):
        summary_path = class_summary_path(args.artifact_dir, args.dataset, class_idx)
        predictions_path = class_predictions_path(args.artifact_dir, args.dataset, class_idx)
        class_y_true, class_probabilities, metadata = load_artifact(summary_path, predictions_path, seed)
        reference_metadata = check_metadata(reference_metadata, metadata, summary_path)
        if y_true is None:
            y_true = class_y_true
        elif not np.array_equal(y_true, class_y_true):
            raise ValueError(f"OVA artifacts disagree on test labels for seed {seed}")

        probabilities.append(class_probabilities)
        class_times.append(float(metadata["train_time_seconds"]))
        training_infos.append(
            {
                "best_val_loss": metadata["best_val_loss"],
                "epochs_trained": metadata["epochs_trained"],
                "stopped_early": metadata["stopped_early"],
            }
        )

    y_pred = np.column_stack(probabilities).argmax(axis=1)
    training_info = summarize_training_infos(training_infos)
    result = append_shared_metadata(
        classification_metrics(y_true, y_pred),
        args,
        experiment_data,
        seed,
        "OVA",
        "decoupled_outputs",
        float(np.sum(class_times)),
        training_info,
    )
    result["parallel_train_time_seconds"] = float(np.max(class_times))
    result["ova_model_train_time_seconds_mean"] = float(np.mean(class_times))
    return result


def aggregate_ova_artifacts(args) -> None:
    seeds = args.seeds if args.seeds else [args.seed]
    results = [aggregate_seed(args, seed) for seed in seeds]
    results_df = pd.DataFrame(results)
    summary_df = aggregate_results(results_df)
    class_summaries = []
    experiment_data = load_experiment_data(args, seeds[0])
    for class_idx in range(experiment_data.target_dim):
        path = class_summary_path(args.artifact_dir, args.dataset, class_idx)
        class_summaries.append(pd.read_csv(path))
    class_summary_df = pd.concat(class_summaries, ignore_index=True)

    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    class_times_csv = output_csv.with_name(f"{output_csv.stem}_class_times.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    class_summary_df.to_csv(class_times_csv, index=False)
    shutil.rmtree(artifact_dataset_dir(args.artifact_dir, args.dataset))
    print(results_df.to_string(index=False), flush=True)
    print(f"\nPer-class times saved to {class_times_csv}\n", flush=True)
    print("\nAverage results across seeds:\n", flush=True)
    print(summary_df.to_string(index=False), flush=True)


def add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=["classification"], default="classification")
    parallel_datasets = sorted(set(STANDARD_CLASSIFICATION_DATASETS) | {"tb_chest_xray"})
    parser.add_argument("--dataset", choices=parallel_datasets, required=True)
    parser.add_argument("--model-arch", choices=["mlp", "vgg", "vgg16-pretrained"], default="mlp")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--vgg-channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--batch-normalization", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--synthetic-samples", type=int, default=600)
    parser.add_argument("--synthetic-features", type=int, default=20)
    parser.add_argument("--synthetic-classes", type=int, default=4)
    parser.add_argument("--dependency-strength", type=float, default=0.3)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--brisc-root", type=str, default="./data/brisc2025")
    parser.add_argument("--tb-root", type=str, default="./data/tb_chest_xray")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--artifact-dir", type=str, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and aggregate class-parallel OVA artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-class", help="Train one OVA classifier and save its test scores")
    add_experiment_arguments(train_parser)
    train_parser.add_argument("--seeds", type=int, nargs="+", default=None)
    train_parser.add_argument("--ova-class-index", type=int, required=True)
    train_parser.set_defaults(func=train_ova_class)

    aggregate_parser = subparsers.add_parser("aggregate", help="Rebuild OVA ensembles from class artifacts")
    add_experiment_arguments(aggregate_parser)
    aggregate_parser.add_argument("--seeds", type=int, nargs="+", default=None)
    aggregate_parser.add_argument("--output-csv", type=str, required=True)
    aggregate_parser.add_argument("--summary-csv", type=str, required=True)
    aggregate_parser.set_defaults(func=aggregate_ova_artifacts)
    return parser


def cli() -> None:
    args = build_parser().parse_args()
    args.func(args)
