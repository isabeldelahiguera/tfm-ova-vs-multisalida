import itertools
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from .config import ExperimentData
from .data import STANDARD_CLASSIFICATION_DATASETS, STANDARD_REGRESSION_DATASETS, load_experiment_data
from .evaluation import (
    evaluate_decoupled_regression,
    evaluate_multiclass_model,
    evaluate_multioutput_regression,
    evaluate_ova_ensemble,
    evaluate_ovo_ensemble,
)
from .models import build_model
from .training import create_data_loaders, fit_model, set_seed


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
    )
    model = build_model(
        input_dim=experiment_data.X_train.shape[1],
        hidden_layers=args.hidden_layers,
        output_dim=experiment_data.target_dim,
        batch_norm=args.batch_normalization,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    fit_model(model, train_loader, val_loader, criterion, optimizer, device, "multiclass", args.epochs)
    return model, time.time() - start


def train_ova_models(experiment_data: ExperimentData, args, device, seed: int):
    models = []
    start = time.time()

    for class_idx in range(experiment_data.target_dim):
        print(f"[seed={seed}] Training OVA model {class_idx + 1}/{experiment_data.target_dim}", flush=True)
        y_train_binary = (experiment_data.y_train == class_idx).astype(np.float32)
        y_val_binary = (experiment_data.y_val == class_idx).astype(np.float32)
        train_loader, val_loader = create_data_loaders(
            experiment_data.X_train,
            y_train_binary,
            experiment_data.X_val,
            y_val_binary,
            args.batch_size,
            seed + class_idx,
            torch.float32,
        )
        model = build_model(
            input_dim=experiment_data.X_train.shape[1],
            hidden_layers=args.hidden_layers,
            output_dim=1,
            batch_norm=args.batch_normalization,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        fit_model(model, train_loader, val_loader, criterion, optimizer, device, "binary", args.epochs)
        models.append(model)

    return models, time.time() - start


def train_ovo_models(experiment_data: ExperimentData, args, device, seed: int):
    pair_models = {}
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
        )
        model = build_model(
            input_dim=experiment_data.X_train.shape[1],
            hidden_layers=args.hidden_layers,
            output_dim=1,
            batch_norm=args.batch_normalization,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        fit_model(model, train_loader, val_loader, criterion, optimizer, device, "binary", args.epochs)
        pair_models[(class_a, class_b)] = model

    return pair_models, time.time() - start


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
    model = build_model(
        input_dim=experiment_data.X_train.shape[1],
        hidden_layers=args.hidden_layers,
        output_dim=experiment_data.target_dim,
        batch_norm=args.batch_normalization,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    start = time.time()
    fit_model(model, train_loader, val_loader, criterion, optimizer, device, "multioutput_regression", args.epochs)
    return model, time.time() - start


def train_decoupled_regression(experiment_data: ExperimentData, args, device, seed: int):
    models = []
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
        model = build_model(
            input_dim=experiment_data.X_train.shape[1],
            hidden_layers=args.hidden_layers,
            output_dim=1,
            batch_norm=args.batch_normalization,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        fit_model(model, train_loader, val_loader, criterion, optimizer, device, "singleoutput_regression", args.epochs)
        models.append(model)

    return models, time.time() - start


def append_shared_metadata(
    results: Dict[str, float],
    args,
    experiment_data: ExperimentData,
    seed: int,
    model_type: str,
    coupling_level: str,
    train_time: float,
) -> Dict[str, object]:
    return {
        "task": experiment_data.task_type,
        "dataset": experiment_data.dataset_name,
        "dependency_strength": experiment_data.dependency_strength,
        "target_dim": experiment_data.target_dim,
        "hidden_layers": str(args.hidden_layers),
        "batch_normalization": args.batch_normalization,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": seed,
        "model_type": model_type,
        "coupling_level": coupling_level,
        "train_time_seconds": train_time,
        **results,
    }


def run_classification_experiment(args, experiment_data: ExperimentData, device, seed: int) -> List[Dict[str, object]]:
    results = []

    multi_model, multi_time = train_multiclass_model(experiment_data, args, device, seed)
    results.append(
        append_shared_metadata(
            evaluate_multiclass_model(multi_model, experiment_data, args, device),
            args,
            experiment_data,
            seed,
            "multi-output",
            "coupled_outputs",
            multi_time,
        )
    )

    if "ova" in args.coupling_modes:
        ova_models, ova_time = train_ova_models(experiment_data, args, device, seed)
        results.append(
            append_shared_metadata(
                evaluate_ova_ensemble(ova_models, experiment_data, args, device),
                args,
                experiment_data,
                seed,
                "OVA",
                "decoupled_outputs",
                ova_time,
            )
        )

    if should_run_ovo(args, experiment_data):
        ovo_models, ovo_time = train_ovo_models(experiment_data, args, device, seed)
        results.append(
            append_shared_metadata(
                evaluate_ovo_ensemble(ovo_models, experiment_data, args, device),
                args,
                experiment_data,
                seed,
                "OVO",
                "pairwise_decomposition",
                ovo_time,
            )
        )

    return results


def run_regression_experiment(args, experiment_data: ExperimentData, device, seed: int) -> List[Dict[str, object]]:
    results = []

    coupled_model, coupled_time = train_multioutput_regression(experiment_data, args, device, seed)
    results.append(
        append_shared_metadata(
            evaluate_multioutput_regression(coupled_model, experiment_data, args, device),
            args,
            experiment_data,
            seed,
            "multi-output",
            "coupled_outputs",
            coupled_time,
        )
    )

    if "decoupled" in args.coupling_modes:
        decoupled_models, decoupled_time = train_decoupled_regression(experiment_data, args, device, seed)
        results.append(
            append_shared_metadata(
                evaluate_decoupled_regression(decoupled_models, experiment_data, args, device),
                args,
                experiment_data,
                seed,
                "decoupled",
                "fully_decoupled_outputs",
                decoupled_time,
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
                "hidden_layers",
                "batch_normalization",
                "batch_size",
                "epochs",
                "learning_rate",
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
        valid_datasets = set(STANDARD_CLASSIFICATION_DATASETS.keys()) | {"synthetic_multiclass"}
        valid_modes = {"multi-output", "ova", "ovo"}
    else:
        valid_datasets = set(STANDARD_REGRESSION_DATASETS.keys()) | {"synthetic_multiregression"}
        valid_modes = {"multi-output", "decoupled"}

    if args.dataset not in valid_datasets:
        raise ValueError(f"Dataset '{args.dataset}' is not valid for task '{args.task}'")

    invalid_modes = set(args.coupling_modes) - valid_modes
    if invalid_modes:
        raise ValueError(f"Coupling modes {sorted(invalid_modes)} are not valid for task '{args.task}'")


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
