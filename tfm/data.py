from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris, load_linnerud, load_wine
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ExperimentData


STANDARD_CLASSIFICATION_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "digits": load_digits,
    "breast_cancer": load_breast_cancer,
}

STANDARD_REGRESSION_DATASETS = {
    "linnerud": load_linnerud,
}


def load_mnist(max_train: int | None = None, max_test: int | None = None):
    from torchvision.datasets import MNIST

    train_dataset = MNIST(root="./data", train=True, download=True)
    test_dataset = MNIST(root="./data", train=False, download=True)

    X_train = train_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_test = test_dataset.targets.numpy()

    if max_train is not None:
        X_train = X_train[:max_train]
        y_train = y_train[:max_train]

    if max_test is not None:
        X_test = X_test[:max_test]
        y_test = y_test[:max_test]

    return X_train, X_test, y_train, y_test


def load_cifar10(max_train: int | None = None, max_test: int | None = None):
    try:
        X, y = fetch_openml(data_id=40927, as_frame=False, return_X_y=True, data_home="./data/openml")
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] < 60000:
            raise RuntimeError(f"Expected 60000 CIFAR-10 samples from OpenML, got {X.shape[0]}.")
        try:
            y = y.astype(int)
        except (TypeError, ValueError):
            y = pd.Categorical(y).codes

        X_train = X[:50000]
        y_train = y[:50000]
        X_test = X[50000:60000]
        y_test = y[50000:60000]
    except Exception:
        from torchvision.datasets import CIFAR10

        train_dataset = CIFAR10(root="./data", train=True, download=True)
        test_dataset = CIFAR10(root="./data", train=False, download=True)
        X_train = train_dataset.data.reshape(-1, 32 * 32 * 3)
        y_train = np.array(train_dataset.targets)
        X_test = test_dataset.data.reshape(-1, 32 * 32 * 3)
        y_test = np.array(test_dataset.targets)

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    if max_train is not None:
        X_train = X_train[:max_train]
        y_train = y_train[:max_train]

    if max_test is not None:
        X_test = X_test[:max_test]
        y_test = y_test[:max_test]

    return X_train, X_test, y_train, y_test


def load_dermatology():
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=33)
    X = dataset.data.features.apply(pd.to_numeric, errors="coerce").values
    y = dataset.data.targets.iloc[:, 0].astype(int).values - 1
    target_names = [
        "psoriasis",
        "seboreic_dermatitis",
        "lichen_planus",
        "pityriasis_rosea",
        "cronic_dermatitis",
        "pityriasis_rubra_pilaris",
    ]
    return type("Dataset", (), {"data": X, "target": y, "target_names": target_names})


def load_heart_disease():
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.apply(pd.to_numeric, errors="coerce").values
    y = dataset.data.targets.iloc[:, 0].astype(int).values
    target_names = [
        "no_heart_disease",
        "heart_disease_1",
        "heart_disease_2",
        "heart_disease_3",
        "heart_disease_4",
    ]
    return type("Dataset", (), {"data": X, "target": y, "target_names": target_names})


def load_energy():
    data = fetch_openml(name="energy-efficiency", version=1, as_frame=True)
    df = data.frame
    X = df.iloc[:, :-2].values
    y = df.iloc[:, -2:].apply(pd.to_numeric).values.astype(float)
    return type("Dataset", (), {"data": X, "target": y})


STANDARD_CLASSIFICATION_DATASETS["mnist"] = load_mnist
STANDARD_CLASSIFICATION_DATASETS["cifar10"] = load_cifar10
STANDARD_CLASSIFICATION_DATASETS["dermatology"] = load_dermatology
STANDARD_CLASSIFICATION_DATASETS["heart_disease"] = load_heart_disease
STANDARD_REGRESSION_DATASETS["energy"] = load_energy


def split_and_scale_data(X, y, seed: int, task_type: str):
    stratify = y if task_type == "classification" and y.ndim == 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
    stratify_train = y_train if task_type == "classification" and y_train.ndim == 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=seed,
        stratify=stratify_train,
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_synthetic_multiclass(
    seed: int,
    n_samples: int,
    n_features: int,
    n_classes: int,
    dependency_strength: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    shared_direction = rng.normal(size=(n_features,))
    class_specific = rng.normal(size=(n_classes, n_features))
    shared_weights = rng.normal(size=(n_classes,))
    shared_component = X @ shared_direction
    logits = []
    for class_idx in range(n_classes):
        coupled_component = dependency_strength * shared_weights[class_idx] * shared_component
        class_component = X @ class_specific[class_idx]
        logits.append(class_component + coupled_component + rng.normal(scale=0.3, size=n_samples))

    y = np.column_stack(logits).argmax(axis=1)
    class_names = [f"class_{idx}" for idx in range(n_classes)]
    return X, y, class_names


def make_synthetic_multiregression(
    seed: int,
    n_samples: int,
    n_features: int,
    n_targets: int,
    dependency_strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    shared_latent = X @ rng.normal(size=(n_features,))
    target_specific_weights = rng.normal(size=(n_features, n_targets))
    independent_component = X @ target_specific_weights
    coupled_component = dependency_strength * np.column_stack(
        [shared_latent + rng.normal(scale=0.2, size=n_samples) for _ in range(n_targets)]
    )
    y = independent_component + coupled_component + rng.normal(scale=0.1, size=(n_samples, n_targets))
    return X, y


def load_experiment_data(args, seed: int) -> ExperimentData:
    if args.task == "classification":
        if args.dataset == "synthetic_multiclass":
            X, y, class_names = make_synthetic_multiclass(
                seed=seed,
                n_samples=args.synthetic_samples,
                n_features=args.synthetic_features,
                n_classes=args.synthetic_classes,
                dependency_strength=args.dependency_strength,
            )
            X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale_data(X, y, seed, "classification")
        elif args.dataset in {"mnist", "cifar10"}:
            if args.dataset == "mnist":
                X_train, X_test, y_train, y_test = load_mnist()
            else:
                X_train, X_test, y_train, y_test = load_cifar10()
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.15,
                random_state=seed,
                stratify=y_train,
            )
            class_names = [str(i) for i in range(10)]
        else:
            dataset = STANDARD_CLASSIFICATION_DATASETS[args.dataset]()
            X = dataset.data
            y = dataset.target
            class_names = [str(name) for name in dataset.target_names]
            X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale_data(X, y, seed, "classification")

        return ExperimentData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            task_type="classification",
            target_dim=len(np.unique(y_train)),
            class_names=class_names,
            dataset_name=args.dataset,
            dependency_strength=args.dependency_strength if args.dataset == "synthetic_multiclass" else np.nan,
        )

    if args.task == "regression":
        if args.dataset == "synthetic_multiregression":
            X, y = make_synthetic_multiregression(
                seed=seed,
                n_samples=args.synthetic_samples,
                n_features=args.synthetic_features,
                n_targets=args.synthetic_targets,
                dependency_strength=args.dependency_strength,
            )
        else:
            dataset = STANDARD_REGRESSION_DATASETS[args.dataset]()
            X = dataset.data
            y = dataset.target

        X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale_data(X, y, seed, "regression")
        return ExperimentData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            task_type="regression",
            target_dim=y.shape[1],
            class_names=[],
            dataset_name=args.dataset,
            dependency_strength=args.dependency_strength if args.dataset == "synthetic_multiregression" else np.nan,
        )

    raise ValueError(f"Unsupported task: {args.task}")
