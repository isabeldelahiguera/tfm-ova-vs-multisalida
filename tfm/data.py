from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
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


HAM10000_CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM10000_BINARY_CLASS_NAMES = ["non_malignant", "malignant"]
# Group used for the nested HAM10000 experiment. "akiec" is clinically
# relevant as a premalignant/malignant-like class in this binary screening view.
HAM10000_MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}


def apply_ham10000_label_mode(rows: pd.DataFrame, label_mode: str) -> pd.DataFrame:
    """Return HAM10000 rows with labels recoded according to the requested task."""
    if label_mode == "original":
        return rows
    rows = rows.copy()
    if "dx_original" not in rows.columns:
        rows["dx_original"] = rows["dx"]
    rows["dx"] = np.where(rows["dx_original"].isin(HAM10000_MALIGNANT_CLASSES), "malignant", "non_malignant")
    return rows


def load_mnist(max_train: int | None = None, max_test: int | None = None, flatten: bool = True):
    from torchvision.datasets import MNIST

    train_dataset = MNIST(root="./data", train=True, download=True)
    test_dataset = MNIST(root="./data", train=False, download=True)

    X_train = train_dataset.data.numpy().astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy()

    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)
    else:
        X_train = X_train[:, None, :, :]
        X_test = X_test[:, None, :, :]

    if max_train is not None:
        X_train = X_train[:max_train]
        y_train = y_train[:max_train]

    if max_test is not None:
        X_test = X_test[:max_test]
        y_test = y_test[:max_test]

    return X_train, X_test, y_train, y_test


def load_cifar10(max_train: int | None = None, max_test: int | None = None, flatten: bool = True):
    from torchvision.datasets import CIFAR10

    train_dataset = CIFAR10(root="./data", train=True, download=True)
    test_dataset = CIFAR10(root="./data", train=False, download=True)
    X_train = train_dataset.data
    y_train = np.array(train_dataset.targets)
    X_test = test_dataset.data
    y_test = np.array(test_dataset.targets)

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    if flatten:
        X_train = X_train.reshape(-1, 32 * 32 * 3)
        X_test = X_test.reshape(-1, 32 * 32 * 3)
    else:
        X_train = X_train.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
        X_test = X_test.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)

    if max_train is not None:
        X_train = X_train[:max_train]
        y_train = y_train[:max_train]

    if max_test is not None:
        X_test = X_test[:max_test]
        y_test = y_test[:max_test]

    return X_train, X_test, y_train, y_test


def load_brisc_split(
    split_dir: Path,
    class_names: List[str],
    image_size: int,
    max_samples: int | None,
    flatten: bool,
):
    images = []
    labels = []
    per_class_limit = None
    extra_samples = 0
    if max_samples is not None:
        per_class_limit, extra_samples = divmod(max_samples, len(class_names))

    for class_idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected BRISC class directory not found: {class_dir}")
        image_paths = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if per_class_limit is not None:
            class_limit = per_class_limit + int(class_idx < extra_samples)
            image_paths = image_paths[:class_limit]
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image = image.convert("L").resize((image_size, image_size), Image.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
            images.append(array)
            labels.append(class_idx)

    X = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    if flatten:
        X = X.reshape(X.shape[0], image_size * image_size)
    else:
        X = X[:, None, :, :]

    return X, y


def load_brisc(
    brisc_root: str | Path = "./data/brisc2025",
    image_size: int = 128,
    max_train: int | None = None,
    max_test: int | None = None,
    flatten: bool = True,
):
    root = Path(brisc_root).expanduser()
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]

    if not (root / "train").exists() or not (root / "test").exists():
        raise FileNotFoundError(f"Expected BRISC train/test directories under: {root}")

    X_train, y_train = load_brisc_split(
        root / "train",
        class_names,
        image_size,
        max_train,
        flatten,
    )
    X_test, y_test = load_brisc_split(
        root / "test",
        class_names,
        image_size,
        max_test,
        flatten,
    )
    return X_train, X_test, y_train, y_test, class_names


def load_tb_chest_xray_images(
    root: str | Path,
    image_size: int,
    flatten: bool,
):
    dataset_root = Path(root).expanduser()
    class_names = ["Normal", "Tuberculosis"]
    images = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected TB class directory not found: {class_dir}")
        image_paths = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image = image.convert("L").resize((image_size, image_size), Image.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
            images.append(array)
            labels.append(class_idx)

    X = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)

    if flatten:
        X = X.reshape(X.shape[0], image_size * image_size)
    else:
        X = X[:, None, :, :]

    return X, y, ["normal", "tuberculosis"]


def find_ham10000_metadata(root: Path) -> Path:
    candidates = [
        root / "HAM10000_metadata",
        root / "HAM10000_metadata.csv",
        root / "HAM10000_metadata.tab",
        root / "raw" / "HAM10000_metadata",
        root / "raw" / "HAM10000_metadata.csv",
        root / "raw" / "HAM10000_metadata.tab",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Expected HAM10000 metadata file under: {root}")


def find_ham10000_official_test_ground_truth(root: Path) -> Path:
    candidates = [
        root / "ISIC2018_Task3_Test_GroundTruth.csv",
        root / "ISIC2018_Task3_Test_GroundTruth.tab",
        root / "raw" / "ISIC2018_Task3_Test_GroundTruth.csv",
        root / "raw" / "ISIC2018_Task3_Test_GroundTruth.tab",
        root / "isic2018_task3_test" / "ISIC2018_Task3_Test_GroundTruth.csv",
        root / "isic2018_task3_test" / "ISIC2018_Task3_Test_GroundTruth.tab",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Expected ISIC2018 Task 3 test ground truth under: {root}")


def ham10000_image_index(root: Path, image_roots: list[Path] | None = None) -> dict[str, Path]:
    if image_roots is None:
        image_roots = [root / "images", root / "raw"]
    image_paths: dict[str, Path] = {}
    for image_root in image_roots:
        if not image_root.exists():
            continue
        for path in image_root.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths[path.stem] = path
    if not image_paths:
        raise FileNotFoundError(
            f"No HAM10000 images found. Extract HAM10000_images_part_*.zip into {root / 'images'}"
        )
    return image_paths


def load_image_rows(
    rows: pd.DataFrame,
    image_paths: dict[str, Path],
    image_size: int,
    flatten: bool,
    class_names: list[str] = HAM10000_CLASS_NAMES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    images = []
    labels = []
    paths = []
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    for row in rows.itertuples(index=False):
        image_id = str(row.image_id)
        image_path = image_paths.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Image for HAM10000 image_id={image_id} not found")
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(array)
        labels.append(class_to_idx[str(row.dx)])
        paths.append(str(image_path))

    X = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    if flatten:
        X = X.reshape(X.shape[0], image_size * image_size * 3)
    else:
        X = X.transpose(0, 3, 1, 2)
    return X, y, paths


def limit_rows(rows: pd.DataFrame, max_samples: int | None, class_names: list[str] = HAM10000_CLASS_NAMES) -> pd.DataFrame:
    if max_samples is None:
        return rows
    if max_samples <= 0:
        return rows.head(0).copy()
    per_class_limit, extra_samples = divmod(max_samples, len(class_names))
    limited_rows = []
    for class_idx, class_name in enumerate(class_names):
        class_limit = per_class_limit + int(class_idx < extra_samples)
        if class_limit <= 0:
            continue
        limited_rows.append(rows[rows["dx"] == class_name].head(class_limit))
    if not limited_rows:
        return rows.head(0).copy()
    return pd.concat(limited_rows, ignore_index=True)


def default_ham10000_split_csv(root: Path, split_seed: int) -> Path:
    return root / f"ham10000_train_test_split_seed{split_seed}.csv"


def create_ham10000_internal_split(metadata: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    lesion_labels = metadata.groupby("lesion_id", as_index=False)["dx"].first()
    train_lesions, test_lesions = train_test_split(
        lesion_labels,
        test_size=0.15,
        random_state=split_seed,
        stratify=lesion_labels["dx"],
    )

    split_by_lesion = {}
    for split_name, split_lesions in [
        ("train", train_lesions),
        ("test", test_lesions),
    ]:
        split_by_lesion.update({lesion_id: split_name for lesion_id in split_lesions["lesion_id"]})

    split_df = metadata.copy()
    split_df["split"] = split_df["lesion_id"].map(split_by_lesion)
    return split_df


def load_or_create_ham10000_internal_split(
    metadata: pd.DataFrame,
    root: Path,
    split_csv: str | Path | None,
    split_seed: int,
) -> pd.DataFrame:
    split_path = Path(split_csv).expanduser() if split_csv else default_ham10000_split_csv(root, split_seed)
    if split_path.exists():
        split_df = pd.read_csv(split_path)
    else:
        split_df = create_ham10000_internal_split(metadata, split_seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(split_path, index=False)
        print(f"HAM10000 internal split saved to {split_path}", flush=True)

    required_columns = {"lesion_id", "image_id", "dx", "split"}
    missing_columns = required_columns - set(split_df.columns)
    if missing_columns:
        raise ValueError(f"HAM10000 split CSV is missing columns: {sorted(missing_columns)}")

    invalid_splits = set(split_df["split"].dropna().unique()) - {"train", "test"}
    if invalid_splits:
        raise ValueError(f"HAM10000 split CSV has invalid split values: {sorted(invalid_splits)}")

    duplicated_images = split_df["image_id"].duplicated()
    if duplicated_images.any():
        raise ValueError("HAM10000 split CSV contains duplicated image_id values")

    lesion_split_counts = split_df.groupby("lesion_id")["split"].nunique()
    leaked_lesions = lesion_split_counts[lesion_split_counts > 1]
    if not leaked_lesions.empty:
        raise ValueError(
            "HAM10000 split CSV assigns the same lesion_id to multiple splits: "
            f"{leaked_lesions.index[:5].tolist()}"
        )

    return split_df.sort_values(["split", "lesion_id", "image_id"]).reset_index(drop=True)


def split_ham10000_train_val(train_pool: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesion_labels = train_pool.groupby("lesion_id", as_index=False)["dx"].first()
    train_lesions, val_lesions = train_test_split(
        lesion_labels,
        test_size=0.17647058823529413,
        random_state=seed,
        stratify=lesion_labels["dx"],
    )
    train_lesion_ids = set(train_lesions["lesion_id"])
    val_lesion_ids = set(val_lesions["lesion_id"])
    train_rows = train_pool[train_pool["lesion_id"].isin(train_lesion_ids)].copy()
    val_rows = train_pool[train_pool["lesion_id"].isin(val_lesion_ids)].copy()
    return train_rows, val_rows


def load_ham10000(
    ham10000_root: str | Path = "./data/ham10000",
    image_size: int = 128,
    max_train: int | None = None,
    max_test: int | None = None,
    flatten: bool = True,
    seed: int = 2000,
    test_split: str = "internal",
    split_csv: str | Path | None = None,
    split_seed: int = 2000,
    exclude_classes: list[str] | None = None,
    label_mode: str = "original",
):
    root = Path(ham10000_root).expanduser()
    if test_split not in {"internal", "official"}:
        raise ValueError("test_split must be 'internal' or 'official'")
    if label_mode not in {"original", "malignant_binary"}:
        raise ValueError("label_mode must be 'original' or 'malignant_binary'")

    metadata_path = find_ham10000_metadata(root)
    metadata = pd.read_csv(metadata_path)
    required_columns = {"lesion_id", "image_id", "dx"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"HAM10000 metadata is missing columns: {sorted(missing_columns)}")

    exclude_classes = exclude_classes or []
    invalid_excluded = set(exclude_classes) - set(HAM10000_CLASS_NAMES)
    if invalid_excluded:
        raise ValueError(f"Unknown HAM10000 classes to exclude: {sorted(invalid_excluded)}")
    active_class_names = [class_name for class_name in HAM10000_CLASS_NAMES if class_name not in set(exclude_classes)]
    if len(active_class_names) < 2:
        raise ValueError("HAM10000 needs at least two active classes after exclusions")

    metadata = metadata[metadata["dx"].isin(active_class_names)].copy()
    if label_mode == "malignant_binary":
        metadata = apply_ham10000_label_mode(metadata, label_mode)
        active_class_names = HAM10000_BINARY_CLASS_NAMES
    metadata = metadata.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)

    if test_split == "internal":
        metadata = load_or_create_ham10000_internal_split(metadata, root, split_csv, split_seed)
        metadata = apply_ham10000_label_mode(metadata, label_mode)
        metadata = metadata[metadata["dx"].isin(active_class_names)].copy()
        train_pool_rows = metadata[metadata["split"] == "train"]
        train_rows_raw, val_rows = split_ham10000_train_val(train_pool_rows, seed)
        test_rows_raw = metadata[metadata["split"] == "test"]
    else:
        lesion_labels = metadata.groupby("lesion_id", as_index=False)["dx"].first()
        train_val_lesions = lesion_labels
        test_lesions = lesion_labels.head(0)

        train_lesions, val_lesions = train_test_split(
            train_val_lesions,
            test_size=0.15,
            random_state=seed,
            stratify=train_val_lesions["dx"],
        )
        split_by_lesion = {}
        for split_name, split_lesions in [
            ("train", train_lesions),
            ("val", val_lesions),
            ("test", test_lesions),
        ]:
            split_by_lesion.update({lesion_id: split_name for lesion_id in split_lesions["lesion_id"]})
        metadata["split"] = metadata["lesion_id"].map(split_by_lesion)
        train_rows_raw = metadata[metadata["split"] == "train"]
        val_rows = metadata[metadata["split"] == "val"]
        test_rows_raw = None

    train_rows = limit_rows(train_rows_raw, max_train, active_class_names)

    train_image_paths = ham10000_image_index(root, [root / "images", root / "raw"])
    X_train, y_train, train_paths = load_image_rows(train_rows, train_image_paths, image_size, flatten, active_class_names)
    X_val, y_val, val_paths = load_image_rows(val_rows, train_image_paths, image_size, flatten, active_class_names)

    if test_split == "internal":
        test_rows = limit_rows(test_rows_raw, max_test, active_class_names)
        test_image_paths_index = train_image_paths
    else:
        test_ground_truth = pd.read_csv(find_ham10000_official_test_ground_truth(root))
        test_ground_truth = apply_ham10000_label_mode(test_ground_truth, label_mode)
        test_rows = limit_rows(test_ground_truth[test_ground_truth["dx"].isin(active_class_names)].copy(), max_test, active_class_names)
        test_image_paths_index = ham10000_image_index(
            root,
            [
                root / "isic2018_task3_test" / "ISIC2018_Task3_Test_Images",
                root / "isic2018_task3_test",
                root / "official_test" / "images",
            ],
        )

    X_test, y_test, test_paths = load_image_rows(test_rows, test_image_paths_index, image_size, flatten, active_class_names)

    return X_train, X_val, X_test, y_train, y_val, y_test, active_class_names, train_paths, val_paths, test_paths


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
STANDARD_CLASSIFICATION_DATASETS["brisc"] = load_brisc
STANDARD_CLASSIFICATION_DATASETS["ham10000"] = load_ham10000
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
        test_image_paths: list[str] | None = None
        if args.dataset == "synthetic_multiclass":
            X, y, class_names = make_synthetic_multiclass(
                seed=seed,
                n_samples=args.synthetic_samples,
                n_features=args.synthetic_features,
                n_classes=args.synthetic_classes,
                dependency_strength=args.dependency_strength,
            )
            X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale_data(X, y, seed, "classification")
        elif args.dataset in {"mnist", "cifar10", "brisc", "tb_chest_xray", "ham10000"}:
            flatten_images = getattr(args, "model_arch", "mlp") == "mlp"
            max_train = getattr(args, "max_train", None)
            max_test = getattr(args, "max_test", None)
            if args.dataset == "mnist":
                X_train, X_test, y_train, y_test = load_mnist(max_train=max_train, max_test=max_test, flatten=flatten_images)
                class_names = [str(i) for i in range(10)]
            elif args.dataset == "cifar10":
                X_train, X_test, y_train, y_test = load_cifar10(max_train=max_train, max_test=max_test, flatten=flatten_images)
                class_names = [str(i) for i in range(10)]
            elif args.dataset == "brisc":
                X_train, X_test, y_train, y_test, class_names = load_brisc(
                    brisc_root=getattr(args, "brisc_root", "./data/brisc2025"),
                    image_size=getattr(args, "image_size", 128),
                    max_train=max_train,
                    max_test=max_test,
                    flatten=flatten_images,
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.15,
                    random_state=seed,
                    stratify=y_train,
                )
            elif args.dataset == "tb_chest_xray":
                X, y, class_names = load_tb_chest_xray_images(
                    root=getattr(args, "tb_root", "./data/tb_chest_xray"),
                    image_size=getattr(args, "image_size", 128),
                    flatten=flatten_images,
                )
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.15,
                    random_state=seed,
                    stratify=y,
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val,
                    y_train_val,
                    test_size=0.17647058823529413,
                    random_state=seed,
                    stratify=y_train_val,
                )
            else:
                (
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    class_names,
                    _train_paths,
                    _val_paths,
                    test_image_paths,
                ) = load_ham10000(
                    ham10000_root=getattr(args, "ham10000_root", "./data/ham10000"),
                    image_size=getattr(args, "image_size", 128),
                    max_train=max_train,
                    max_test=max_test,
                    flatten=flatten_images,
                    seed=seed,
                    test_split=getattr(args, "ham10000_test", "internal"),
                    split_csv=getattr(args, "ham10000_split_csv", None),
                    split_seed=getattr(args, "ham10000_split_seed", 2000),
                    exclude_classes=getattr(args, "ham10000_exclude_classes", []),
                    label_mode=getattr(args, "ham10000_label_mode", "original"),
                )
            if args.dataset in {"mnist", "cifar10"}:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.15,
                    random_state=seed,
                    stratify=y_train,
                )
        else:
            dataset = STANDARD_CLASSIFICATION_DATASETS[args.dataset]()
            X = dataset.data
            y = dataset.target
            class_names = [str(name) for name in dataset.target_names]
            X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale_data(X, y, seed, "classification")

        dataset_name = args.dataset
        if args.dataset == "ham10000" and getattr(args, "ham10000_exclude_classes", []):
            excluded_suffix = "_".join(getattr(args, "ham10000_exclude_classes", []))
            dataset_name = f"ham10000_without_{excluded_suffix}"
        if args.dataset == "ham10000" and getattr(args, "ham10000_label_mode", "original") != "original":
            dataset_name = f"{dataset_name}_{getattr(args, 'ham10000_label_mode')}"

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
            dataset_name=dataset_name,
            dependency_strength=args.dependency_strength if args.dataset == "synthetic_multiclass" else np.nan,
            test_image_paths=test_image_paths,
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
