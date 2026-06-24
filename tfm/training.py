from __future__ import annotations

import random
import copy
from typing import Dict, Tuple

import numpy as np
import sklearn
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sklearn.utils.check_random_state(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor_dataset(X: np.ndarray, y: np.ndarray, y_dtype) -> TensorDataset:
    return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=y_dtype))


class AugmentedTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, y_dtype, transform):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=y_dtype)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        image = self.X[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.y[idx]


def build_train_transform(data_augmentation: str, X_train: np.ndarray):
    if data_augmentation == "none":
        return None
    if X_train.ndim != 4:
        raise ValueError("Data augmentation is only supported for image tensors with shape (N, C, H, W)")
    if data_augmentation == "ham10000-basic":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=20,
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
            ]
        )
    raise ValueError(f"Unknown data augmentation mode: {data_augmentation}")


def build_train_sampler(train_sampler: str, labels: np.ndarray, seed: int):
    if train_sampler == "none":
        return None
    if train_sampler != "balanced":
        raise ValueError(f"Unknown train sampler mode: {train_sampler}")

    sample_weights = compute_sample_weight(class_weight="balanced", y=np.asarray(labels).astype(int))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    seed: int,
    y_dtype,
    data_augmentation: str = "none",
    train_sampler: str = "none",
    sampler_labels: np.ndarray | None = None,
) -> Tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    train_transform = build_train_transform(data_augmentation, X_train)
    train_dataset = (
        AugmentedTensorDataset(X_train, y_train, y_dtype, train_transform)
        if train_transform is not None
        else to_tensor_dataset(X_train, y_train, y_dtype)
    )
    sampler = build_train_sampler(
        train_sampler,
        y_train if sampler_labels is None else sampler_labels,
        seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=None if sampler is not None else generator,
    )
    val_loader = DataLoader(
        to_tensor_dataset(X_val, y_val, y_dtype),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def build_test_loader(X_test: np.ndarray, y_test: np.ndarray, batch_size: int, y_dtype) -> DataLoader:
    return DataLoader(to_tensor_dataset(X_test, y_test, y_dtype), batch_size=batch_size, shuffle=False)


def run_epoch(model, data_loader, criterion, optimizer, device, problem_mode: str) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(inputs)
            if problem_mode == "multiclass":
                loss = criterion(outputs, targets.long())
            elif problem_mode == "binary":
                loss = criterion(outputs.squeeze(-1), targets.float())
            elif problem_mode == "multioutput_regression":
                loss = criterion(outputs, targets.float())
            elif problem_mode == "singleoutput_regression":
                loss = criterion(outputs.squeeze(-1), targets.float())
            else:
                raise ValueError(f"Unknown problem_mode: {problem_mode}")

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    problem_mode: str,
    epochs: int,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
) -> Dict[str, float]:
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for _ in range(epochs):
        run_epoch(model, train_loader, criterion, optimizer, device, problem_mode)
        val_loss = run_epoch(model, val_loader, criterion, None, device, problem_mode)
        epochs_trained += 1

        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    stopped_early = early_stopping_patience > 0 and epochs_trained < epochs
    return {
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
        "total_epochs_trained": epochs_trained,
        "stopped_early": stopped_early,
        "models_stopped_early": int(stopped_early),
    }


def collect_predictions(model, data_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets_all = []
    outputs_all = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs_all.append(outputs.cpu().numpy())
            targets_all.append(targets.numpy())

    return np.concatenate(targets_all, axis=0), np.concatenate(outputs_all, axis=0)
