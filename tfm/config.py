from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ExperimentData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    task_type: str
    target_dim: int
    class_names: List[str]
    dataset_name: str
    dependency_strength: float
    test_image_paths: Optional[List[str]] = None
