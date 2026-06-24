from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
import numpy as np

datasets = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
    "Digits": load_digits,
}

for name, loader in datasets.items():
    data = loader()
    X = data.data

    print(name)
    print("shape:", X.shape)
    print("nan_count:", np.isnan(X).sum())
    print("inf_count:", np.isinf(X).sum())
    print("min:", np.nanmin(X))
    print("max:", np.nanmax(X))
    print()