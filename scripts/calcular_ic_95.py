from pathlib import Path

import numpy as np
import pandas as pd


CSV_FILES = [
    "resultados_actualizados/secuencial/exp_iris_mlp_138320.csv",
    "resultados_actualizados/secuencial/exp_wine_mlp_138320.csv",
    "resultados_actualizados/secuencial/exp_breast_cancer_mlp_138320.csv",
    "resultados_actualizados/secuencial/exp_digits_mlp_138320.csv",
    "resultados_actualizados/secuencial/exp_mnist_vgg_138333.csv",
    "resultados_actualizados/secuencial/exp_cifar10_vgg_138334.csv",
    "resultados_actualizados/secuencial/exp_brisc_vgg_128_138367.csv",
    "resultados_actualizados/secuencial/exp_tb_chest_xray_vgg_128_138338.csv",
]


def main():
    rng = np.random.default_rng(12345)

    for csv_file in CSV_FILES:
        if not Path(csv_file).exists():
            print(f"Skipping missing CSV: {csv_file}")
            continue
        df = pd.read_csv(csv_file)
        dataset = df["dataset"].iloc[0]

        wide = df.pivot_table(index="seed", columns="model_type", values="f1_macro")
        diffs = wide["OVA"] - wide["multi-output"]

        bootstrap_means = []
        for _ in range(10000):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            bootstrap_means.append(sample.mean())

        ic_low, ic_high = np.percentile(bootstrap_means, [2.5, 97.5])

        print("=" * 80)
        print(csv_file)
        print("=" * 80)
        print(f"Dataset: {dataset}")
        print("Metric: f1_macro")
        print("Difference: OVA - multi-output")
        print(f"Seeds used: {len(diffs)}")
        print(f"Mean difference: {diffs.mean():.6f}")
        print(f"95% bootstrap CI: [{ic_low:.6f}, {ic_high:.6f}]")
        print()


if __name__ == "__main__":
    main()
