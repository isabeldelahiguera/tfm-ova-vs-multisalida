import numpy as np
import pandas as pd


CSV_FILES = [
    "resultados_10semillas/iris_mlp_22semillas.csv",
    "resultados_10semillas/wine_mlp_10semillas.csv",
    "resultados_10semillas/breast_cancer_mlp_10semillas.csv",
    "resultados_10semillas/digits_mlp_10semillas.csv",
    "resultados_10semillas/mnist_vgg_10semillas.csv",
    "resultados_10semillas/cifar10_vgg_10semillas.csv",
    "resultados_10semillas/brisc_vgg_128_12semillas.csv",
    "resultados_10semillas/tuberculosis_vgg_128_10semillas.csv",
]


EQUIVALENCE_MARGIN = 0.02


def main():
    rng = np.random.default_rng(12345)

    for csv_file in CSV_FILES:
        df = pd.read_csv(csv_file)
        dataset = df["dataset"].iloc[0]

        wide = df.pivot_table(index="seed", columns="model_type", values="f1_macro")
        diffs = wide["OVA"] - wide["multi-output"]

        bootstrap_means = []
        for _ in range(10000):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            bootstrap_means.append(sample.mean())

        ic_low, ic_high = np.percentile(bootstrap_means, [5, 95])
        equivalent = ic_low > -EQUIVALENCE_MARGIN and ic_high < EQUIVALENCE_MARGIN

        print("=" * 80)
        print(csv_file)
        print("=" * 80)
        print(f"Dataset: {dataset}")
        print("Metric: f1_macro")
        print("Difference: OVA - multi-output")
        print(f"Equivalence margin: [-{EQUIVALENCE_MARGIN:.2f}, {EQUIVALENCE_MARGIN:.2f}]")
        print(f"Seeds used: {len(diffs)}")
        print(f"Mean difference: {diffs.mean():.6f}")
        print(f"90% bootstrap CI: [{ic_low:.6f}, {ic_high:.6f}]")
        if equivalent:
            print("TOST bootstrap decision: equivalent within +/-0.02")
        else:
            print("TOST bootstrap decision: equivalence not established")
        print()


if __name__ == "__main__":
    main()
