import subprocess


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


def main():
    for csv_file in CSV_FILES:
        print("=" * 80, flush=True)
        print(csv_file, flush=True)
        print("=" * 80, flush=True)

        subprocess.run(
            [
                "python",
                "scripts/wilcoxon_paired_test.py",
                "--csv",
                csv_file,
                "--metric",
                "f1_macro",
            ],
            check=True,
        )
        print(flush=True)


if __name__ == "__main__":
    main()
