import argparse

from .experiment import main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TFM experimental framework for studying output coupling in neural networks"
    )
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--dataset", type=str, default="synthetic_multiclass")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--batch-normalization", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--coupling-modes", type=str, nargs="+", default=["ova"])
    parser.add_argument("--synthetic-samples", type=int, default=600)
    parser.add_argument("--synthetic-features", type=int, default=20)
    parser.add_argument("--synthetic-classes", type=int, default=4)
    parser.add_argument("--synthetic-targets", type=int, default=3)
    parser.add_argument("--dependency-strength", type=float, default=0.3)
    parser.add_argument("--output-csv", type=str, default="tfm_results.csv")
    parser.add_argument("--summary-csv", type=str, default="tfm_results_summary.csv")
    return parser


def cli() -> None:
    main(build_parser().parse_args())


if __name__ == "__main__":
    cli()

