import argparse

from .experiment import main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TFM experimental framework for studying output coupling in neural networks"
    )
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--dataset", type=str, default="synthetic_multiclass")
    parser.add_argument("--model-arch", choices=["mlp", "vgg", "vgg16-pretrained", "vit-b-16-pretrained"], default="mlp")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--vgg-channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--batch-normalization", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--coupling-modes", type=str, nargs="+", default=["ova"])
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    parser.add_argument("--ova-loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--focal-alpha",
        type=str,
        default="balanced",
        help="Alpha de focal loss para la clase positiva OVA: balanced, none o valor en (0, 1).",
    )
    parser.add_argument("--data-augmentation", choices=["none", "ham10000-basic"], default="none")
    parser.add_argument("--train-sampler", choices=["none", "balanced"], default="none")
    parser.add_argument("--pretrained-finetune", choices=["frozen", "block5", "last-block", "full"], default="frozen")
    parser.add_argument("--ova-calibration", choices=["none", "platt", "threshold", "threshold-f1"], default="none")
    parser.add_argument("--synthetic-samples", type=int, default=600)
    parser.add_argument("--synthetic-features", type=int, default=20)
    parser.add_argument("--synthetic-classes", type=int, default=4)
    parser.add_argument("--synthetic-targets", type=int, default=3)
    parser.add_argument("--dependency-strength", type=float, default=0.3)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--brisc-root", type=str, default="./data/brisc2025")
    parser.add_argument("--tb-root", type=str, default="./data/tb_chest_xray")
    parser.add_argument("--ham10000-root", type=str, default="./data/ham10000")
    parser.add_argument("--ham10000-test", choices=["internal", "official"], default="internal")
    parser.add_argument("--ham10000-split-csv", type=str, default=None)
    parser.add_argument("--ham10000-split-seed", type=int, default=2000)
    parser.add_argument("--ham10000-exclude-classes", type=str, nargs="*", default=[])
    parser.add_argument(
        "--ham10000-label-mode",
        choices=["original", "malignant_binary"],
        default="original",
        help=(
            "HAM10000 label formulation. 'original' keeps the 7 diagnostic "
            "classes; 'malignant_binary' groups akiec/bcc/mel as malignant "
            "and bkl/df/nv/vasc as non_malignant."
        ),
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-csv", type=str, default="tfm_results.csv")
    parser.add_argument("--summary-csv", type=str, default="tfm_results_summary.csv")
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help=(
            "Opcional. CSV donde guardar predicciones por muestra del test para "
            "experimentos de clasificacion."
        ),
    )
    return parser


def cli() -> None:
    main(build_parser().parse_args())


if __name__ == "__main__":
    cli()
