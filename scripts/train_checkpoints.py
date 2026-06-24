from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from checkpoint_utils import checkpoints_exist, save_model_checkpoints
from tfm.data import load_experiment_data
from tfm.experiment import train_multiclass_model, train_ova_models
from tfm.training import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train multi-output and OVA models and save checkpoints for later "
            "evaluation or explainability analyses."
        )
    )
    parser.add_argument("--task", choices=["classification"], default="classification")
    parser.add_argument("--dataset", choices=["brisc", "tb_chest_xray", "ham10000"], required=True)
    parser.add_argument(
        "--model-arch",
        choices=["vgg", "vgg16-pretrained", "vit-b-16-pretrained"],
        default="vgg",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--vgg-channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    parser.add_argument("--ova-loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=str, default="balanced")
    parser.add_argument("--data-augmentation", choices=["none", "ham10000-basic"], default="none")
    parser.add_argument("--train-sampler", choices=["none", "balanced"], default="none")
    parser.add_argument("--pretrained-finetune", choices=["frozen", "block5", "last-block", "full"], default="frozen")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--brisc-root", default="/mnt/homeGPU/imhiguera/data/brisc2025")
    parser.add_argument("--tb-root", default="/mnt/homeGPU/imhiguera/data/tb_chest_xray")
    parser.add_argument("--ham10000-root", default="/mnt/homeGPU/imhiguera/data/ham10000")
    parser.add_argument("--ham10000-test", choices=["internal", "official"], default="internal")
    parser.add_argument("--ham10000-split-csv", default=None)
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
    parser.add_argument("--checkpoint-dir", default="resultados_actualizados/checkpoints")
    parser.add_argument("--run-tag", default="")
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Skip seeds whose checkpoint files already exist.",
    )
    return parser


def experiment_args(parsed: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        task="classification",
        dataset=parsed.dataset,
        model_arch=parsed.model_arch,
        hidden_layers=[32, 16],
        vgg_channels=parsed.vgg_channels,
        batch_normalization=False,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        early_stopping_patience=parsed.early_stopping_patience,
        early_stopping_min_delta=parsed.early_stopping_min_delta,
        learning_rate=parsed.learning_rate,
        seed=seed,
        seeds=[seed],
        coupling_modes=["multi-output", "ova"],
        class_weighting=parsed.class_weighting,
        ova_loss=parsed.ova_loss,
        focal_gamma=parsed.focal_gamma,
        focal_alpha=parsed.focal_alpha,
        data_augmentation=parsed.data_augmentation,
        train_sampler=parsed.train_sampler,
        pretrained_finetune=parsed.pretrained_finetune,
        ova_calibration="none",
        synthetic_samples=600,
        synthetic_features=20,
        synthetic_classes=4,
        synthetic_targets=3,
        dependency_strength=0.3,
        max_train=parsed.max_train if parsed.max_train and parsed.max_train > 0 else None,
        max_test=parsed.max_test if parsed.max_test and parsed.max_test > 0 else None,
        brisc_root=parsed.brisc_root,
        tb_root=parsed.tb_root,
        ham10000_root=parsed.ham10000_root,
        ham10000_test=parsed.ham10000_test,
        ham10000_split_csv=parsed.ham10000_split_csv,
        ham10000_split_seed=parsed.ham10000_split_seed,
        ham10000_exclude_classes=parsed.ham10000_exclude_classes,
        ham10000_label_mode=parsed.ham10000_label_mode,
        image_size=parsed.image_size,
    )


def parsed_for_seed(parsed: argparse.Namespace, seed: int) -> argparse.Namespace:
    seed_parsed = argparse.Namespace(**vars(parsed))
    seed_parsed.seed = seed
    return seed_parsed


def main() -> None:
    parsed = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training checkpoints on device={device}", flush=True)

    for seed in parsed.seeds:
        seed_parsed = parsed_for_seed(parsed, seed)
        args = experiment_args(seed_parsed, seed)
        set_seed(seed)
        print(f"===== Training checkpoint seed {seed} =====", flush=True)
        experiment_data = load_experiment_data(args, seed)
        print(
            f"Loaded {experiment_data.dataset_name}, target_dim={experiment_data.target_dim}, "
            f"train={len(experiment_data.y_train)}, val={len(experiment_data.y_val)}, "
            f"test={len(experiment_data.y_test)}",
            flush=True,
        )
        if parsed.reuse_checkpoints and checkpoints_exist(seed_parsed, experiment_data.target_dim):
            print("Checkpoints already exist; skipping seed.", flush=True)
            continue

        multi_model, _, multi_training_info = train_multiclass_model(experiment_data, args, device, seed)
        ova_models, _, ova_training_info = train_ova_models(experiment_data, args, device, seed)
        save_model_checkpoints(
            seed_parsed,
            experiment_data,
            multi_model,
            ova_models,
            multi_training_info,
            ova_training_info,
            device,
            args,
        )


if __name__ == "__main__":
    main()
