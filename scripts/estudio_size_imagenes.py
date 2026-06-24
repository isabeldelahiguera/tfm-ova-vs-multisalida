from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


BRISC_CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
TB_CLASS_NAMES = ["Normal", "Tuberculosis"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estudia los tamanos nativos de las imagenes antes del resize usado por los modelos."
    )
    parser.add_argument("--brisc-train-root", default="./data/brisc2025/train")
    parser.add_argument("--brisc-test-root", default="./data/brisc2025/test")
    parser.add_argument("--tb-root", default="./data/tb_chest_xray")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="resultados_actualizados/analisis_dataset/tamanos_imagenes",
        help="Directorio donde se guardan los CSV/TXT del estudio de tamanos.",
    )
    return parser


def collect_paths(root: str | Path, class_names: list[str]) -> list[tuple[int, str, Path]]:
    rows = []
    dataset_root = Path(root)
    for class_idx, class_name in enumerate(class_names):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class directory not found: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                rows.append((class_idx, class_name, image_path))
    return rows


def summarize_image_sizes(image_paths_with_class: list[tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    for class_name, image_path in image_paths_with_class:
        with Image.open(image_path) as image:
            width, height = image.size
        rows.append(
            {
                "class_name": class_name,
                "path": str(image_path),
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
            }
        )
    return pd.DataFrame(rows)


def summarize_dataset(title: str, image_paths_with_class: list[tuple[str, Path]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = summarize_image_sizes(image_paths_with_class)
    summary_by_class = df.groupby("class_name")[["width", "height", "aspect_ratio"]].agg(
        ["mean", "median", "min", "max", "std"]
    )
    summary_global = df[["width", "height", "aspect_ratio"]].agg(["mean", "median", "min", "max", "std"])
    summary_by_size = (
        df.groupby(["width", "height"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "width", "height"], ascending=[False, True, True])
    )

    print(f"=== {title} ===")
    print("Resumen por clase:")
    print(summary_by_class)
    print()
    print("Resumen global:")
    print(summary_global)
    print()
    print("Tamanos mas frecuentes:")
    print(summary_by_size.head(20).to_string(index=False))
    print()
    return df, summary_by_class, summary_global


def select_tb_train_split(tb_rows: list[tuple[int, str, Path]], seed: int) -> list[tuple[int, str, Path]]:
    tb_labels = [class_idx for class_idx, _, _ in tb_rows]
    tb_train_val, _ = train_test_split(
        tb_rows,
        test_size=0.15,
        random_state=seed,
        stratify=tb_labels,
    )
    tb_train_val_labels = [class_idx for class_idx, _, _ in tb_train_val]
    tb_train, _ = train_test_split(
        tb_train_val,
        test_size=0.17647058823529413,
        random_state=seed,
        stratify=tb_train_val_labels,
    )
    return tb_train


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    brisc_rows = collect_paths(args.brisc_train_root, BRISC_CLASS_NAMES)
    brisc_train_df, brisc_train_by_class, brisc_train_global = summarize_dataset(
        "BRISC train",
        [(class_name, image_path) for _, class_name, image_path in brisc_rows],
    )
    brisc_train_df.to_csv(output_dir / "brisc_train_tamanos_por_imagen.csv", index=False)
    brisc_train_by_class.to_csv(output_dir / "brisc_train_tamanos_por_clase.csv")
    brisc_train_global.to_csv(output_dir / "brisc_train_tamanos_global.csv")

    brisc_test_root = Path(args.brisc_test_root)
    if brisc_test_root.exists():
        brisc_test_rows = collect_paths(brisc_test_root, BRISC_CLASS_NAMES)
        brisc_test_df, brisc_test_by_class, brisc_test_global = summarize_dataset(
            "BRISC test",
            [(class_name, image_path) for _, class_name, image_path in brisc_test_rows],
        )
        brisc_test_df.to_csv(output_dir / "brisc_test_tamanos_por_imagen.csv", index=False)
        brisc_test_by_class.to_csv(output_dir / "brisc_test_tamanos_por_clase.csv")
        brisc_test_global.to_csv(output_dir / "brisc_test_tamanos_global.csv")

    tb_rows = collect_paths(args.tb_root, TB_CLASS_NAMES)
    tb_train = select_tb_train_split(tb_rows, args.seed)
    tb_train_df, tb_train_by_class, tb_train_global = summarize_dataset(
        f"TB train (same split logic as loader, seed={args.seed})",
        [(class_name, image_path) for _, class_name, image_path in tb_train],
    )
    tb_train_df.to_csv(output_dir / "tb_train_tamanos_por_imagen.csv", index=False)
    tb_train_by_class.to_csv(output_dir / "tb_train_tamanos_por_clase.csv")
    tb_train_global.to_csv(output_dir / "tb_train_tamanos_global.csv")

    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
