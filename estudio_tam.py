from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def summarize_image_sizes(image_paths_with_class):
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


def collect_paths(root, class_names):
    rows = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(root) / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            rows.append((class_idx, class_name, image_path))
    return rows


def summarize_and_print(title, image_paths_with_class):
    df = summarize_image_sizes(image_paths_with_class)

    summary_by_class = df.groupby("class_name")[["width", "height", "aspect_ratio"]].agg(
        ["mean", "median", "min", "max", "std"]
    )
    summary_global = df[["width", "height", "aspect_ratio"]].agg(["mean", "median", "min", "max", "std"])

    print(f"=== {title} ===")
    print("Resumen por clase:")
    print(summary_by_class)
    print()
    print("Resumen global:")
    print(summary_global)
    print()


brisc_root = "./data/brisc2025/train"
brisc_class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
brisc_rows = collect_paths(brisc_root, brisc_class_names)
summarize_and_print("BRISC train", [(class_name, image_path) for _, class_name, image_path in brisc_rows])


tb_root = "./data/tb_chest_xray"
tb_class_names = ["Normal", "Tuberculosis"]
tb_rows = collect_paths(tb_root, tb_class_names)
tb_labels = [class_idx for class_idx, _, _ in tb_rows]

tb_train_val, _ = train_test_split(
    tb_rows,
    test_size=0.15,
    random_state=1,
    stratify=tb_labels,
)
tb_train_val_labels = [class_idx for class_idx, _, _ in tb_train_val]
tb_train, _ = train_test_split(
    tb_train_val,
    test_size=0.17647058823529413, #0.15/0.85
    random_state=1,
    stratify=tb_train_val_labels,
)

summarize_and_print("TB train (same split logic as loader, seed=1)", [(class_name, image_path) for _, class_name, image_path in tb_train])
