from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("resultados_actualizados")
OUTPUT_DIR = ROOT / "analisis_arquitecturas_ova"

REFERENCE_SEQUENTIAL = ROOT / "secuencial"
REFERENCE_PARALLEL = ROOT / "paralelo" / "ova_parallel_actual"

REFERENCE_ARCHITECTURES = {
    "mlp": "[32, 16]",
    "vgg": "[32, 64, 128]",
}

DATASET_GROUPS = {
    "iris": "clasico",
    "wine": "clasico",
    "breast_cancer": "clasico",
    "digits": "clasico",
    "mnist": "vgg",
    "cifar10": "vgg",
    "brisc": "vgg",
    "tb_chest_xray": "vgg",
}


def summary_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*_summary.csv"):
        relative = path.relative_to(ROOT)
        if str(relative).startswith("analisis_"):
            continue
        files.append(path)
    return sorted(files)


def family_for(path: Path) -> str:
    rel_parent = str(path.parent.relative_to(ROOT))
    if rel_parent == "secuencial":
        return "referencia_secuencial"
    if rel_parent == "ampliados":
        return "ampliado"
    if rel_parent == "arquitecturas_ova":
        return "ova_reducida_mlp_secuencial"
    if rel_parent.startswith("arquitecturas_ova_vgg"):
        return "ova_reducida_vgg_secuencial"
    if rel_parent == "paralelo/ova_parallel_actual":
        return "ova_referencia_paralelo"
    if rel_parent.startswith("paralelo/ova_parallel_reducida"):
        return "ova_reducida_paralelo"
    if rel_parent.startswith("paralelo/"):
        return "ova_hiperparametros_paralelo"
    return rel_parent


def normalized_architecture(row: pd.Series) -> str:
    if row["model_arch"] == "mlp":
        return str(row.get("hidden_layers", "n/a"))
    return str(row.get("vgg_channels", "n/a"))


def load_experiments() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in summary_files():
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            continue
        for _, row in df.iterrows():
            if "seed" in df.columns and len(df) > 1 and str(row.get("seed", "")) != "mean":
                continue
            model_arch = str(row.get("model_arch", ""))
            dataset = str(row.get("dataset", ""))
            rows.append(
                {
                    "familia": family_for(path),
                    "grupo_dataset": DATASET_GROUPS.get(dataset, "otro"),
                    "dataset": dataset,
                    "model_type": row.get("model_type"),
                    "model_arch": model_arch,
                    "architecture": normalized_architecture(row),
                    "target_dim": row.get("target_dim"),
                    "batch_size": row.get("batch_size"),
                    "learning_rate": row.get("learning_rate"),
                    "early_stopping_patience": row.get("early_stopping_patience"),
                    "early_stopping_min_delta": row.get("early_stopping_min_delta"),
                    "epochs": row.get("epochs"),
                    "accuracy": row.get("accuracy"),
                    "balanced_accuracy": row.get("balanced_accuracy"),
                    "precision_macro": row.get("precision_macro"),
                    "recall_macro": row.get("recall_macro"),
                    "f1_macro": row.get("f1_macro"),
                    "tpr_macro": row.get("tpr_macro"),
                    "fpr_macro": row.get("fpr_macro"),
                    "tnr_macro": row.get("tnr_macro"),
                    "fnr_macro": row.get("fnr_macro"),
                    "train_time_seconds": row.get("train_time_seconds"),
                    "parallel_train_time_seconds": row.get("parallel_train_time_seconds"),
                    "ova_model_train_time_seconds_mean": row.get("ova_model_train_time_seconds_mean"),
                    "best_val_loss": row.get("best_val_loss"),
                    "epochs_trained": row.get("epochs_trained"),
                    "total_epochs_trained": row.get("total_epochs_trained"),
                    "models_stopped_early": row.get("models_stopped_early"),
                    "source_path": str(path),
                }
            )
    return pd.DataFrame(rows)


def first_value(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    value = df.iloc[0][column]
    if pd.isna(value):
        return None
    return float(value)


def reference_table(experiments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in experiments.groupby("dataset", sort=True):
        if dataset not in DATASET_GROUPS:
            continue
        model_arch = str(group["model_arch"].dropna().iloc[0])
        ref_arch = REFERENCE_ARCHITECTURES[model_arch]
        multi_seq = group[
            (group["familia"] == "referencia_secuencial")
            & (group["model_type"] == "multi-output")
            & (group["architecture"] == ref_arch)
        ]
        ova_seq = group[
            (group["familia"] == "referencia_secuencial")
            & (group["model_type"] == "OVA")
            & (group["architecture"] == ref_arch)
        ]
        ova_par = group[
            (group["familia"] == "ova_referencia_paralelo")
            & (group["model_type"] == "OVA")
            & (group["architecture"] == ref_arch)
        ]
        multi_time = first_value(multi_seq, "train_time_seconds")
        ova_seq_time = first_value(ova_seq, "train_time_seconds")
        ova_par_time = first_value(ova_par, "parallel_train_time_seconds")
        rows.append(
            {
                "grupo_dataset": DATASET_GROUPS[dataset],
                "dataset": dataset,
                "model_arch": model_arch,
                "arquitectura_referencia": ref_arch,
                "multi_f1": first_value(multi_seq, "f1_macro"),
                "multi_accuracy": first_value(multi_seq, "accuracy"),
                "multi_time_seconds": multi_time,
                "ova_seq_f1": first_value(ova_seq, "f1_macro"),
                "ova_seq_accuracy": first_value(ova_seq, "accuracy"),
                "ova_seq_time_seconds": ova_seq_time,
                "ova_parallel_f1": first_value(ova_par, "f1_macro"),
                "ova_parallel_accuracy": first_value(ova_par, "accuracy"),
                "ova_parallel_wall_time_seconds": ova_par_time,
                "ova_seq_vs_multi_time_ratio": (
                    ova_seq_time / multi_time if multi_time and ova_seq_time else None
                ),
                "ova_parallel_vs_multi_time_ratio": (
                    ova_par_time / multi_time if multi_time and ova_par_time else None
                ),
                "ova_parallel_speedup_vs_ova_seq": (
                    ova_seq_time / ova_par_time if ova_seq_time and ova_par_time else None
                ),
                "ova_seq_f1_minus_multi": (
                    first_value(ova_seq, "f1_macro") - first_value(multi_seq, "f1_macro")
                    if first_value(ova_seq, "f1_macro") is not None
                    and first_value(multi_seq, "f1_macro") is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def add_reference_comparisons(experiments: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    ref = references.set_index("dataset")
    rows = []
    for _, row in experiments.iterrows():
        dataset = row["dataset"]
        if dataset not in ref.index:
            continue
        r = ref.loc[dataset]
        measured_time = row["parallel_train_time_seconds"]
        time_mode = "paralelo" if pd.notna(measured_time) else "secuencial"
        if pd.isna(measured_time):
            measured_time = row["train_time_seconds"]
        rows.append(
            {
                **row.to_dict(),
                "time_mode_usado": time_mode,
                "measured_wall_time_seconds": measured_time,
                "ref_architecture": r["arquitectura_referencia"],
                "ref_multi_f1": r["multi_f1"],
                "ref_ova_seq_f1": r["ova_seq_f1"],
                "ref_ova_parallel_f1": r["ova_parallel_f1"],
                "ref_multi_time_seconds": r["multi_time_seconds"],
                "ref_ova_seq_time_seconds": r["ova_seq_time_seconds"],
                "ref_ova_parallel_wall_time_seconds": r["ova_parallel_wall_time_seconds"],
                "f1_minus_multi_ref": row["f1_macro"] - r["multi_f1"],
                "f1_minus_ova_seq_ref": row["f1_macro"] - r["ova_seq_f1"],
                "time_ratio_vs_multi_ref": (
                    measured_time / r["multi_time_seconds"]
                    if pd.notna(measured_time) and pd.notna(r["multi_time_seconds"]) and r["multi_time_seconds"] != 0
                    else None
                ),
                "time_ratio_vs_ova_seq_ref": (
                    measured_time / r["ova_seq_time_seconds"]
                    if pd.notna(measured_time) and pd.notna(r["ova_seq_time_seconds"]) and r["ova_seq_time_seconds"] != 0
                    else None
                ),
                "time_ratio_vs_ova_parallel_ref": (
                    measured_time / r["ova_parallel_wall_time_seconds"]
                    if pd.notna(measured_time)
                    and pd.notna(r["ova_parallel_wall_time_seconds"])
                    and r["ova_parallel_wall_time_seconds"] != 0
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(references: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    lines = [
        "# Resumen de arquitecturas y referencias",
        "",
        "Este resumen separa los casos de referencia de las pruebas posteriores.",
        "",
        "## Casos de referencia",
        "",
        "- Clasicos MLP: `multi-output [32, 16]` y `OVA [32, 16]`.",
        "- Imagen VGG: `multi-output [32, 64, 128]` y `OVA [32, 64, 128]`.",
        "- Para OVA hay dos tiempos de referencia: secuencial y paralelo ideal/medido por `parallel_train_time_seconds`.",
        "",
        references[
            [
                "grupo_dataset",
                "dataset",
                "arquitectura_referencia",
                "multi_f1",
                "ova_seq_f1",
                "multi_time_seconds",
                "ova_seq_time_seconds",
                "ova_parallel_wall_time_seconds",
                "ova_parallel_speedup_vs_ova_seq",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Pruebas comparadas",
        "",
        "La tabla `comparaciones_pruebas_vs_referencias.csv` contiene cada prueba con:",
        "",
        "- diferencia de `f1_macro` frente a Multi de referencia;",
        "- diferencia de `f1_macro` frente a OVA secuencial de referencia;",
        "- ratio de tiempo frente a Multi, OVA secuencial y OVA paralelo de referencia;",
        "- modo de tiempo usado: secuencial o paralelo.",
        "",
        "Filas por familia:",
        "",
        comparisons.groupby("familia").size().reset_index(name="n").to_markdown(index=False),
        "",
    ]
    (OUTPUT_DIR / "resumen_arquitecturas_referencias_y_pruebas.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments()
    references = reference_table(experiments)
    comparisons = add_reference_comparisons(experiments, references)

    experiments.to_csv(OUTPUT_DIR / "experimentos_normalizados.csv", index=False)
    references.to_csv(OUTPUT_DIR / "referencias_secuencial_paralelo.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR / "comparaciones_pruebas_vs_referencias.csv", index=False)
    write_markdown(references, comparisons)

    print(f"Experimentos normalizados: {len(experiments)}")
    print(f"Referencias: {len(references)}")
    print(f"Comparaciones: {len(comparisons)}")
    print(f"Salida: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
