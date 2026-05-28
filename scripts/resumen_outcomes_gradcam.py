from pathlib import Path

import pandas as pd

# Para aciertos/outcomes da igual usar seed_1 o seed_1_true_target,
# porque true_label, multi_pred, ova_pred y outcome son iguales.
CSV_PATH = Path("resultados_actualizados/explicabilidad/brisc/seed_1/gradcam_index.csv")
OUTPUT_DIR = CSV_PATH.parent / "resumen_outcomes"


df = pd.read_csv(CSV_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Archivo: {CSV_PATH}")
print(f"Numero de imagenes: {len(df)}")


# 1. Accuracy global
multi_correct = (df["multi_pred"] == df["true_label"]).sum()
ova_correct = (df["ova_pred"] == df["true_label"]).sum()
n = len(df)

accuracy_summary = pd.DataFrame(
    [
        {
            "model": "multi-output",
            "correct": multi_correct,
            "total": n,
            "accuracy": multi_correct / n,
        },
        {
            "model": "OVA",
            "correct": ova_correct,
            "total": n,
            "accuracy": ova_correct / n,
        },
    ]
)

print("\nAccuracy global")
print(accuracy_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 2. Accuracy por clase real
rows = []
for label, group in df.groupby("true_label", sort=True):
    class_n = len(group)
    class_multi_correct = (group["multi_pred"] == group["true_label"]).sum()
    class_ova_correct = (group["ova_pred"] == group["true_label"]).sum()
    rows.append(
        {
            "class": label,
            "n": class_n,
            "multi_correct": class_multi_correct,
            "multi_accuracy": class_multi_correct / class_n,
            "ova_correct": class_ova_correct,
            "ova_accuracy": class_ova_correct / class_n,
            "ova_minus_multi_correct": class_ova_correct - class_multi_correct,
            "ova_minus_multi_accuracy": (class_ova_correct - class_multi_correct) / class_n,
        }
    )

class_accuracy_summary = pd.DataFrame(rows)

print("\nAccuracy por clase")
print(class_accuracy_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 3. Conteo de outcomes
outcome_order = [
    "both_correct",
    "multi_wrong_ova_correct",
    "multi_correct_ova_wrong",
    "both_wrong",
]
outcome_counts = (
    df["outcome"]
    .value_counts()
    .reindex(outcome_order, fill_value=0)
    .rename_axis("outcome")
    .reset_index(name="n")
)

print("\nConteo de outcomes")
print(outcome_counts.to_string(index=False))


# 4. Outcomes por clase real
outcome_by_class = pd.crosstab(df["true_label"], df["outcome"])
for outcome in outcome_order:
    if outcome not in outcome_by_class.columns:
        outcome_by_class[outcome] = 0
outcome_by_class = outcome_by_class[outcome_order].reset_index()

print("\nOutcomes por clase")
print(outcome_by_class.to_string(index=False))


# 5. Matrices de confusion
confusion_multi = pd.crosstab(df["true_label"], df["multi_pred"]).reset_index()
confusion_ova = pd.crosstab(df["true_label"], df["ova_pred"]).reset_index()

print("\nMatriz de confusion multi-output")
print(confusion_multi.to_string(index=False))

print("\nMatriz de confusion OVA")
print(confusion_ova.to_string(index=False))


# 6. Guardar CSVs
accuracy_summary.to_csv(OUTPUT_DIR / "accuracy_summary.csv", index=False)
class_accuracy_summary.to_csv(OUTPUT_DIR / "class_accuracy_summary.csv", index=False)
outcome_counts.to_csv(OUTPUT_DIR / "outcome_counts.csv", index=False)
outcome_by_class.to_csv(OUTPUT_DIR / "outcome_by_class.csv", index=False)
confusion_multi.to_csv(OUTPUT_DIR / "confusion_multi.csv", index=False)
confusion_ova.to_csv(OUTPUT_DIR / "confusion_ova.csv", index=False)

print(f"\nCSVs guardados en: {OUTPUT_DIR}")
