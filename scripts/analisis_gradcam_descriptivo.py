from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PREDICTED_INDEX = Path("resultados_actualizados/explicabilidad/brisc/seed_1/gradcam_index.csv")
TRUE_INDEX = Path("resultados_actualizados/explicabilidad/brisc/seed_1_true_target/gradcam_index.csv")
OUTPUT_DIR = Path("resultados_actualizados/explicabilidad/brisc/analisis_descriptivo")

METRICS = {
    "cam_top20_dice": "Dice@20",
    "cam_top20_iou": "IoU@20",
    "cam_inside_frac": "Activacion dentro tumor",
    "cam_pointing_game_hit": "Pointing Game",
    "cam_active_area_frac_50": "Area activa >= 0.5",
    "cam_gini": "Gini",
}


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cargar(path, modo):
    df = pd.read_csv(path)
    df["modo_gradcam"] = modo
    return df


predicted = cargar(PREDICTED_INDEX, "predicted")
true_target = cargar(TRUE_INDEX, "true_target")
df = pd.concat([predicted, true_target], ignore_index=True)


# 1. Tabla resumen global
rows = []
for modo, group in df.groupby("modo_gradcam"):
    for metric, label in METRICS.items():
        multi = group[f"multi_{metric}"].mean()
        ova = group[f"ova_{metric}"].mean()
        rows.append(
            {
                "modo_gradcam": modo,
                "metric": label,
                "multi_mean": multi,
                "ova_mean": ova,
                "diff_ova_minus_multi": ova - multi,
            }
        )

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / "gradcam_resumen_global.csv", index=False)

print("\nResumen global")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 2. Tabla por clase
rows = []
for modo, modo_group in df.groupby("modo_gradcam"):
    for true_label, class_group in modo_group.groupby("true_label"):
        for metric, label in METRICS.items():
            multi = class_group[f"multi_{metric}"].mean()
            ova = class_group[f"ova_{metric}"].mean()
            rows.append(
                {
                    "modo_gradcam": modo,
                    "true_label": true_label,
                    "metric": label,
                    "n": len(class_group),
                    "multi_mean": multi,
                    "ova_mean": ova,
                    "diff_ova_minus_multi": ova - multi,
                }
            )

by_class = pd.DataFrame(rows)
by_class.to_csv(OUTPUT_DIR / "gradcam_resumen_por_clase.csv", index=False)

print("\nResumen por clase")
print(by_class.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 3. Tabla por outcome
rows = []
for modo, modo_group in df.groupby("modo_gradcam"):
    for outcome, outcome_group in modo_group.groupby("outcome"):
        for metric, label in METRICS.items():
            multi = outcome_group[f"multi_{metric}"].mean()
            ova = outcome_group[f"ova_{metric}"].mean()
            rows.append(
                {
                    "modo_gradcam": modo,
                    "outcome": outcome,
                    "metric": label,
                    "n": len(outcome_group),
                    "multi_mean": multi,
                    "ova_mean": ova,
                    "diff_ova_minus_multi": ova - multi,
                }
            )

by_outcome = pd.DataFrame(rows)
by_outcome.to_csv(OUTPUT_DIR / "gradcam_resumen_por_outcome.csv", index=False)

print("\nResumen por outcome")
print(by_outcome.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 4. Graficos globales multi vs OVA
for modo, modo_group in df.groupby("modo_gradcam"):
    plot_rows = []
    for metric, label in METRICS.items():
        plot_rows.append({"metric": label, "model": "Multi-output", "value": modo_group[f"multi_{metric}"].mean()})
        plot_rows.append({"metric": label, "model": "OVA", "value": modo_group[f"ova_{metric}"].mean()})
    plot_df = pd.DataFrame(plot_rows)

    plt.figure(figsize=(11, 5))
    for model, color in [("Multi-output", "#4C78A8"), ("OVA", "#F58518")]:
        subset = plot_df[plot_df["model"] == model]
        positions = range(len(subset))
        offset = -0.18 if model == "Multi-output" else 0.18
        plt.bar([p + offset for p in positions], subset["value"], width=0.36, label=model, color=color)
    plt.xticks(range(len(METRICS)), list(METRICS.values()), rotation=25, ha="right")
    plt.ylabel("Media")
    plt.title(f"Metricas Grad-CAM globales ({modo})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"global_{modo}.png", dpi=180)
    plt.close()


# 5. Histogramas de diferencias OVA - multi
for modo, modo_group in df.groupby("modo_gradcam"):
    for metric, label in METRICS.items():
        diff = modo_group[f"ova_{metric}"] - modo_group[f"multi_{metric}"]
        plt.figure(figsize=(6, 4))
        plt.hist(diff, bins=40, color="#72B7B2", edgecolor="white")
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("OVA - multi-output")
        plt.ylabel("Numero de imagenes")
        plt.title(f"Diferencia pareada: {label} ({modo})")
        plt.tight_layout()
        safe_name = metric.replace("cam_", "")
        plt.savefig(OUTPUT_DIR / f"hist_diff_{modo}_{safe_name}.png", dpi=180)
        plt.close()


# 6. Scatter multi vs OVA para las metricas principales
for modo, modo_group in df.groupby("modo_gradcam"):
    for metric, label in METRICS.items():
        x = modo_group[f"multi_{metric}"]
        y = modo_group[f"ova_{metric}"]
        min_value = min(x.min(), y.min())
        max_value = max(x.max(), y.max())
        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, s=12, alpha=0.45)
        plt.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)
        plt.xlabel("Multi-output")
        plt.ylabel("OVA")
        plt.title(f"{label}: OVA vs multi ({modo})")
        plt.tight_layout()
        safe_name = metric.replace("cam_", "")
        plt.savefig(OUTPUT_DIR / f"scatter_{modo}_{safe_name}.png", dpi=180)
        plt.close()


print(f"\nResultados guardados en: {OUTPUT_DIR}")
