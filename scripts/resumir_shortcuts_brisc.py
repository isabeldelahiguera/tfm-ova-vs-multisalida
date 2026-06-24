from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = "resultados_actualizados/analisis_dataset/brisc_train"


FAMILY_INFO: dict[str, dict[str, str]] = {
    "intensidad_global_imagen": {
        "que_mide": (
            "Intensidad global de la imagen; candidato a shortcut si una clase/plano "
            "es globalmente mas claro u oscuro."
        ),
        "lectura_diff": (
            "Mas positivo: imagen globalmente mas brillante. Mas negativo: imagen "
            "globalmente mas oscura."
        ),
    },
    "area_visible_imagen": {
        "que_mide": (
            "Fraccion de imagen no negra/visible; puede reflejar campo de vision, "
            "recorte o plano de adquisicion."
        ),
        "lectura_diff": (
            "Mas positivo: mayor fraccion visible/no negra. Mas negativo: menor campo visible."
        ),
    },
    "intensidad_zona_visible": {
        "que_mide": (
            "Intensidad dentro de la zona visible/no negra; puede reflejar campo de "
            "vision, recorte o protocolo."
        ),
        "lectura_diff": (
            "Mas positivo: zona visible mas intensa/variable. Mas negativo: zona visible "
            "menos intensa/variable."
        ),
    },
    "posicion_area_visible": {
        "que_mide": (
            "Centroide de la zona visible/no negra; puede capturar desplazamientos del "
            "campo de vision."
        ),
        "lectura_diff": "Cambio de posicion de la zona visible; puede reflejar recorte o plano.",
    },
    "intensidad_tumor": {
        "que_mide": (
            "Intensidad dentro de la mascara tumoral; puede indicar que una clase tiene "
            "tumores sistematicamente mas brillantes u oscuros."
        ),
        "lectura_diff": "Mas positivo: tumor mas brillante. Mas negativo: tumor mas oscuro.",
    },
    "intensidad_peritumor": {
        "que_mide": (
            "Intensidad en el anillo peritumoral; puede indicar diferencias del entorno "
            "inmediato del tumor."
        ),
        "lectura_diff": (
            "Mas positivo: peritumor mas intenso/variable. Mas negativo: peritumor menos "
            "intenso/variable."
        ),
    },
    "contexto_fuera_peritumor": {
        "que_mide": (
            "Intensidad/area fuera del tumor y peritumor; candidato claro a shortcut "
            "contextual porque no pertenece a la mascara tumoral."
        ),
        "lectura_diff": (
            "Mas positivo: contexto fuera del tumor/peritumor mas intenso o mayor. "
            "Mas negativo: contexto exterior mas oscuro o menor."
        ),
    },
    "contraste_tumor_vs_fuera": {
        "que_mide": (
            "Diferencia/ratio entre intensidad del tumor y tejido fuera del entorno "
            "tumoral; si es alta, la clase puede distinguirse por un tumor que destaca "
            "frente al contexto."
        ),
        "lectura_diff": (
            "Mas positivo: tumor mas contrastado contra el exterior. Mas negativo: menor "
            "contraste tumor-exterior que el resto."
        ),
    },
    "contraste_tumor_vs_peritumor": {
        "que_mide": (
            "Diferencia/ratio entre tumor y peritumor; si es baja, el tumor se separa "
            "poco de su entorno inmediato."
        ),
        "lectura_diff": (
            "Mas positivo: tumor separado del peritumor. Mas negativo: tumor parecido "
            "al peritumor o peritumor relativamente intenso."
        ),
    },
    "posicion_tumor_centro": {
        "que_mide": (
            "Distancia del centroide tumoral al centro de la imagen; mide si una clase "
            "tiende a estar mas centrada o desplazada."
        ),
        "lectura_diff": (
            "Mas positivo: tumor mas alejado del centro. Mas negativo: tumor mas centrado."
        ),
    },
    "posicion_bbox_centro": {
        "que_mide": (
            "Distancia del centro de la bbox tumoral al centro de la imagen; similar a "
            "posicion del tumor."
        ),
        "lectura_diff": (
            "Mas positivo: bbox mas alejada del centro. Mas negativo: bbox mas centrada."
        ),
    },
    "bbox_tumor": {
        "que_mide": "Tamano, forma o extension de la caja envolvente de la mascara tumoral.",
        "lectura_diff": (
            "Mas positivo: bbox mayor o mas desplazada segun la variable concreta. "
            "Mas negativo: bbox menor o en sentido contrario."
        ),
    },
    "forma_tumor": {
        "que_mide": "Forma de la mascara tumoral: compacidad, relleno, perimetro o diametro.",
        "lectura_diff": "La lectura depende de la variable concreta de forma.",
    },
    "otra_variable": {
        "que_mide": "Variable morfologica o descriptiva no agrupada en una familia principal.",
        "lectura_diff": "La lectura depende de la variable concreta.",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume los contrastes de BRISC en tablas de posibles shortcuts. "
            "La regla es: contrastes completos -> familia de variable -> mejor "
            "feature por familia -> top señales por clase o clase+plano."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-abs-std", type=float, default=0.15)
    parser.add_argument("--top-clase", type=int, default=7)
    parser.add_argument("--top-clase-plano", type=int, default=6)
    return parser


def family_for_feature(feature: str) -> str:
    if feature.startswith("image_intensity_"):
        return "intensidad_global_imagen"
    if feature == "image_nonzero_area_frac":
        return "area_visible_imagen"
    if feature.startswith("image_nonzero_intensity_"):
        return "intensidad_zona_visible"
    if feature.startswith("image_nonzero_centroid_"):
        return "posicion_area_visible"
    if feature.startswith("tumor_intensity_") or feature == "tumor_area_frac":
        return "intensidad_tumor"
    if feature.startswith("peritumor_"):
        return "intensidad_peritumor"
    if feature.startswith("outside_peritumor_"):
        return "contexto_fuera_peritumor"
    if feature.startswith("tumor_vs_outside_") or feature == "tumor_vs_outside_mean_diff":
        return "contraste_tumor_vs_fuera"
    if feature.startswith("tumor_vs_peritumor_") or feature == "tumor_vs_peritumor_mean_diff":
        return "contraste_tumor_vs_peritumor"
    if feature == "mask_centroid_distance_center_norm":
        return "posicion_tumor_centro"
    if feature == "mask_bbox_center_distance_center_norm":
        return "posicion_bbox_centro"
    if feature.startswith("mask_bbox_"):
        return "bbox_tumor"
    if feature.startswith(("mask_compactness", "mask_perimeter", "mask_equivalent", "mask_area")):
        return "forma_tumor"
    return "otra_variable"


def strength(abs_std: float) -> str:
    if abs_std >= 1.25:
        return "muy_alta"
    if abs_std >= 0.75:
        return "alta"
    if abs_std >= 0.40:
        return "media"
    return "baja"


def annotate(contrasts: pd.DataFrame) -> pd.DataFrame:
    df = contrasts.copy()
    df["familia_variable"] = df["feature"].map(family_for_feature)
    df["que_mide"] = df["familia_variable"].map(
        lambda family: FAMILY_INFO[family]["que_mide"]
    )
    df["lectura_diff"] = df["familia_variable"].map(
        lambda family: FAMILY_INFO[family]["lectura_diff"]
    )
    df["direccion"] = df["standardized_diff"].map(
        lambda value: "mayor_en_grupo" if value > 0 else "menor_en_grupo"
    )
    df["fuerza_senal"] = df["abs_standardized_diff"].map(strength)
    return df


def select_key_shortcuts(
    contrasts: pd.DataFrame,
    group_cols: list[str],
    *,
    top_per_group: int,
    min_abs_std: float,
) -> pd.DataFrame:
    df = annotate(contrasts)
    df = df[df["abs_standardized_diff"] >= min_abs_std].copy()
    df = df.sort_values("abs_standardized_diff", ascending=False)

    # Regla central: no repetimos variantes casi iguales de la misma familia.
    # Ejemplo: outside_peritumor_r5/r10/r15 compiten dentro de contexto_fuera_peritumor.
    df = df.drop_duplicates([*group_cols, "familia_variable"], keep="first")
    df = df.groupby(group_cols, group_keys=False).head(top_per_group)
    return df.sort_values([*group_cols, "abs_standardized_diff"], ascending=[True] * len(group_cols) + [False])


def write_digest(shortcuts: pd.DataFrame, group_cols: list[str], output_path: Path) -> None:
    rows = []
    groupby_key = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, group in shortcuts.groupby(groupby_key, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n_senales"] = int(len(group))
        row["max_abs_standardized_diff"] = float(group["abs_standardized_diff"].max())
        row["senales_principales"] = "; ".join(
            f"{item.feature} ({item.direccion}, d={item.standardized_diff:.2f})"
            for item in group.itertuples(index=False)
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_readme(output_dir: Path, min_abs_std: float, top_clase: int, top_clase_plano: int) -> None:
    text = f"""# Resumen de posibles shortcuts en BRISC train

Estos ficheros condensan los contrastes grandes del analisis del dataset.
Cada fila compara un grupo frente al resto del train.

Regla de generacion:
1. Se parte de `contrastes_train_por_clase.csv` y `contrastes_train_por_clase_plano.csv`.
2. Para cada grupo y cada feature se usa `abs_standardized_diff` como fuerza discriminativa.
3. Cada feature se asigna a una `familia_variable` mediante reglas por nombre:
   - `image_intensity_*` -> `intensidad_global_imagen`
   - `image_nonzero_area_frac` -> `area_visible_imagen`
   - `image_nonzero_intensity_*` -> `intensidad_zona_visible`
   - `image_nonzero_centroid_*` -> `posicion_area_visible`
   - `tumor_intensity_*` -> `intensidad_tumor`
   - `peritumor_*` -> `intensidad_peritumor`
   - `outside_peritumor_*` -> `contexto_fuera_peritumor`
   - `tumor_vs_outside_*` -> `contraste_tumor_vs_fuera`
   - `tumor_vs_peritumor_*` -> `contraste_tumor_vs_peritumor`
   - `mask_centroid_distance_center_norm` -> `posicion_tumor_centro`
   - `mask_bbox_center_distance_center_norm` -> `posicion_bbox_centro`
   - `mask_bbox_*` -> `bbox_tumor`
   - variables de compacidad/perimetro/diametro/area -> `forma_tumor`
4. Dentro de cada grupo y familia se conserva solo la feature con mayor `abs_standardized_diff`.
5. Se descartan senales con `abs_standardized_diff < {min_abs_std}`.
6. Se conservan como maximo {top_clase} senales por clase y {top_clase_plano} por clase+plano.

Columnas clave:
- `group_mean`: media de la variable en la clase o clase+plano.
- `rest_mean`: media en el resto de imagenes.
- `diff_group_minus_rest`: `group_mean - rest_mean`.
- `standardized_diff`: diferencia estandarizada; cuanto mayor el valor absoluto, mas discriminativa es la variable.
- `familia_variable`: agrupacion legible para no repetir radios 5/10/15 o variantes muy parecidas.
- `direccion`: indica si la variable es mayor o menor en el grupo.
- `fuerza_senal`: clasificacion orientativa segun `abs_standardized_diff`.

Ficheros:
- `shortcuts_clave_por_clase.csv`: principales senales por clase.
- `shortcuts_clave_por_clase_plano.csv`: principales senales por combinacion clase+plano.
- `shortcuts_digest_por_clase.csv`: version ultra-resumida, una fila por clase.
- `shortcuts_digest_por_clase_plano.csv`: version ultra-resumida, una fila por clase+plano.

Interpretacion metodologica:
Estas variables no prueban por si solas que el modelo las use. Son candidatos a shortcut porque son regularidades discriminativas del conjunto de entrenamiento. Se deben conectar con Grad-CAM/LRP y oclusion: si una clase tiene baja relevancia tumoral y baja sensibilidad a ocluir el tumor, estas senales contextuales o globales ganan plausibilidad como explicacion alternativa.
"""
    (output_dir / "README_shortcuts.md").write_text(text)


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "resumen_shortcuts"
    output_dir.mkdir(parents=True, exist_ok=True)

    by_class = pd.read_csv(input_dir / "contrastes_train_por_clase.csv")
    by_class_plane = pd.read_csv(input_dir / "contrastes_train_por_clase_plano.csv")

    class_shortcuts = select_key_shortcuts(
        by_class,
        ["true_label"],
        top_per_group=args.top_clase,
        min_abs_std=args.min_abs_std,
    )
    class_plane_shortcuts = select_key_shortcuts(
        by_class_plane,
        ["true_label", "plane", "plane_name"],
        top_per_group=args.top_clase_plano,
        min_abs_std=args.min_abs_std,
    )

    class_shortcuts.to_csv(output_dir / "shortcuts_clave_por_clase.csv", index=False)
    class_plane_shortcuts.to_csv(output_dir / "shortcuts_clave_por_clase_plano.csv", index=False)
    write_digest(
        class_shortcuts,
        ["true_label"],
        output_dir / "shortcuts_digest_por_clase.csv",
    )
    write_digest(
        class_plane_shortcuts,
        ["true_label", "plane", "plane_name"],
        output_dir / "shortcuts_digest_por_clase_plano.csv",
    )
    write_readme(output_dir, args.min_abs_std, args.top_clase, args.top_clase_plano)


if __name__ == "__main__":
    main()
