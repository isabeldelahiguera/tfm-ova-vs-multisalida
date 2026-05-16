# Capacidades adicionales del código

Este documento recoge funcionalidades que el código soporta, aunque no formen parte del análisis principal ni de los resultados finales versionados en `resultados_10semillas/`.

El foco actual del TFM es la comparación en clasificación entre una red `multi-output` y la descomposición `OVA`. Lo descrito aquí debe entenderse como soporte exploratorio o extensiones posibles.

## Tareas Soportadas

El argumento `--task` permite dos modos:

- `classification`: modo principal del TFM.
- `regression`: modo implementado para experimentos multi-output de regresión, pero no usado en los resultados finales actuales.

En clasificación, el código siempre entrena una red `multi-output`. Si se usa `--coupling-modes ova`, entrena también una red binaria por clase. Si se usa `--coupling-modes ovo`, puede entrenar una descomposición One-vs-One cuando hay más de dos clases.

En regresión, el código siempre entrena una red `multi-output`. Si se usa `--coupling-modes decoupled`, entrena una red independiente por variable objetivo.

## Arquitecturas

El argumento `--model-arch` soporta:

- `mlp`: red fully connected configurable con `--hidden-layers`. Es la arquitectura por defecto.
- `vgg`: CNN compacta inspirada en VGG. Solo se permite en clasificación con `mnist`, `cifar10`, `brisc` o `tb_chest_xray`.

En datasets de imagen, `mlp` y `vgg` tratan los datos de forma distinta:

- Con `mlp`, las imágenes se aplanan como vectores.
- Con `vgg`, se conserva la estructura espacial como tensor de imagen.

Por ejemplo:

- MNIST con `mlp`: `28x28 -> 784`.
- MNIST con `vgg`: `1x28x28`.
- CIFAR-10 con `mlp`: `32x32x3 -> 3072`.
- CIFAR-10 con `vgg`: `3x32x32`.

## Datasets De Clasificación

El código soporta estos datasets de clasificación:

- `synthetic_multiclass`
- `iris`
- `wine`
- `digits`
- `breast_cancer`
- `mnist`
- `cifar10`
- `brisc`
- `tb_chest_xray`
- `dermatology`
- `heart_disease`

Los resultados finales actuales solo usan:

- `iris`
- `wine`
- `breast_cancer`
- `digits`
- `mnist`
- `cifar10`
- `brisc`
- `tb_chest_xray`

Por tanto, `synthetic_multiclass`, `dermatology` y `heart_disease` quedan como soporte adicional del código, no como parte de la línea final de resultados.

## Datasets De Regresión

El modo `regression` soporta:

- `synthetic_multiregression`
- `linnerud`
- `energy`

Estos datasets no forman parte de los resultados finales actuales. Se mantienen porque permiten estudiar la versión de la pregunta del TFM en problemas con varias salidas continuas.

## Datasets Sintéticos

El código incluye dos generadores sintéticos:

- `synthetic_multiclass`: clasificación multiclase artificial.
- `synthetic_multiregression`: regresión multi-output artificial.

Los parámetros principales son:

- `--synthetic-samples`: número de observaciones.
- `--synthetic-features`: número de variables de entrada.
- `--synthetic-classes`: número de clases en clasificación.
- `--synthetic-targets`: número de salidas en regresión.
- `--dependency-strength`: intensidad de una componente compartida entre salidas o clases.

Estos datasets son útiles para pruebas controladas, pero no están incluidos en los resultados finales de `resultados_10semillas/`.

## OVO

`OVO` está implementado como extensión para clasificación multiclase. Entrena un clasificador binario por cada par de clases y predice mediante votación.

No se usa como comparación central porque no responde directamente a la pregunta principal del TFM: comparar una red con `t` salidas frente a `t` redes independientes. En `OVA` sí hay una red por clase, por eso es la descomposición usada en los resultados finales.

## Regresión

La rama de regresión compara:

- `multi-output`: una red con una salida por variable objetivo.
- `decoupled`: una red independiente por variable objetivo.

Las métricas disponibles son:

- `mse`
- `mae`
- `r2`

Esta parte puede servir como extensión futura, pero no forma parte de los resultados finales actuales.

## Parámetros Generales

El parser admite, entre otros:

- `--hidden-layers`
- `--batch-normalization`
- `--batch-size`
- `--epochs`
- `--early-stopping-patience`
- `--early-stopping-min-delta`
- `--learning-rate`
- `--seed`
- `--seeds`
- `--max-train`
- `--max-test`
- `--image-size`

`--max-train` y `--max-test` son útiles para pruebas rápidas con datasets grandes.

## Scripts Auxiliares

Además de `run_experiments.py`, hay scripts auxiliares en `scripts/`:

- `estudio_tamano_imagenes.py`: resume los tamaños originales y relaciones de aspecto de las imágenes de BRISC y tuberculosis. Sirve para justificar el redimensionado común usado en los experimentos VGG.
- `power_analysis_paired.py`: estima el número de semillas necesarias para comparar `OVA` y `multi-output` a partir de resultados piloto.
- `wilcoxon_paired_test.py`: aplica un test pareado de Wilcoxon sobre una métrica concreta.
