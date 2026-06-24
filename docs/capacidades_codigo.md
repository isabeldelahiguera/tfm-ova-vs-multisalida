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
- `ham10000`
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

### HAM10000

`ham10000` está preparado como extensión dermatológica para clasificación con imágenes RGB y explicabilidad espacial.
Las imágenes originales tienen tamaño `600x450` y se redimensionan con `--image-size`, igual que el resto de datasets
de imagen.

El dataset descargado incluye:

- `HAM10000_images_part_1.zip` y `HAM10000_images_part_2.zip`: 10015 imágenes de HAM10000.
- `HAM10000_metadata`: etiquetas diagnósticas y metadatos.
- `HAM10000_segmentations_lesion_tschandl.zip`: máscaras de lesión para las 10015 imágenes de HAM10000.
- `ISIC2018_Task3_Test_Images.zip` y `ISIC2018_Task3_Test_GroundTruth.csv`: test oficial independiente de clasificación
  de ISIC 2018 Task 3.

El código soporta dos modos:

- `--ham10000-test internal`: modo recomendado para las primeras pruebas y para explicabilidad, porque mantiene
  coherencia con las máscaras disponibles. Si no existe, crea un holdout fijo `train/test` por `lesion_id` en
  `data/ham10000/ham10000_train_test_split_seed2000.csv`. En cada ejecución, la parte `train` se divide después en
  `train/val` por `lesion_id`, de forma equivalente al esquema de BRISC: test fijo separado y validación derivada
  del train.
- `--ham10000-test official`: usa HAM10000 como `train/val` y evalúa en el test independiente de ISIC 2018 Task 3.
  Este modo existe para evaluación predictiva externa, pero no se usa en las primeras pruebas si se quiere mantener
  el mismo test con máscaras para rendimiento y explicabilidad.

En ambos casos, las particiones derivadas de HAM10000 se hacen agrupando por `lesion_id`: todas las imágenes de una
misma lesión quedan en el mismo subconjunto y se evita fuga de información entre train, validación y test.

El lanzador SLURM para las pruebas VGG de HAM10000 es `scripts/run_tfm_ham10000_vgg_slurm.sh`. Además de la VGG propia
desde cero, el código soporta `--model-arch vgg16-pretrained`, con pesos ImageNet y tres políticas de ajuste:
`--pretrained-finetune frozen`, `block5` o `full`. Para HAM10000 también están disponibles `--class-weighting balanced`,
`--data-augmentation ham10000-basic` y `--ova-calibration platt`; la calibración de OVA se considera análisis auxiliar,
no la comparación principal.

El análisis Grad-CAM de HAM10000 se lanza con `scripts/run_explicabilidad_gradcam_slurm.sh`. Usa las máscaras de
`HAM10000_segmentations_lesion_tschandl` cuando se trabaja con `--ham10000-test internal`, por lo que permite comparar
rendimiento predictivo y concentración espacial de la explicación sobre el mismo test interno. Para comparaciones de
tiempo o explicabilidad se recomienda fijar `atenea` con `sbatch --nodelist=atenea`.

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
- `--class-weighting`: `none` o `balanced`. En clasificación multiclase usa pesos por clase en
  `CrossEntropyLoss`; en `OVA` usa `pos_weight` en cada pérdida binaria.
- `--seed`
- `--seeds`
- `--max-train`
- `--max-test`
- `--image-size`

`--max-train` y `--max-test` son útiles para pruebas rápidas con datasets grandes.

## Scripts Auxiliares

Además de `run_experiments.py`, hay scripts auxiliares en `scripts/`:

- `estudio_size_imagenes.py`: resume los tamaños originales y relaciones de aspecto de las imágenes de BRISC y tuberculosis. Sirve para justificar el redimensionado común usado en los experimentos VGG.
- `analisis_potencia_wilcoxon.py`: estima por simulación Monte Carlo el número de semillas necesarias para que el test de Wilcoxon detecte una diferencia relevante entre `OVA` y `multi-output`.
- `test_wilcoxon_pareado.py`: aplica el test pareado de Wilcoxon sobre una métrica concreta.
- `analisis_potencia_tost_bootstrap.py`: estima por simulación Monte Carlo el número de semillas necesarias para declarar equivalencia mediante un IC bootstrap de la mediana dentro de un margen práctico.
- `test_equivalencia_tost_bootstrap.py`: aplica el análisis de equivalencia TOST/bootstrap usando un IC bootstrap de la mediana de las diferencias pareadas.

Los análisis de potencia se usan como referencia para valorar si el número de semillas es suficiente para detectar diferencias relevantes o declarar equivalencia práctica. Los tests finales se interpretan sobre las diferencias pareadas observadas en cada CSV.
