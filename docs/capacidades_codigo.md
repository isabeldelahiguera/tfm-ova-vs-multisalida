# Capacidades del código

Este documento recoge funcionalidades que el código soporta, distinguiendo entre la línea principal del TFM y extensiones exploratorias. Los resultados experimentales se generan localmente en `resultados_actualizados/`, pero no se versionan en GitHub.

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
- `vgg16-pretrained`: VGG16 preentrenada en ImageNet para imágenes RGB, usada en pruebas dermatológicas de HAM10000.
- `vit-b-16-pretrained`: ViT-B/16 preentrenada en ImageNet. Queda como soporte exploratorio, no como configuración final del TFM.

En datasets de imagen, `mlp` y `vgg` tratan los datos de forma distinta:

- Con `mlp`, las imágenes se aplanan como vectores.
- Con `vgg`, se conserva la estructura espacial como tensor de imagen.

Por ejemplo:

- MNIST con `mlp`: `28x28 -> 784`.
- MNIST con `vgg`: `1x28x28`.
- CIFAR-10 con `mlp`: `32x32x3 -> 3072`.
- CIFAR-10 con `vgg`: `3x32x32`.

## Datasets de clasificación

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

`ham10000` se usa como extensión dermatológica y análisis adicional, no como eje de la comparación final. `synthetic_multiclass`, `dermatology` y `heart_disease` quedan como soporte adicional del código.

`dermatology` y `heart_disease` dependen de `ucimlrepo`. Esa dependencia no está en `requirements.txt` porque esos datasets no se usaron en los resultados principales; si se quieren ejecutar, instalarla aparte:

```bash
pip install ucimlrepo
```

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

El código soporta dos modos de test:

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
`--pretrained-finetune frozen`, `block5` o `full`. También existe soporte para `vit-b-16-pretrained`, aunque no se
usa como configuración final.

Para HAM10000 están disponibles:

- `--class-weighting balanced`: pesos por clase en la pérdida.
- `--train-sampler balanced`: muestreo balanceado en entrenamiento.
- `--data-augmentation ham10000-basic`: aumentos sencillos de imagen.
- `--ova-loss focal`: focal loss para clasificadores OVA.
- `--ova-calibration platt|threshold|threshold-f1`: calibración o selección auxiliar de umbrales OVA.
- `--ham10000-exclude-classes`: exclusión de clases concretas.
- `--ham10000-label-mode malignant_binary`: formulación binaria maligno/no maligno.

Estas opciones se añadieron para explorar el desbalance y posibles atajos visuales del dataset. No sustituyen la
comparación principal multi-output frente a OVA.

El análisis Grad-CAM de HAM10000 se lanza con `scripts/run_explicabilidad_gradcam_slurm.sh`. Usa las máscaras de
`HAM10000_segmentations_lesion_tschandl` cuando se trabaja con `--ham10000-test internal`, por lo que permite comparar
rendimiento predictivo y concentración espacial de la explicación sobre el mismo test interno. Para comparaciones de
tiempo o explicabilidad se recomienda fijar `atenea` con `sbatch --nodelist=atenea`.

## Datasets de regresión

El modo `regression` soporta:

- `synthetic_multiregression`
- `linnerud`
- `energy`

Estos datasets no forman parte de los resultados finales actuales. Se mantienen porque permiten estudiar la versión de la pregunta del TFM en problemas con varias salidas continuas.

## Datasets sintéticos

El código incluye dos generadores sintéticos:

- `synthetic_multiclass`: clasificación multiclase artificial.
- `synthetic_multiregression`: regresión multi-output artificial.

Los parámetros principales son:

- `--synthetic-samples`: número de observaciones.
- `--synthetic-features`: número de variables de entrada.
- `--synthetic-classes`: número de clases en clasificación.
- `--synthetic-targets`: número de salidas en regresión.
- `--dependency-strength`: intensidad de una componente compartida entre salidas o clases.

Estos datasets son útiles para pruebas controladas, pero no están incluidos en los resultados finales del TFM.

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
- `--ova-loss`: `bce` o `focal`, para la pérdida de los clasificadores OVA.
- `--focal-gamma` y `--focal-alpha`: parámetros de focal loss.
- `--train-sampler`: `none` o `balanced`, para balancear el muestreo de entrenamiento.
- `--pretrained-finetune`: `frozen`, `block5`, `last-block` o `full`, según arquitectura.
- `--ova-calibration`: `none`, `platt`, `threshold` o `threshold-f1`, usado como análisis auxiliar de OVA.
- `--seed`
- `--seeds`
- `--max-train`
- `--max-test`
- `--image-size`
- `--predictions-csv`: guarda predicciones por muestra para análisis de errores y shortcuts.

`--max-train` y `--max-test` son útiles para pruebas rápidas con datasets grandes.

## Scripts Auxiliares

Además de `run_experiments.py`, hay scripts auxiliares en `scripts/`:

- `estudio_size_imagenes.py`: resume los tamaños originales y relaciones de aspecto de las imágenes de BRISC y tuberculosis. Sirve para justificar el redimensionado común usado en los experimentos VGG.
- `analisis_potencia_wilcoxon.py`: estima por simulación Monte Carlo el número de semillas necesarias para que el test de Wilcoxon detecte una diferencia relevante entre `OVA` y `multi-output`.
- `test_wilcoxon_pareado.py`: aplica el test pareado de Wilcoxon sobre una métrica concreta.
- `analisis_potencia_tost_bootstrap.py`: estima por simulación Monte Carlo el número de semillas necesarias para declarar equivalencia mediante un IC bootstrap de la mediana dentro de un margen práctico.
- `test_equivalencia_tost_bootstrap.py`: aplica el análisis de equivalencia TOST/bootstrap usando un IC bootstrap de la mediana de las diferencias pareadas.
- `analisis_tiempos_paralelo_ova.py`: compara tiempo secuencial, tiempo OVA acumulado y tiempo OVA paralelo ideal.
- `analisis_arquitecturas_ova.py`: resume pruebas con arquitecturas OVA reducidas.
- `analisis_configuraciones_seleccionadas_estadistico.py`: contrasta configuraciones finales seleccionadas.
- `figuras_tfm_configuraciones_seleccionadas.py` y `figuras_tfm_delta_sensibilidad_configuraciones.py`: generan figuras para la memoria.
- `explicabilidad_gradcam_vgg.py`, `explicabilidad_lrp_vgg.py` y `oclusion_tumor_brisc.py`: auditoría de explicabilidad.
- `analisis_dataset_brisc_train_test.py`, `resumir_shortcuts_brisc.py` y `analisis_errores_dataset_explicabilidad.py`: análisis de posibles shortcuts en BRISC.
- `analisis_atajos_ham10000.py` y `analisis_atajos_ham10000_artifacts.py`: análisis exploratorio de atajos visuales en HAM10000.

Los análisis de potencia se usan como referencia para valorar si el número de semillas es suficiente para detectar diferencias relevantes o declarar equivalencia práctica. Los tests finales se interpretan sobre las diferencias pareadas observadas en cada CSV.
