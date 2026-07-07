# Capacidades del código

Este documento resume qué permite ejecutar el código del repositorio y separa
la línea usada en el TFM de extensiones exploratorias que quedaron
implementadas.

## Línea principal del TFM

El foco del trabajo es la comparación en clasificación supervisada entre:

- `multi-output`: una única red con una salida por clase;
- `ova`: una red binaria independiente por clase frente al resto.

La comparación principal usa rendimiento predictivo, coste de entrenamiento,
equivalencia práctica y, en conjuntos biomédicos, explicabilidad visual.

## Tareas soportadas

El argumento `--task` admite:

- `classification`: modo principal utilizado en el TFM.
- `regression`: soporte exploratorio para problemas con varias salidas
  continuas.

En clasificación, `--coupling-modes multi-output ova` ejecuta la comparación
central del TFM. El modo `ovo` también está implementado, pero no forma parte de
los resultados finales. En regresión, el modo `decoupled` entrena una red
independiente por variable objetivo.

## Arquitecturas

El argumento `--model-arch` soporta:

- `mlp`: red fully connected configurable con `--hidden-layers`.
- `vgg`: VGG compacta para MNIST, CIFAR-10, BRISC y TB Chest X-ray.
- `vgg16-pretrained`: VGG16 preentrenada en ImageNet, usada en HAM10000.
- `vit-b-16-pretrained`: ViT-B/16 preentrenado, usado solo como referencia
  exploratoria.

En datasets de imagen, `mlp` aplana la imagen y `vgg` conserva su estructura
espacial como tensor.

## Datasets

Resultados principales:

- `iris`
- `wine`
- `breast_cancer`
- `digits`
- `mnist`
- `cifar10`
- `brisc`
- `tb_chest_xray`

Extensiones o soporte adicional:

- `ham10000`: bloque complementario de explicabilidad visual.
- `synthetic_multiclass`: pruebas controladas de clasificación.
- `synthetic_multiregression`: pruebas controladas de regresión.
- `dermatology` y `heart_disease`: soporte exploratorio mediante `ucimlrepo`.
- `linnerud` y `energy`: soporte exploratorio de regresión.

`ucimlrepo` no se incluye en `requirements.txt` porque esos datasets no se
usaron en los resultados finales.

## HAM10000

HAM10000 se incorpora como extensión dermatológica para el bloque de
explicabilidad. El código soporta:

- split interno por `lesion_id`, evitando que imágenes de una misma lesión
  aparezcan en particiones distintas;
- evaluación opcional con el test oficial de ISIC 2018 Task 3;
- máscaras de lesión para el análisis espacial cuando se usa el test interno;
- VGG compacta, VGG16 preentrenada y ViT-B/16 preentrenado;
- `--class-weighting balanced`;
- `--train-sampler balanced`;
- `--data-augmentation ham10000-basic`;
- `--ova-loss focal`;
- `--ova-calibration platt|threshold|threshold-f1`;
- `--ham10000-label-mode malignant_binary`.

En la memoria, HAM10000 no sustituye al protocolo principal. Se usa para
estudiar si las diferencias entre multi-salida y OVA también aparecen en un
escenario dermatológico multiclase, desbalanceado y con máscaras de lesión.

## Parámetros útiles

Algunos argumentos relevantes:

- `--coupling-modes multi-output ova`
- `--hidden-layers`
- `--vgg-channels`
- `--batch-size`
- `--epochs`
- `--early-stopping-patience`
- `--learning-rate`
- `--class-weighting none|balanced`
- `--train-sampler none|balanced`
- `--ova-loss bce|focal`
- `--pretrained-finetune frozen|block5|last-block|full`
- `--image-size`
- `--predictions-csv`

## Scripts auxiliares

Además de `run_experiments.py` y `run_parallel_ova.py`, el repositorio contiene
scripts para:

- análisis estadístico: Wilcoxon, TOST/bootstrap y potencia;
- análisis de tiempo secuencial/paralelo de OVA;
- selección de configuraciones rendimiento-tiempo;
- Grad-CAM, Grad-CAM++, LRP y oclusión;
- análisis de señales contextuales en BRISC;
- análisis de artefactos y jerarquía maligno/no maligno en HAM10000;
- generación de figuras y tablas para la memoria.

Los resultados, checkpoints, figuras derivadas y CSV experimentales no se
versionan.
