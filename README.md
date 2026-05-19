# TFM - Output coupling in neural networks

Este repositorio contiene el código experimental del TFM sobre acoplamiento de salidas en redes neuronales. La comparación principal estudia si una única red con varias salidas se comporta de forma equivalente, o suficientemente parecida, a entrenar redes independientes para cada salida.

En clasificación, el caso estudiado es:

- `multi-output`: una única red con una salida por clase.
- `OVA`: una red binaria por clase, entrenada contra el resto de clases.

La comparación se interpreta de forma predictiva y funcional. No se asume equivalencia computacional: entrenar una red por clase suele ser más costoso que entrenar una única red multiclase. Por eso los resultados guardan tanto métricas predictivas como tiempo de entrenamiento.

## Resultados

Los resultados detallados de los experimentos se guardan localmente en `resultados_10semillas/`, pero esa carpeta no se versiona en GitHub. Contiene los CSV generados a partir de las ejecuciones experimentales y se mantiene como material de trabajo local.

La base experimental usa 10 semillas por configuración, con ampliaciones puntuales cuando el análisis estadístico lo aconseja o cuando ya se dispone de ejecuciones adicionales. En la versión local, cada dataset puede tener dos ficheros:

- `*_10semillas.csv`: resultados detallados por semilla y tipo de modelo.
- `*_10semillas_summary.csv`: medias agregadas por configuración.

Algunos datasets tienen además versiones con más semillas, por ejemplo `iris_mlp_22semillas.csv` y `brisc_vgg_128_24semillas.csv`. Los CSV proceden de ejecuciones en SLURM, pero se copian localmente a `resultados_10semillas/` con nombres estables y sin identificadores internos de trabajos. La carpeta local `resultados_slurm/` queda como zona de trabajo y tampoco se versiona.

## Experimentos Actuales

Los experimentos versionados cubren:

- `iris`, `wine`, `breast_cancer` y `digits` con MLP.
- `mnist` y `cifar10` con VGG compacta.
- `brisc` y `tb_chest_xray` con VGG compacta e imágenes redimensionadas a `128x128`.

El foco actual del repositorio es clasificación con `multi-output` frente a `OVA`. El código conserva algunas extensiones exploratorias, pero no forman parte de los resultados finales incluidos aquí.

Para ver esas capacidades adicionales del código, como regresión, datasets sintéticos, `OVO` o datasets no usados en los resultados actuales, consultar `docs/capacidades_codigo.md`.

## Estructura

```text
.
|-- run_experiments.py              # Entrada principal para lanzar experimentos
|-- tfm/                            # Paquete con carga de datos, modelos, entrenamiento y métricas
|-- docs/                           # Notas sobre capacidades adicionales del código
|-- scripts/                        # Scripts de análisis auxiliares
|-- requirements.txt                # Dependencias Python principales
`-- .gitignore                      # Archivos locales excluidos de Git
```

No se versionan datasets descargados, entornos locales, notebooks exploratorios, cachés de Python, logs de SLURM ni resultados experimentales locales como `resultados_10semillas/`, `resultados_estadisticos/` o `resultados_slurm/`.

## Instalación

```bash
pip install -r requirements.txt
```

Los datasets grandes se mantienen en local dentro de `data/`. Esa carpeta está ignorada por Git.

## Ejecución

Ejemplo con un dataset tabular:

```bash
python run_experiments.py \
  --task classification \
  --dataset wine \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --coupling-modes ova \
  --output-csv exp_wine.csv \
  --summary-csv exp_wine_summary.csv
```

Ejemplo con MNIST usando VGG compacta:

```bash
python run_experiments.py \
  --task classification \
  --dataset mnist \
  --model-arch vgg \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --coupling-modes ova \
  --epochs 50 \
  --early-stopping-patience 10 \
  --early-stopping-min-delta 1e-4 \
  --batch-size 64 \
  --output-csv exp_mnist_vgg.csv \
  --summary-csv exp_mnist_vgg_summary.csv
```

Si se trabaja en un servidor con SLURM, los experimentos pueden lanzarse mediante scripts locales de tipo `scripts/*_slurm.sh`. Esos scripts no se versionan porque dependen de la configuración concreta del servidor. Un ejemplo de uso sería:

```bash
sbatch scripts/run_tfm_classic_slurm.sh
sbatch scripts/run_tfm_mnist_vgg_slurm.sh
sbatch scripts/run_tfm_cifar10_vgg_slurm.sh
sbatch scripts/run_tfm_brisc_vgg_slurm.sh
sbatch scripts/run_tfm_tb_vgg_slurm.sh
```

Los scripts escriben sus salidas en `resultados_slurm/` y los logs en `slurm_logs/`. Ambas carpetas son locales.

## Datasets Locales

BRISC debe estar descargado localmente con esta estructura:

```text
data/brisc2025/train/
data/brisc2025/test/
```

Cada partición debe contener las carpetas `glioma`, `meningioma`, `pituitary` y `no_tumor`.

Tuberculosis debe estar en:

```text
data/tb_chest_xray/
```

con las carpetas `Normal` y `Tuberculosis`.

MNIST y CIFAR-10 se descargan o leen desde `data/` durante la carga de datos.

## Métricas

En clasificación se guardan:

- `accuracy`
- `balanced_accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `tpr_macro`
- `fpr_macro`
- `tnr_macro`
- `fnr_macro`
- `train_time_seconds`

También se guardan variables asociadas al `early stopping`, como `best_val_loss`, `epochs_trained`, `total_epochs_trained` y `models_stopped_early`.

Estas métricas permiten comparar no solo el porcentaje global de aciertos, sino también el comportamiento por clase y el coste computacional de entrenar una red multiclase frente a varias redes binarias.

## Análisis Estadístico

Los scripts auxiliares de `scripts/` incluyen análisis de potencia y pruebas estadísticas sobre las diferencias pareadas `OVA - multi-output`:

- `analisis_potencia_wilcoxon.py`: estima por simulación Monte Carlo el número de semillas necesario para que Wilcoxon detecte una diferencia relevante.
- `test_wilcoxon_pareado.py`: aplica el test pareado de Wilcoxon a uno o varios CSV.
- `analisis_potencia_tost_bootstrap.py`: estima por simulación Monte Carlo el número de semillas necesario para declarar equivalencia práctica mediante IC bootstrap de la mediana.
- `test_equivalencia_tost_bootstrap.py`: aplica el análisis de equivalencia TOST/bootstrap usando un margen práctico, por defecto `±0.02` en `f1_macro`.

Wilcoxon evalúa evidencia de diferencia estadística. El análisis TOST/bootstrap evalúa equivalencia práctica dentro de un margen definido. Ambos enfoques se interpretan de forma complementaria.

Por defecto, estos scripts esperan encontrar los CSV experimentales en la carpeta local `resultados_10semillas/`. Si se clona el repositorio en otra máquina, esos CSV deben copiarse desde el servidor de trabajo o regenerarse ejecutando los experimentos.
