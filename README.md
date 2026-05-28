# TFM - Output coupling in neural networks

Este repositorio contiene el código experimental del TFM sobre acoplamiento de salidas en redes neuronales. La comparación principal estudia si una única red con varias salidas se comporta de forma equivalente, o suficientemente parecida, a entrenar redes independientes para cada salida.

En clasificación, el caso estudiado es:

- `multi-output`: una única red con una salida por clase.
- `OVA`: una red binaria por clase, entrenada contra el resto de clases.

La comparación se interpreta de forma predictiva y funcional. No se asume equivalencia computacional: entrenar una red por clase suele ser más costoso que entrenar una única red multiclase. Por eso los resultados guardan tanto métricas predictivas como tiempo de entrenamiento, y el repositorio incluye un flujo separado para estimar el coste de OVA cuando sus clasificadores se entrenan en paralelo.

## Resultados

Los resultados nuevos de los experimentos se guardan localmente en `resultados_actualizados/`, pero esa carpeta no se versiona en GitHub. La tanda principal secuencial escribe en `resultados_actualizados/secuencial/`, los estudios OVA paralelos en `resultados_actualizados/paralelo/` y los análisis derivados en subcarpetas de `resultados_actualizados/`.

La base experimental usa 10 semillas por configuración. Se realizaron ampliaciones puntuales de semillas cuando el
análisis de potencia lo aconsejó para los contrastes estadísticos: Iris y BRISC cuentan con ejecuciones ampliadas
en `resultados_actualizados/ampliados/`. Estas ampliaciones se usan para Wilcoxon/TOST sobre métricas predictivas,
no para el análisis temporal. Cada ejecución detallada genera dos ficheros por dataset:

- `exp_*.csv`: resultados detallados por semilla y tipo de modelo.
- `exp_*_summary.csv`: medias agregadas por configuración.

Los CSV secuenciales conservan el identificador del trabajo SLURM en el nombre cuando proceden de los scripts de lanzamiento. `resultados_slurm/` queda como zona de trabajo heredada para scripts antiguos y no es la salida recomendada de la tanda final actual.

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

No se versionan datasets descargados, entornos locales, notebooks exploratorios, cachés de Python, logs de SLURM ni resultados experimentales locales como `resultados_actualizados/`, `resultados_10semillas/`, `resultados_estadisticos/` o `resultados_slurm/`.

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
  --coupling-modes multi-output ova \
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
  --coupling-modes multi-output ova \
  --epochs 50 \
  --early-stopping-patience 10 \
  --early-stopping-min-delta 1e-4 \
  --batch-size 64 \
  --output-csv exp_mnist_vgg.csv \
  --summary-csv exp_mnist_vgg_summary.csv
```

Si se trabaja en un servidor con SLURM, la tanda secuencial final se envía por dataset o grupo de datasets:

```bash
sbatch scripts/run_tfm_classic_slurm.sh
sbatch scripts/run_tfm_mnist_vgg_slurm.sh
sbatch scripts/run_tfm_cifar10_vgg_slurm.sh
sbatch scripts/run_tfm_brisc_vgg_slurm.sh
sbatch scripts/run_tfm_tb_vgg_slurm.sh
```

Los scripts secuenciales actualizados escriben sus salidas en `resultados_actualizados/secuencial/` y los logs en
`slurm_logs/`. Esas carpetas son locales; `RESULT_DIR` permite cambiar la carpeta de salida.

Para que los tiempos sean comparables, los scripts SLURM actuales fijan el nodo de referencia por dataset:

- `zeus`: clásicos MLP y BRISC.
- `atenea`: MNIST y CIFAR-10.
- `titan`: TB Chest X-ray.

Esta restricción es importante para análisis de tiempo y arquitecturas. Los experimentos ampliados de Iris y BRISC
se interpretan solo como refuerzo estadístico de rendimiento, por lo que no se usan para comparar tiempos.

### Protocolo de semillas

La comparación estadística usa un diseño pareado por repetición experimental. En cada repetición, la columna
`seed` identifica la semilla base que fija la generación o partición de datos y el entrenamiento comparado para
ambos enfoques.

Para clasificación, el protocolo actual es:

- `multi-output` entrena su red multiclase con la semilla `x` de la repetición.
- `OVA` reutiliza esa misma semilla `x` para cada clasificador binario de la repetición.
- El flujo OVA paralelo aplica la misma política de semillas que el flujo secuencial.

La semilla compartida reduce variabilidad accidental de inicialización y barajado entre los enfoques, pero no
convierte sus entrenamientos en idénticos: `multi-output` sigue siendo una red multiclase y `OVA` sigue siendo un
ensemble de redes binarias con otra función objetivo. Los tests se aplican sobre diferencias pareadas por semilla,
por ejemplo `f1_macro(OVA, x) - f1_macro(multi-output, x)`.

### OVA paralelo en SLURM

La comparación principal mantiene el flujo anterior de `run_experiments.py`. Para estudiar el coste de OVA en un
escenario paralelizable existe además un flujo separado por clases OVA:

```bash
HIDDEN_LAYERS="32 16" \
RUN_NAME="ova_parallel_32_16" \
bash scripts/submit_tfm_all_parallel_ova_slurm.sh
```

Por defecto ese lanzador incluye `iris`, `wine`, `breast_cancer` y `digits` con MLP, y `mnist`, `cifar10`,
`brisc` y `tb_chest_xray` con VGG. Crea un array SLURM por dataset con una tarea por clase OVA. Cada tarea recorre todas las semillas
solicitadas para esa clase y guarda CSV intermedios bajo `resultados_actualizados/paralelo/$RUN_NAME/artifacts/`: un
`class_*_summaries.csv` con los tiempos de esa clase por semilla y un `class_*_predictions.csv` con las
probabilidades de test que necesita la agregación. Cuando el array termina correctamente, un job dependiente
reconstruye el ensemble por semilla, escribe en `resultados_actualizados/paralelo/$RUN_NAME/` los CSV finales, incluido un CSV
conjunto `*_class_times.csv` con los tiempos por clase, y borra los CSV intermedios de ese dataset.

Para que los tiempos paralelos no mezclen GPUs de nodos distintos, el lanzador fija las tareas de entrenamiento
OVA al nodo secuencial de referencia de cada dataset. La tanda actual usa `zeus` para los clásicos y BRISC,
`atenea` para MNIST y CIFAR-10, y `titan` para TB.

```bash
RUN_NAME="ova_parallel_actual" \
bash scripts/submit_tfm_all_parallel_ova_slurm.sh
```

Los valores por dataset se pueden cambiar con `CLASSIC_PARALLEL_NODE`, `MNIST_PARALLEL_NODE`,
`CIFAR10_PARALLEL_NODE`, `BRISC_PARALLEL_NODE` y `TB_PARALLEL_NODE`. `PARALLEL_NODE` sigue disponible como
override global cuando se quiera forzar todos los paralelos a un mismo nodo.

Esto fija el hardware de los clasificadores OVA paralelos. No implica que todas las clases se ejecuten a la vez:
la partición actual ofrece como máximo 4 GPUs por nodo, de modo que datasets con 10 clases como `digits`,
`mnist` o `cifar10` se planifican por tandas en ese mismo nodo.

Los CSV agregados conservan `train_time_seconds` como suma de los tiempos de las redes OVA, equivalente al coste
secuencial acumulado. Añaden `parallel_train_time_seconds`, que toma el máximo de los tiempos por clase para estimar
el tiempo paralelo ideal si las redes se entrenan simultáneamente con recursos suficientes, y
`ova_model_train_time_seconds_mean`, que resume el coste medio por clasificador.

El script `scripts/analisis_tiempos_paralelo_ova.py` cruza los CSV secuenciales con los CSV agregados del flujo
paralelo y produce detalle por semilla y resumen por dataset. Calcula la aceleración estimada de OVA al pasar del
entrenamiento secuencial al paralelo ideal y la razón entre ese tiempo ideal de OVA y el tiempo de `multi-output`.
La tabla final conserva solo las columnas interpretables para el informe: tiempo medio de `multi-output`, tiempo
medio de OVA secuencial, tiempo medio de OVA paralelo ideal, aceleración ideal y razón frente a `multi-output`.

### Arquitecturas OVA reducidas

Además de la comparación principal, el repositorio incluye un estudio de arquitecturas reducidas para comprobar si
OVA mantiene rendimiento al usar modelos más pequeños.

Para datasets clásicos MLP se comparan:

- referencia `[32, 16]`, procedente de la tanda secuencial principal;
- OVA `[24, 12]`;
- OVA `[16, 8]`.

Los scripts correspondientes son:

```bash
sbatch scripts/run_tfm_classic_OVA_24_12_slurm.sh
sbatch scripts/run_tfm_classic_OVA_16_8_slurm.sh
```

Para datasets VGG se pueden lanzar arquitecturas reducidas cambiando `VGG_CHANNELS` y escribiendo en una carpeta
separada para no pisar los resultados base. Por ejemplo:

```bash
RESULT_DIR=resultados_actualizados/arquitecturas_ova_vgg/vgg_24_48_96 \
COUPLING_MODES=ova \
VGG_CHANNELS="24 48 96" \
sbatch scripts/run_tfm_mnist_vgg_slurm.sh

RESULT_DIR=resultados_actualizados/arquitecturas_ova_vgg/vgg_16_32_64 \
COUPLING_MODES=ova \
VGG_CHANNELS="16 32 64" \
sbatch scripts/run_tfm_mnist_vgg_slurm.sh
```

El mismo patrón se aplica a `run_tfm_cifar10_vgg_slurm.sh`, `run_tfm_brisc_vgg_slurm.sh` y
`run_tfm_tb_vgg_slurm.sh`. El análisis conjunto se genera con:

```bash
python scripts/analisis_arquitecturas_ova.py
```

Este script espera que existan las dos arquitecturas reducidas para todos los datasets incluidos. Para VGG busca los
CSV por patrón dentro de `resultados_actualizados/arquitecturas_ova_vgg/vgg_24_48_96/` y
`resultados_actualizados/arquitecturas_ova_vgg/vgg_16_32_64/`.

### Tandas recientes de eficiencia OVA

Tras comprobar la equivalencia práctica entre `multi-output` y `OVA` en varios datasets, se añadieron tandas
centradas en el principal inconveniente de OVA: el coste de entrenar un clasificador por clase. Estas tandas no
proponen un algoritmo nuevo, sino que evalúan si el coste adicional puede mitigarse mediante entrenamiento paralelo,
arquitecturas reducidas y ajustes simples de entrenamiento sin perder rendimiento predictivo.

Las carpetas relevantes son:

- `resultados_actualizados/paralelo/ova_vgg32_64_128_bs128_lr1e3_pat10_ep50/`: MNIST y CIFAR-10 con VGG
  `[32, 64, 128]`, `batch_size=128`, `epochs=50`, `patience=10` y `learning_rate=0.001`.
- `resultados_actualizados/paralelo/ova_tb_vgg16_32_64_bs64_lr1e3_pat10_ep50/`: TB con VGG reducido
  `[16, 32, 64]`, `batch_size=64`, `epochs=50`, `patience=10` y `learning_rate=0.001`.
- `resultados_actualizados/paralelo/ova_tb_vgg16_32_64_lr3e3_pat5_ep30_bs64/`: TB con configuración más agresiva
  (`batch_size=64`, `epochs=30`, `patience=5`, `learning_rate=0.003`).
- `resultados_actualizados/paralelo/ova_brisc_vgg24_48_96_bs32_lr1e3_pat10_ep50/` y
  `resultados_actualizados/paralelo/ova_brisc_vgg24_48_96_bs64_lr1e3_pat10_ep50/`: BRISC con VGG reducido
  `[24, 48, 96]`.
- `resultados_actualizados/paralelo/ova_brisc_vgg32_64_128_bs64_lr1e3_pat10_ep50/` y
  `resultados_actualizados/paralelo/ova_brisc_vgg32_64_128_bs32_lr1e3_pat7_ep50/`: BRISC con VGG original
  `[32, 64, 128]`, variando `batch_size` o `patience`.
- `resultados_actualizados/paralelo/ova_digits_mlp32_16_bs64_lr1e3_pat10_ep50/` y
  `resultados_actualizados/paralelo/ova_digits_mlp16_8_bs64_lr1e3_pat10_ep50/`: `digits` con MLP y
  `batch_size=64`.
- `resultados_actualizados/paralelo/ova_classic_mlp32_16_bs64_lr1e3_pat10_ep50/`: `iris`, `wine` y
  `breast_cancer` con MLP `[32, 16]` y `batch_size=64`.

La comparación temporal usa `parallel_train_time_seconds` para OVA y `train_time_seconds` para `multi-output`.
Los resultados más relevantes hasta ahora son:

| Dataset | Configuración OVA destacada | `balanced_accuracy` OVA | `f1_macro` OVA | Tiempo OVA paralelo | `balanced_accuracy` multi | `f1_macro` multi | Tiempo multi |
|---|---|---:|---:|---:|---:|---:|---:|
| TB Chest X-ray | VGG `[16, 32, 64]`, batch 32, patience 10 | 0.9689 | 0.9737 | 73.04 s | 0.9623 | 0.9698 | 106.38 s |
| MNIST | VGG `[32, 64, 128]`, batch 128, patience 10 | 0.9937 | 0.9937 | 109.76 s | 0.9929 | 0.9930 | 112.84 s |
| CIFAR-10 | VGG `[32, 64, 128]`, batch 128, patience 10 | 0.7669 | 0.7662 | 109.61 s | 0.7677 | 0.7663 | 128.00 s |
| BRISC | VGG `[32, 64, 128]`, batch 32, patience 7 | 0.9291 | 0.9205 | 134.31 s | 0.9177 | 0.9054 | 135.96 s |
| BRISC | VGG `[32, 64, 128]`, batch 32, patience 10 | 0.9379 | 0.9306 | 137.91 s | 0.9177 | 0.9054 | 135.96 s |
| Digits | MLP `[32, 16]`, batch 64, patience 10 | 0.9799 | 0.9799 | 3.06 s | 0.9636 | 0.9635 | 3.02 s |

Estos resultados apoyan una interpretación matizada. OVA no elimina su coste computacional: el coste secuencial
acumulado sigue siendo mayor porque se entrena una red por clase. Sin embargo, cuando se estima el escenario
paralelo por clase, el coste efectivo puede ser comparable o menor que el de `multi-output` en varios datasets,
manteniendo equivalencia o mejora predictiva. Esto es claro en TB, MNIST, BRISC con `patience=7` y `digits`.
CIFAR-10 queda como caso de rendimiento prácticamente equivalente con menor tiempo paralelo.

En datasets pequeños como `iris`, `wine` y `breast_cancer`, el beneficio temporal es menos estable. El entrenamiento
multi-output ya dura menos de unos pocos segundos, por lo que el overhead de lanzar varios clasificadores OVA pesa
más. En esos casos OVA mejora métricas con `batch_size=32`, pero en la tanda con `batch_size=64` se observa una
degradación del rendimiento en `iris` y `wine`. Esta comparación puede verse en
`resultados_actualizados/paralelo/ova_parallel_actual/exp_iris_mlp_parallel_ova_summary.csv`,
`resultados_actualizados/paralelo/ova_parallel_actual/exp_wine_mlp_parallel_ova_summary.csv`,
`resultados_actualizados/paralelo/ova_classic_mlp32_16_bs64_lr1e3_pat10_ep50/exp_iris_mlp_parallel_ova_summary.csv`
y
`resultados_actualizados/paralelo/ova_classic_mlp32_16_bs64_lr1e3_pat10_ep50/exp_wine_mlp_parallel_ova_summary.csv`.
Por tanto, estas tandas se interpretan como análisis de sensibilidad, no como configuraciones principales.

La conclusión metodológica es que la hipótesis inicial de equivalencia predictiva se complementa con un análisis de
eficiencia: si OVA alcanza rendimiento equivalente o superior, su coste adicional puede reducirse mediante
paralelización, reducción arquitectónica o ajustes de entrenamiento. El efecto no es universal y depende del dataset,
del número de clases, del tamaño del conjunto y de la dificultad del problema.

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

Los resultados OVA agregados desde el flujo paralelo añaden:

- `parallel_train_time_seconds`
- `ova_model_train_time_seconds_mean`

También se guardan variables asociadas al `early stopping`, como `best_val_loss`, `epochs_trained`, `total_epochs_trained` y `models_stopped_early`.

Estas métricas permiten comparar no solo el porcentaje global de aciertos, sino también el comportamiento por clase y el coste computacional de entrenar una red multiclase frente a varias redes binarias.

## Análisis Estadístico

Los scripts auxiliares de `scripts/` incluyen análisis de potencia y pruebas estadísticas sobre las diferencias pareadas `OVA - multi-output`:

- `analisis_potencia_wilcoxon.py`: estima por simulación Monte Carlo el número de semillas necesario para que Wilcoxon detecte una diferencia relevante.
- `test_wilcoxon_pareado.py`: aplica el test pareado de Wilcoxon a uno o varios CSV.
- `analisis_potencia_tost_bootstrap.py`: estima por simulación Monte Carlo el número de semillas necesario para declarar equivalencia práctica mediante IC bootstrap de la mediana.
- `test_equivalencia_tost_bootstrap.py`: aplica el análisis de equivalencia TOST/bootstrap usando un margen práctico, por defecto `±0.02` en `f1_macro`.
- `analisis_tiempos_paralelo_ova.py`: compara tiempos de `multi-output`, OVA secuencial y OVA paralelo ideal.
- `analisis_arquitecturas_ova.py`: resume variantes reducidas de arquitectura OVA para clásicos MLP y datasets VGG.

Wilcoxon evalúa evidencia de diferencia estadística. El análisis TOST/bootstrap evalúa equivalencia práctica dentro de un margen definido. Ambos enfoques se interpretan de forma complementaria.

Varios scripts conservan rutas por defecto a CSV concretos de ejecuciones locales. Si se regeneran los
experimentos, conviene pasar explícitamente los nuevos CSV cuando el script lo permita o actualizar esas rutas de
trabajo antes de generar las tablas finales.
