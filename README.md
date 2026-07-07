# TFM - OVA frente a redes multi-salida

Repositorio con el código experimental del Trabajo Fin de Máster sobre la
comparación entre modelos neuronales multi-salida y descomposiciones
One-vs-All (OVA) en problemas de clasificación supervisada.

La pregunta central es si una red con una salida por clase puede comportarse,
en la práctica, de forma comparable a entrenar un clasificador binario
independiente por clase. La comparación se analiza desde tres perspectivas:

- rendimiento predictivo y equivalencia práctica;
- coste de entrenamiento secuencial y paralelo;
- explicabilidad visual en conjuntos biomédicos de imagen.

## Estructura del repositorio

```text
.
|-- run_experiments.py          # Experimentos multi-salida y OVA secuencial
|-- run_parallel_ova.py         # Entrenamiento/agregación OVA por clase
|-- tfm/                        # Paquete principal
|   |-- cli.py                  # Argumentos de línea de comandos
|   |-- data.py                 # Carga y particiones de datos
|   |-- models.py               # MLP, VGG compacta, VGG16 y ViT
|   |-- training.py             # Entrenamiento y early stopping
|   |-- experiment.py           # Flujo experimental principal
|   |-- parallel_ova.py         # Flujo OVA paralelizable
|   |-- metrics.py              # Métricas de evaluación
|   `-- evaluation.py           # Utilidades de evaluación
|-- scripts/                    # Análisis estadístico, explicabilidad y figuras
|-- docs/                       # Documentación complementaria
`-- requirements.txt            # Dependencias principales
```

No se versionan datasets, checkpoints, resultados pesados, logs ni la memoria
en LaTeX.

## Instalación

```bash
pip install -r requirements.txt
```

Las dependencias principales incluyen `torch`, `torchvision`, `numpy`,
`pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn` y `zennit`.

## Conjuntos utilizados

La comparación principal del TFM utiliza:

- conjuntos tabulares clásicos: Iris, Wine, Breast Cancer y Digits;
- conjuntos de imagen estándar: MNIST y CIFAR-10;
- conjuntos biomédicos de imagen: BRISC y TB Chest X-ray.

HAM10000 se utiliza como conjunto complementario para el bloque de
explicabilidad visual.

## Ejecución básica

Ejemplo con Wine:

```bash
python run_experiments.py \
  --task classification \
  --dataset wine \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --coupling-modes multi-output ova \
  --output-csv resultados_actualizados/secuencial/exp_wine.csv \
  --summary-csv resultados_actualizados/secuencial/exp_wine_summary.csv
```

Ejemplo con MNIST y VGG compacta:

```bash
python run_experiments.py \
  --task classification \
  --dataset mnist \
  --model-arch vgg \
  --image-size 128 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --coupling-modes multi-output ova \
  --epochs 50 \
  --early-stopping-patience 10 \
  --batch-size 64 \
  --output-csv resultados_actualizados/secuencial/exp_mnist_vgg.csv \
  --summary-csv resultados_actualizados/secuencial/exp_mnist_vgg_summary.csv
```

Las métricas principales guardadas por ejecución son `accuracy`,
`balanced_accuracy`, `precision_macro`, `recall_macro`, `f1_macro` y
`train_time_seconds`.

## OVA paralelo

El flujo paralelo de OVA se ejecuta con `run_parallel_ova.py`. La idea es
entrenar los clasificadores binarios por clase de forma independiente y agregar
después sus predicciones.

El tiempo paralelo efectivo usado en los análisis se define como el máximo de
los tiempos de entrenamiento de los clasificadores binarios de una misma
semilla. Por tanto, representa un escenario ideal en el que las clases pueden
entrenarse simultáneamente con recursos suficientes.

Scripts relacionados:

- `scripts/analisis_tiempos_paralelo_ova.py`;
- `scripts/analisis_arquitecturas_ova.py`;
- `scripts/analisis_configuraciones_seleccionadas_estadistico.py`;
- `scripts/figuras_tfm_configuraciones_seleccionadas.py`;
- `scripts/figuras_tfm_delta_sensibilidad_configuraciones.py`.

## Análisis estadístico

El análisis estadístico se realiza sobre diferencias pareadas por semilla:

```text
f1_macro(OVA, semilla) - f1_macro(multi-salida, semilla)
```

Scripts principales:

- `scripts/test_wilcoxon_pareado.py`;
- `scripts/test_equivalencia_tost_bootstrap.py`;
- `scripts/analisis_potencia_wilcoxon.py`;
- `scripts/analisis_potencia_tost_bootstrap.py`;
- `scripts/analisis_potencia_wilcoxon_configuraciones_seleccionadas.py`;
- `scripts/analisis_potencia_tost_configuraciones_seleccionadas.py`.

## Explicabilidad visual

El bloque de explicabilidad estudia si los mapas de relevancia de los modelos
multi-salida y OVA son espacialmente similares y, cuando hay máscaras, si se
alinean con la región anotada.

Scripts principales:

- `scripts/explicabilidad_gradcam_vgg.py`: Grad-CAM y Grad-CAM++;
- `scripts/explicabilidad_lrp_vgg.py`: LRP;
- `scripts/oclusion_tumor_brisc.py`: oclusión tumoral y peritumoral en BRISC;
- `scripts/analisis_dataset_brisc_train_test.py`: descriptores de imagen y máscara;
- `scripts/intervenciones_shortcuts_brisc.py`: intervenciones exploratorias;
- `scripts/analisis_atajos_ham10000_artifacts.py`: proxies de artefactos en HAM10000;
- `scripts/compute_ham_nested_probabilities.py`: análisis jerárquico en HAM10000;
- `scripts/figuras_tfm_xai_paneles.py`: figuras finales de explicabilidad.

## Hardware utilizado

Los experimentos se ejecutaron en servidores con GPU. Para evitar depender de
nombres internos de la infraestructura, se resumen por sus características:

| Equipo | CPU | Memoria | GPU | Uso principal |
| --- | --- | --- | --- | --- |
| A | 2x Intel Xeon E5-2630 | 128 GB DDR3 | 4x RTX 2080 Ti | Conjuntos tabulares clásicos y BRISC |
| B | 2x Intel Xeon E5-2630 | 128 GB DDR4 | 4x GTX Titan Xp | MNIST y CIFAR-10 |
| C | 2x Intel Xeon E5-2630 | 128 GB DDR4 | 3x GTX Titan X Pascal + 1x GTX Titan Xp | TB Chest X-ray |

En los análisis temporales, las comparaciones se realizan siempre dentro del
mismo conjunto de datos y bajo el mismo entorno de ejecución.

## Datos locales

Rutas esperadas por defecto:

```text
data/brisc2025/train/
data/brisc2025/test/
data/tb_chest_xray/
data/ham10000/
```

MNIST y CIFAR-10 se descargan o leen desde `data/` mediante `torchvision`.

## Documentación complementaria

- `docs/capacidades_codigo.md`: capacidades implementadas no necesariamente usadas en los resultados finales.
- `docs/protocolo_explicabilidad_actual.md`: protocolo final de explicabilidad.
- `docs/resumen_ham10000_decision_atajos.md`: decisión metodológica sobre HAM10000.
- `docs/ham10000_atajos_por_artefacto_y_clase.md`: análisis de artefactos en HAM10000.

## Nota sobre ejecución en clúster

El repositorio no depende de un gestor de colas concreto. Los experimentos
principales pueden ejecutarse mediante los scripts Python anteriores. En un
entorno con ejecución por lotes, el entrenamiento OVA por clase puede
distribuirse lanzando ejecuciones independientes de `run_parallel_ova.py` y
agregando después los resultados; esta parte depende de la infraestructura
disponible y no se incluye como requisito del repositorio.
