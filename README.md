# TFM - OVA frente a redes multi-salida

Este repositorio contiene el código experimental del TFM sobre clasificadores neuronales `multi-output` y clasificadores `One-vs-All` (`OVA`). La pregunta central es si una red con una salida por clase se comporta, en la práctica, de forma equivalente o suficientemente parecida a entrenar una red binaria independiente por clase.

La comparación no se limita al rendimiento predictivo. El trabajo también estudia el coste de entrenamiento, la posibilidad de paralelizar OVA, la sensibilidad a arquitecturas reducidas y, en datasets biomédicos, si las predicciones parecen apoyarse en regiones anatómicamente plausibles o en señales contextuales del dataset.

## Historia experimental

La línea principal del TFM compara:

- `multi-output`: una única red multiclase con una salida por clase.
- `OVA`: un conjunto de redes binarias, una por clase frente al resto.

La hipótesis inicial se evalúa primero como equivalencia práctica de rendimiento mediante métricas pareadas por semilla. Después se añade la parte computacional: OVA entrena más modelos, pero esos modelos pueden entrenarse por clase en paralelo. Por eso el código guarda tanto el coste secuencial acumulado como una estimación del tiempo paralelo ideal basada en el máximo de los tiempos por clase.

Sobre esa base se probaron extensiones de eficiencia:

- paralelización de los clasificadores OVA;
- reducción arquitectónica en MLP y VGG compacta;
- configuraciones seleccionadas por rendimiento y tiempo;
- análisis estadístico con Wilcoxon y TOST/bootstrap;
- generación de figuras y tablas para la memoria.

En la parte biomédica se añadió una auditoría de explicabilidad. Para BRISC se combinaron Grad-CAM/Grad-CAM++, LRP, oclusión tumoral y análisis de posibles shortcuts. La lectura final es prudente: OVA puede mejorar el rendimiento en BRISC, pero no muestra de forma sistemática una mejor localización tumoral. Para HAM10000 se dejaron implementadas pruebas dermatológicas adicionales con VGG16 preentrenada, balanceo, selección de clases, análisis de atajos visuales y un análisis jerárquico complementario maligno/no maligno con tres clases malignas.

## Organización del código

```text
.
|-- run_experiments.py          # Entrada principal para experimentos secuenciales
|-- run_parallel_ova.py         # Entrenamiento/agregación de OVA por clase
|-- tfm/                        # Paquete principal del experimento
|   |-- cli.py                  # Argumentos de línea de comandos
|   |-- data.py                 # Carga y partición de datasets
|   |-- models.py               # MLP, VGG compacta, VGG16/ViT preentrenadas
|   |-- training.py             # Entrenamiento, early stopping y loaders
|   |-- experiment.py           # Flujo multi-output, OVA, métricas y CSV
|   |-- parallel_ova.py         # Flujo OVA paralelizable
|   |-- metrics.py              # Métricas de clasificación y regresión
|   `-- evaluation.py           # Utilidades de evaluación
|-- scripts/                    # Análisis estadístico, explicabilidad y figuras
|-- docs/                       # Documentación de protocolos y extensiones
|-- requirements.txt            # Dependencias principales usadas por el TFM
`-- .gitignore                  # Datos, resultados, memoria y artefactos locales
```

Los resultados experimentales, datasets descargados, checkpoints, logs y la memoria en LaTeX no se versionan. En local se usan carpetas como `data/`, `resultados_actualizados/`, `slurm_logs/`, `Memoria TFM/` y `figuras_caepia/`.

## Datasets principales

Los experimentos centrales usan:

- `iris`, `wine`, `breast_cancer` y `digits` con MLP;
- `mnist` y `cifar10` con VGG compacta;
- `brisc` y `tb_chest_xray` con VGG compacta e imágenes `128x128`;
- `ham10000` como extensión dermatológica, no como eje principal de la comparación final.

El código conserva datasets y modos extra para exploración, como regresión, datasets sintéticos, `OVO`, `dermatology` y `heart_disease`. Esos casos se describen en `docs/capacidades_codigo.md` y no forman parte de los resultados principales del TFM.

## Instalación

```bash
pip install -r requirements.txt
```

`requirements.txt` contiene las dependencias principales usadas en la línea final del TFM. Algunas extensiones exploratorias pueden necesitar dependencias opcionales; por ejemplo, `dermatology` y `heart_disease` usan `ucimlrepo`, pero no se incluye como dependencia principal porque esos datasets no se utilizaron en los resultados finales.

## Ejecución básica

Ejemplo tabular con Wine:

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

Cada ejecución detallada genera:

- `exp_*.csv`: resultados por semilla y tipo de modelo;
- `exp_*_summary.csv`: resumen agregado por configuración.

Las métricas principales son `accuracy`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `f1_macro` y `train_time_seconds`. En el flujo OVA paralelo se añaden `parallel_train_time_seconds` y `ova_model_train_time_seconds_mean`.

## Flujos de análisis

### Comparación principal

`run_experiments.py` ejecuta la comparación secuencial entre `multi-output` y `OVA`. El diseño estadístico es pareado por semilla: para una semilla `x`, se compara `f1_macro(OVA, x) - f1_macro(multi-output, x)`.

Los scripts estadísticos principales son:

- `scripts/test_wilcoxon_pareado.py`;
- `scripts/test_equivalencia_tost_bootstrap.py`;
- `scripts/analisis_potencia_wilcoxon.py`;
- `scripts/analisis_potencia_tost_bootstrap.py`.

### OVA paralelo y eficiencia

`run_parallel_ova.py` separa el entrenamiento OVA por clase. Primero genera artefactos por clase y después reconstruye el ensemble completo por semilla. Este flujo permite estimar el coste paralelo ideal de OVA.

Scripts relacionados:

- `scripts/analisis_tiempos_paralelo_ova.py`;
- `scripts/analisis_arquitecturas_ova.py`;
- `scripts/analisis_configuraciones_seleccionadas_estadistico.py`;
- `scripts/figuras_tfm_configuraciones_seleccionadas.py`;
- `scripts/figuras_tfm_delta_sensibilidad_configuraciones.py`.

### Explicabilidad en BRISC

El protocolo final de explicabilidad combina mapas, perturbaciones y análisis del dataset:

- `scripts/explicabilidad_gradcam_vgg.py`: Grad-CAM y Grad-CAM++;
- `scripts/explicabilidad_lrp_vgg.py`: LRP con `zennit`;
- `scripts/oclusion_tumor_brisc.py`: oclusión de máscara tumoral y peritumoral;
- `scripts/analisis_dataset_brisc_train_test.py`: descriptores morfológicos, de intensidad y contexto;
- `scripts/resumir_shortcuts_brisc.py`: resumen de posibles shortcuts;
- `scripts/analisis_errores_dataset_explicabilidad.py`: cruce entre predicciones, errores y descriptores.

El detalle metodológico está en `docs/protocolo_explicabilidad_actual.md`. El documento histórico `docs/resumen_explicabilidad.md` conserva pruebas exploratorias anteriores.

### HAM10000

HAM10000 queda como extensión dermatológica. El código soporta split interno por `lesion_id`, test oficial de ISIC 2018 Task 3, VGG compacta, VGG16 preentrenada, balanceo por sampler, pesos de clase, focal loss para OVA, calibración auxiliar y modo binario maligno/no maligno.

La versión final incluye además scripts para combinar probabilidades jerárquicas aproximadas en HAM10000 y generar paneles de mapas Grad-CAM++ comparando el modelo de siete clases, la etapa binaria y el clasificador de tres clases malignas.

Documentos relacionados:

- `docs/resumen_ham10000_decision_atajos.md`;
- `docs/ham10000_atajos_por_artefacto_y_clase.md`;
- `docs/capacidades_codigo.md`.

## Parámetros útiles

Algunos argumentos relevantes de `run_experiments.py`:

- `--coupling-modes multi-output ova`: compara red multiclase y OVA.
- `--model-arch mlp|vgg|vgg16-pretrained|vit-b-16-pretrained`: arquitectura del modelo.
- `--hidden-layers`: capas ocultas de MLP.
- `--vgg-channels`: canales de la VGG compacta.
- `--class-weighting balanced`: pesos por clase en la pérdida.
- `--train-sampler balanced`: sampler balanceado en entrenamiento.
- `--ova-loss bce|focal`: pérdida binaria para clasificadores OVA.
- `--ova-calibration none|platt|threshold|threshold-f1`: calibración auxiliar de OVA.
- `--pretrained-finetune frozen|block5|last-block|full`: política de ajuste para modelos preentrenados.
- `--predictions-csv`: guarda predicciones por muestra para análisis posteriores.

## Datos locales

BRISC:

```text
data/brisc2025/train/
data/brisc2025/test/
```

Tuberculosis:

```text
data/tb_chest_xray/
```

HAM10000:

```text
data/ham10000/
  HAM10000_metadata
  images/
  raw/
  masks/
```

MNIST y CIFAR-10 se descargan o leen desde `data/` mediante `torchvision`.

## Documentación

- `docs/capacidades_codigo.md`: capacidades implementadas que no son necesariamente resultados finales.
- `docs/experimentos_recientes.md`: resumen de eficiencia OVA, arquitecturas reducidas, BRISC y HAM10000.
- `docs/protocolo_explicabilidad_actual.md`: protocolo final de explicabilidad en BRISC.
- `docs/resumen_explicabilidad.md`: historial exploratorio de explicabilidad.
- `docs/resumen_ham10000_decision_atajos.md`: decisión metodológica sobre HAM10000.
- `docs/ham10000_atajos_por_artefacto_y_clase.md`: análisis de atajos visuales en HAM10000.

## Nota de reproducibilidad

El repositorio versiona código y documentación, no datasets ni resultados pesados. Varios scripts de análisis tienen rutas por defecto orientadas al entorno local usado durante el TFM. Si se regeneran experimentos en otra máquina, conviene pasar rutas explícitas por argumentos o variables de entorno y mantener la salida bajo `resultados_actualizados/`.
