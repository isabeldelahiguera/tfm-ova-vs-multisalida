# TFM - Output coupling in neural networks

Este repositorio contiene el código experimental del TFM sobre acoplamiento de salidas en redes neuronales. La motivación teórica parte de una equivalencia funcional ideal: bajo el marco considerado en el artículo de referencia, un problema multisalida puede reformularse como `t` problemas escalares, uno por componente.

El objetivo experimental es estudiar hasta qué punto esa equivalencia se observa en redes neuronales entrenadas en la práctica. En este contexto intervienen factores que no aparecen en la formulación ideal, como la parametrización, la optimización aproximada, la arquitectura compartida y la función de pérdida.

La comparación principal debe interpretarse en términos funcionales y predictivos: se analiza si una red con `t` salidas y `t` redes independientes producen salidas equivalentes, o suficientemente similares, sobre los mismos datos. No se asume equivalencia computacional. Entrenar `t` redes independientes puede requerir más parámetros, más tiempo y más coste que entrenar una única red con `t` salidas. Por eso el tiempo de entrenamiento se registra como información complementaria, no como la hipótesis central del TFM.

Además de comprobar si la equivalencia funcional se mantiene de forma aproximada, el análisis estudia si la descomposición One-vs-All (`OVA`) puede igualar o incluso mejorar el comportamiento predictivo de la red `multi-output` en algunos datasets. Por este motivo no se observa solo la `accuracy`, sino también métricas por clase como `TPR`, `FPR`, `TNR` y `FNR`, que permiten detectar si un enfoque reduce falsos positivos o falsos negativos aunque la accuracy global sea parecida.

## Comparación principal

En clasificación, el experimento relevante para la pregunta del TFM es:

- **Multi-output / multiclase**: una única red con una salida por clase.
- **OVA**: `t` redes binarias, una para cada clase frente al resto.

El análisis no se limita a comprobar si `OVA` reproduce aproximadamente la salida de la red `multi-output`. También se evalúa si esta descomposición puede mejorar algunas métricas predictivas al especializar cada red binaria en una clase concreta.

`OVO` aparece como extensión posible en el código, pero no es la comparación central porque no responde directamente a la equivalencia entre `t` redes y una red con `t` salidas. En la notebook de resultados se mantiene fuera por defecto.

En regresión, la comparación equivalente es:

- **Multi-output**: una red con `t` salidas continuas.
- **Decoupled**: `t` redes independientes, una por variable objetivo.

La parte de regresión está contemplada en el código, aunque el foco principal actual del análisis está en clasificación.

## Consideraciones

En esta primera aproximación se mantiene fija la configuración base de la red para aislar el efecto principal que se quiere estudiar: compartir salidas en una única red frente a entrenar redes independientes. Por defecto se usa un MLP con dos capas ocultas `[32, 16]`, activación ReLU, sin batch normalization, `batch_size=32` y `learning_rate=0.001`.

En análisis posteriores se pueden estudiar otras configuraciones, por ejemplo cambiando `--hidden-layers`, activando `--batch-normalization`, variando `--epochs`, `--learning-rate` o `--batch-size`. Esto permite comprobar si las conclusiones se mantienen con otras arquitecturas e hiperparámetros.

En datasets de imagen, como `MNIST`, `digits` y especialmente `CIFAR-10`, tiene sentido plantear una extensión con redes convolucionales (`CNN`). El MLP aplanado se mantiene como experimento base porque permite comparar todos los datasets bajo una misma familia de modelos y aislar el efecto de compartir salidas frente a entrenar modelos independientes. Sin embargo, al aplanar las imágenes se pierde la estructura espacial local, precisamente el tipo de información que una CNN puede aprovechar mediante filtros compartidos.

Por este motivo, una comparación con CNNs puede incluirse como análisis de robustez arquitectónica en datasets de imagen: se repetiría la comparación entre una única CNN `multi-output` y `t` CNNs binarias independientes en esquema `OVA`. Esta extensión no cambia la hipótesis central del TFM, sino que permite estudiar si las conclusiones sobre acoplamiento de salidas se mantienen cuando cambia la parametrización y se introduce un sesgo inductivo espacial más adecuado para imágenes. En datasets tabulares no se considera una extensión natural, salvo que se reformulen explícitamente las variables como una estructura espacial artificial, lo cual no forma parte del objetivo principal.

La primera extensión convolucional implementada es `--model-arch vgg`, una VGG compacta o `VGG-like`, no una VGG-16 estándar. Mantiene el patrón característico de VGG con convoluciones `3x3`, activaciones ReLU y bloques terminados en `MaxPool`, pero usa menos bloques y menos canales para que el experimento `OVA` sea viable. Esto es importante porque en MNIST y CIFAR-10 se entrenan 11 redes por semilla cuando se comparan `multi-output` y `OVA`: una red multiclase y diez redes binarias.

La arquitectura `vgg` actual se usa como primer experimento de robustez en imagen. El plan experimental es:

- Ejecutar primero `MNIST` y `CIFAR-10` con la VGG compacta para comprobar que el pipeline de imágenes funciona y medir el coste real de `OVA` en GPU.
- Comparar esos resultados con los experimentos MLP aplanados ya existentes o equivalentes.
- Si el coste computacional es razonable, añadir después una variante `vgg16` adaptada a imágenes pequeñas como extensión más estándar.

Por tanto, en el estado actual del código, `vgg` debe interpretarse como una CNN inspirada en VGG y no como la arquitectura VGG-16 original.

El preprocesamiento se mantiene deliberadamente sencillo: partición de los datos, estratificación en clasificación cuando `y` es un vector unidimensional, imputación básica de valores perdidos cuando existen, estandarización de variables tabulares y normalización específica para MNIST y CIFAR-10. La misma preparación de datos se aplica a todos los enfoques comparados para que la comparación no dependa de diferencias de preprocesamiento.

El conjunto de validación se usa para `early stopping`: durante el entrenamiento se monitoriza la pérdida de validación, se conservan los pesos del modelo con menor `val_loss` y al final se restauran esos pesos antes de evaluar en test. Por defecto se usa `--early-stopping-patience 10` y `--early-stopping-min-delta 1e-4`. La paciencia indica cuántas épocas consecutivas sin mejora se permiten antes de parar; `min_delta` exige una reducción mínima de la pérdida para considerar que ha habido una mejora real. Si se quiere desactivar este criterio, se puede usar `--early-stopping-patience 0`.

El objetivo no es optimizar cada dataset al máximo, sino comparar el efecto de compartir salidas frente a entrenar modelos separados. Técnicas como selección de variables, reducción de dimensionalidad, normalizaciones alternativas o aumentación de datos quedan como posibles extensiones.

## Estructura

```text
.
|-- run_experiments.py              # Script principal para lanzar experimentos desde terminal
|-- tfm/
|   |-- config.py                   # Dataclass ExperimentData
|   |-- data.py                     # Carga de datasets, datos sintéticos, partición y escalado
|   |-- models.py                   # Definición del MLP y la VGG compacta
|   |-- training.py                 # DataLoaders, bucle de entrenamiento y predicciones
|   |-- metrics.py                  # Métricas de clasificación y regresión
|   |-- evaluation.py               # Evaluación de modelos entrenados
|   |-- experiment.py               # Orquestación de experimentos y agregación de resultados
|   `-- cli.py                      # Argumentos de línea de comandos
|-- docs/
|   `-- resultados_clasificacion.md # Registro resumido de resultados y conclusiones provisionales
|-- resultados_clasificacion.ipynb  # Análisis de resultados de clasificación
|-- resultados_clasificacion/       # Tablas combinadas generadas por la notebook
|-- scripts/                        # Scripts SLURM para lanzar experimentos en GPU
|-- exp_*.csv                       # Resultados detallados de experimentos
`-- exp_*_summary.csv               # Resultados medios agregados por configuración
```

La parte reproducible del entrenamiento está en los scripts `.py`. La notebook `resultados_clasificacion.ipynb` organiza el análisis de resultados de clasificación. Carga los CSV por semilla, usa los `*_summary.csv` como medias por configuración y genera gráficas de `accuracy`, `f1_macro`, tiempo, `TPR`, `FPR`, `TNR` y `FNR`.

El documento `docs/resultados_clasificacion.md` resume los resultados ya obtenidos, incluyendo la tabla comparativa por dataset y una lectura provisional de las conclusiones. Sirve como registro de trabajo para no depender solo de la notebook mientras se van acumulando experimentos.

La carpeta `resultados_clasificacion/` no contiene experimentos nuevos por sí misma, sino tablas derivadas que genera la notebook a partir de los CSV principales:

- `clasificacion_detalle_combinado.csv`: une los ficheros `exp_*.csv` de clasificación. Mantiene una fila por dataset, semilla y modelo, por lo que sirve para revisar la variabilidad entre ejecuciones.
- `clasificacion_summary_combinado.csv`: une los ficheros `exp_*_summary.csv`. Sus valores son medias por configuración, normalmente sobre las semillas `1`, `2` y `3`. Este fichero es el que se usa para las tablas y gráficas globales.

Por tanto, cuando se interpretan las gráficas globales de la notebook, se usan medias agregadas por configuración a partir de `clasificacion_summary_combinado.csv`. Las conclusiones no se basan en una única semilla aislada, aunque la notebook conserva el detalle por semilla en `clasificacion_detalle_combinado.csv` para comprobar si el comportamiento es estable.

## Ejecución

Ejemplo de clasificación sintética:

```bash
python run_experiments.py --task classification --dataset synthetic_multiclass --dependency-strength 0.3 --seeds 1 2 3 --coupling-modes ova --output-csv exp_clasif_sint_0.3.csv --summary-csv exp_clasif_sint_0.3_summary.csv
```

Ejemplo con un dataset real:

```bash
python run_experiments.py --task classification --dataset wine --seeds 1 2 3 --coupling-modes ova --output-csv exp_wine.csv --summary-csv exp_wine_summary.csv
```

Ejemplo de regresión:

```bash
python run_experiments.py --task regression --dataset synthetic_multiregression --dependency-strength 0.3 --seeds 1 2 3 --coupling-modes decoupled --output-csv exp_reg_sint_0.3.csv --summary-csv exp_reg_sint_0.3_summary.csv
```

Ejemplo con MNIST, que tiene 10 clases y es más costoso:

```bash
python run_experiments.py --task classification --dataset mnist --seeds 1 --coupling-modes ova --output-csv exp_mnist.csv --summary-csv exp_mnist_summary.csv
```

Ejemplo con MNIST usando VGG compacta sobre la imagen, sin aplanarla:

```bash
python run_experiments.py --task classification --dataset mnist --model-arch vgg --seeds 1 --coupling-modes ova --epochs 50 --early-stopping-patience 10 --early-stopping-min-delta 1e-4 --batch-size 64 --output-csv exp_mnist_vgg.csv --summary-csv exp_mnist_vgg_summary.csv
```

Ejemplo con CIFAR-10:

```bash
python run_experiments.py --task classification --dataset cifar10 --seeds 1 --coupling-modes ova --output-csv exp_cifar10.csv --summary-csv exp_cifar10_summary.csv
```

Ejemplo con CIFAR-10 usando VGG compacta:

```bash
python run_experiments.py --task classification --dataset cifar10 --model-arch vgg --seeds 1 --coupling-modes ova --epochs 50 --early-stopping-patience 10 --early-stopping-min-delta 1e-4 --batch-size 64 --output-csv exp_cifar10_vgg.csv --summary-csv exp_cifar10_vgg_summary.csv
```

Para lanzar los experimentos VGG en SLURM, es recomendable enviar un job por dataset:

```bash
sbatch scripts/run_tfm_mnist_vgg_slurm.sh
sbatch scripts/run_tfm_cifar10_vgg_slurm.sh
```

Así, si CIFAR-10 tarda más o falla por memoria, no arrastra el experimento de MNIST. Se pueden cambiar parámetros sin editar los ficheros, por ejemplo:

```bash
SEEDS="1 2 3" EPOCHS=50 EARLY_STOPPING_PATIENCE=10 EARLY_STOPPING_MIN_DELTA=0.0001 BATCH_SIZE=128 sbatch scripts/run_tfm_mnist_vgg_slurm.sh
SEEDS="1 2 3" EPOCHS=50 EARLY_STOPPING_PATIENCE=10 EARLY_STOPPING_MIN_DELTA=0.0001 BATCH_SIZE=128 sbatch scripts/run_tfm_cifar10_vgg_slurm.sh
```

También se conserva `scripts/run_tfm_slurm.sh` como lanzador combinado si se quiere ejecutar más de un dataset dentro del mismo job.

Ejemplo con Dermatology:

```bash
python run_experiments.py --task classification --dataset dermatology --seeds 1 2 3 --coupling-modes ova --output-csv exp_dermatology.csv --summary-csv exp_dermatology_summary.csv
```

Ejemplo con Heart Disease:

```bash
python run_experiments.py --task classification --dataset heart_disease --seeds 1 2 3 --coupling-modes ova --output-csv exp_heart_disease.csv --summary-csv exp_heart_disease_summary.csv
```

Para lanzar MNIST con las tres semillas:

```bash
python run_experiments.py --task classification --dataset mnist --seeds 1 2 3 --coupling-modes ova --output-csv exp_mnist.csv --summary-csv exp_mnist_summary.csv
```

## Datasets

El código permite combinar datasets reales con datasets sintéticos parametrizados. Los nombres de esta sección son los valores que se pueden pasar al argumento `--dataset`.

Los datasets reales sirven como validación en problemas conocidos. Los sintéticos son importantes para estudiar de forma controlada el número de salidas, el número de muestras, la dimensión de entrada y el grado de dependencia entre salidas.

Datasets de clasificación implementados:

- `synthetic_multiclass`: clasificación sintética multiclase.
- `iris`: dataset pequeño de referencia.
- `wine`: dataset tabular pequeño/medio.
- `digits`: imágenes pequeñas en formato tabular.
- `breast_cancer`: clasificación binaria real.
- `mnist`: clasificación de dígitos con 10 clases, más costoso que los anteriores.
- `cifar10`: clasificación de imágenes RGB de 10 clases, aplanadas como entrada tabular.
- `dermatology`: clasificación dermatológica de UCI con 6 clases.
- `heart_disease`: clasificación clínica de UCI con 5 clases en la variable objetivo `num`.

Datasets de regresión implementados:

- `synthetic_multiregression`: regresión sintética multi-output.
- `linnerud`: regresión multi-output pequeña.
- `energy`: regresión multi-output real descargada desde OpenML.

En los datasets sintéticos se pueden variar principalmente:

- `--synthetic-samples`: número de observaciones.
- `--synthetic-features`: número de variables de entrada.
- `--synthetic-classes`: número de clases en clasificación.
- `--synthetic-targets`: número de salidas en regresión.
- `--dependency-strength`: intensidad de la dependencia compartida entre salidas.

Para estudiar la dependencia entre salidas se puede repetir el mismo experimento con varios valores de `dependency_strength`, por ejemplo:

```text
0.0, 0.1, 0.3, 0.6, 0.8
```

Así se puede analizar si una única red con varias salidas mejora especialmente cuando existe estructura compartida entre las salidas, y si las redes independientes se comportan de forma similar cuando esa dependencia es baja.

## Semillas y repetición de experimentos

Los experimentos se ejecutan con varias semillas para reducir la dependencia de una única partición de los datos o de una única inicialización aleatoria de la red neuronal. En este trabajo se usan habitualmente tres semillas (`1`, `2` y `3`) como primera aproximación experimental: no pretenden agotar toda la variabilidad posible, pero sí permiten comprobar si el comportamiento observado se repite de forma razonablemente estable.

Cada semilla afecta principalmente a:

- la división de los datos en entrenamiento, validación y test;
- la inicialización de los pesos de la red;
- el orden de los lotes durante el entrenamiento;
- la generación de datos, cuando el dataset es sintético.

Por eso los resultados detallados se guardan por semilla en los ficheros `exp_*.csv`, mientras que los ficheros `exp_*_summary.csv` recogen la media por configuración. En el análisis, la media sirve para resumir el comportamiento general, pero las filas por semilla son importantes para detectar si una conclusión depende de una ejecución concreta.

## Generación de datos sintéticos de clasificación multiclase

La función `make_synthetic_multiclass` genera un dataset artificial para clasificación multiclase. El objetivo es construir un problema controlado en el que se pueda variar el número de muestras, el número de variables de entrada, el número de clases y la intensidad de una estructura compartida entre clases.

Primero se generan las variables de entrada `X` a partir de una distribución normal. Después se define, para cada clase, una puntuación o `logit`. Cada `logit` combina tres partes:

- una componente específica de clase, obtenida mediante una combinación lineal propia de `X`;
- una componente compartida, basada en una dirección latente común;
- ruido aleatorio, para evitar un problema completamente determinista.

La componente compartida no se suma exactamente igual a todas las clases. Cada clase tiene un peso propio (`shared_weights[class_idx]`) que determina cómo le afecta esa dirección común. Por eso el parámetro `dependency_strength` controla la influencia de esa estructura compartida: cuando vale `0.0`, los logits dependen solo de componentes específicas de clase y ruido; cuando aumenta, las clases quedan más condicionadas por una fuente común de variación.

De forma simplificada, para cada clase se calcula:

```text
shared_component = X @ shared_direction
class_component = X @ class_specific[class_idx]
coupled_component = dependency_strength * shared_weights[class_idx] * shared_component

logit[class_idx] = class_component + coupled_component + noise
```

Una vez calculados los logits de todas las clases, la etiqueta se asigna con:

```text
y = argmax(logits)
```

Es decir, cada observación se asigna a la clase con mayor puntuación. La función devuelve:

- `X`: variables de entrada.
- `y`: etiqueta de clase asignada a cada observación.
- `class_names`: nombres de las clases generadas.

## Preprocesamiento

Antes de entrenar los modelos, los datasets tabulares se dividen en entrenamiento, validación y test. En clasificación, cuando `y` es un vector unidimensional de etiquetas, la partición se hace de forma estratificada para mantener proporciones de clases similares en los tres conjuntos.

Las variables de entrada se estandarizan usando `StandardScaler`: la media y la desviación típica se calculan solo con el conjunto de entrenamiento, y después se aplica la misma transformación a validación y test. Esto evita usar información de validación o test durante el preprocesamiento.

Si un dataset contiene valores perdidos, como ocurre en `dermatology` con la variable `age` o en `heart_disease` con algunas variables clínicas, se imputan con la mediana calculada solo sobre el conjunto de entrenamiento. Después se aplica esa misma imputación a validación y test. La imputación se hace después de separar los datos para evitar fuga de información desde validación o test hacia entrenamiento.

El objetivo de esta imputación y estandarización no es mejorar un enfoque frente a otro, sino evitar errores por valores ausentes y reducir problemas de escala que pueden afectar al entrenamiento de redes neuronales. La misma partición y las mismas transformaciones se aplican a todos los enfoques comparados, por lo que la comparación entre la red `multi-output` y las redes independientes se mantiene en igualdad de condiciones.

En MNIST se usa un preprocesamiento específico de imagen: los píxeles se normalizan dividiendo entre `255.0`, quedando en el rango `[0, 1]`. Con `--model-arch mlp`, cada imagen se aplana de `28x28` a `784` variables. Con `--model-arch vgg`, se conserva la forma espacial como tensor `1x28x28`.

En CIFAR-10 se aplica un tratamiento análogo: las imágenes RGB se normalizan dividiendo entre `255.0`. Con `--model-arch mlp`, cada imagen se aplana de `32x32x3` a `3072` variables. Con `--model-arch vgg`, se conserva como tensor `3x32x32`.

## Salidas

Cada ejecución genera dos ficheros:

- `output_csv`: resultados por semilla y modelo.
- `summary_csv`: media de resultados por configuración.

Además de las métricas predictivas, se guarda `train_time_seconds`, que mide el tiempo de entrenamiento de cada enfoque. Esta variable es útil para comparar el coste computacional de entrenar una única red `multi-output` frente a entrenar varias redes independientes. Debe interpretarse con cuidado, porque depende del hardware, del uso de CPU/GPU y de la carga del sistema durante la ejecución.

También se guardan variables asociadas al `early stopping`: `best_val_loss`, `epochs_trained`, `total_epochs_trained` y `models_stopped_early`. En modelos únicos, `epochs_trained` y `total_epochs_trained` coinciden. En enfoques descompuestos como `OVA`, `epochs_trained` resume la media de épocas entrenadas por red binaria y `total_epochs_trained` suma las épocas de todas las redes entrenadas.

## Métricas

En clasificación se guardan:

- `accuracy`: proporción total de aciertos.
- `balanced_accuracy`: media del recall por clase; es útil si las clases están desbalanceadas.
- `precision_macro`: precisión media por clase, calculada dando el mismo peso a cada clase.
- `recall_macro`: recall medio por clase; coincide conceptualmente con el TPR macro.
- `f1_macro`: media del F1 por clase; resume precisión y recall sin favorecer clases mayoritarias.
- `tpr_macro`: true positive rate medio, calculado one-vs-rest por clase.
- `fpr_macro`: false positive rate medio, calculado one-vs-rest por clase.
- `tnr_macro`: true negative rate medio, calculado one-vs-rest por clase.
- `fnr_macro`: false negative rate medio, calculado one-vs-rest por clase.

Estas métricas sirven para este problema porque la comparación no debe depender solo del porcentaje global de aciertos. Si una red `multi-output` y un conjunto de redes `OVA` tienen una `accuracy` parecida, las métricas por clase permiten ver si una de las estrategias falla más en clases minoritarias, genera más falsos positivos o sacrifica recall. Esto es importante porque la pregunta no es solo si ambos enfoques aciertan igual, sino si se comportan de forma equivalente por salida o clase.

En regresión se guardan:

- `mse`: error cuadrático medio.
- `mae`: error absoluto medio.
- `r2`: proporción de varianza explicada.

Estas métricas permiten comparar la red `multi-output` con las redes desacopladas midiendo tanto el error medio como la capacidad explicativa del modelo.

## Estado actual

El código está modularizado en `tfm/`. La entrada principal para ejecutar experimentos es `run_experiments.py`. El análisis principal de clasificación está organizado en `resultados_clasificacion.ipynb` y en `docs/resultados_clasificacion.md`.
