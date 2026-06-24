# Protocolo actual de explicabilidad

Este documento resume el hilo final de explicabilidad usado para BRISC. La idea central no es demostrar que OVA localiza mejor el tumor, sino evaluar si Multi y OVA se apoyan en la región tumoral, en el peritumor o en señales contextuales que puedan actuar como shortcuts.

## Preguntas del análisis

El análisis se organiza alrededor de tres preguntas:

1. ¿Las explicaciones espaciales localizan el tumor?
2. Si no lo localizan, ¿el tumor sigue siendo causal para la predicción?
3. Si el tumor no parece causal en algunas clases, ¿existen regularidades del dataset que puedan explicar shortcuts, especialmente en OVA?

Esta separación es importante porque un mapa de explicabilidad pobre no prueba por sí solo que el modelo no use el tumor. Por eso se combinan mapas, oclusión y análisis descriptivo del dataset.

## Métodos usados

### Grad-CAM y Grad-CAM++

Los mapas Grad-CAM y Grad-CAM++ se calculan con la librería `pytorch-grad-cam`, no con una implementación propia.

Scripts principales:

- `scripts/explicabilidad_gradcam_vgg.py`
- `scripts/run_explicabilidad_gradcam_slurm.sh`
- `scripts/analisis_gradcam_por_plano_clase.py`

Se estudian:

- `multi-output` frente a `OVA`;
- clase predicha o clase verdadera como objetivo;
- clase real;
- plano (`axial`, `coronal`, `sagittal`);
- semillas múltiples;
- métricas de solapamiento con máscara tumoral y activación por zonas.

### LRP

LRP se calcula con `zennit`, usando el compuesto `EpsilonGammaBox`.

Scripts principales:

- `scripts/explicabilidad_lrp_vgg.py`
- `scripts/run_explicabilidad_lrp_slurm.sh`

LRP se usa como comprobación alternativa a Grad-CAM/Grad-CAM++. La motivación es evitar que la conclusión dependa de un único método de explicabilidad.

### Oclusión tumoral

La oclusión es un experimento de perturbación diseñado para BRISC. Se tapa la máscara tumoral y versiones dilatadas de la máscara para medir si cae la probabilidad de la clase verdadera o cambia la predicción.

Script principal:

- `scripts/oclusion_tumor_brisc.py`
- `scripts/run_oclusion_tumor_brisc_slurm.sh`

La dilatación usa `skimage.morphology.binary_dilation` con `disk(radius)`. La región ocluida se rellena con la media local de un anillo alrededor de la lesión.

Radios usados:

- `0`: solo máscara tumoral;
- `5`, `10`, `15`: tumor más entorno peritumoral progresivamente mayor.

La oclusión no se sustituye por `captum.attr.Occlusion` porque aquí no se quieren ventanas regulares, sino intervenir sobre la máscara tumoral real. Por tanto, es un experimento causal específico del estudio, no un algoritmo genérico de atribución.

### Análisis del dataset y shortcuts

El análisis de dataset cuantifica regularidades de las imágenes de entrenamiento y test que podrían ser usadas como shortcuts.

Script principal:

- `scripts/analisis_dataset_brisc_train_test.py`
- `scripts/run_analisis_dataset_brisc_slurm.sh`
- `scripts/resumir_shortcuts_brisc.py`

Para morfología de máscaras se usa `skimage`:

- `skimage.measure.regionprops` para bbox, centroide, área, diámetro equivalente, perímetro y forma;
- `skimage.morphology.binary_dilation` y `disk` para construir peritumor.

Se estudian variables como:

- intensidad global de la imagen;
- área visible/no negra;
- centroide de la zona visible;
- posición del tumor y de su bbox;
- tamaño y forma de la máscara;
- intensidad tumoral;
- intensidad peritumoral;
- intensidad fuera de tumor/peritumor;
- contraste tumor-vs-peritumor;
- contraste tumor-vs-fuera.

Estas variables no se introducen explícitamente en el modelo. Sirven para cuantificar patrones visuales que una CNN podría aprender directamente desde la imagen.

Los contrastes completos se guardan en:

- `contrastes_train_por_clase.csv`;
- `contrastes_train_por_clase_plano.csv`;
- equivalentes de test en `resumen_shortcuts_test/`.

El resumen legible de posibles shortcuts se genera con `scripts/resumir_shortcuts_brisc.py`. La regla es:

1. Para cada grupo se compara `grupo` frente a `resto`.
2. El grupo puede ser una clase (`glioma`) o una combinación clase+plano (`glioma|axial`).
3. Para cada variable se calcula `standardized_diff`, es decir, la diferencia de medias normalizada por la variabilidad.
4. Se ordenan las variables por `abs_standardized_diff`.
5. Las variables se agrupan en familias para no repetir variantes casi iguales. Por ejemplo, `outside_peritumor_r5_intensity_mean`, `outside_peritumor_r10_intensity_mean` y `outside_peritumor_r15_intensity_mean` pertenecen a `contexto_fuera_peritumor`.
6. Dentro de cada familia se conserva la variable más discriminativa.
7. Se escriben tablas resumidas por clase y por clase+plano.

Las familias principales son:

- `intensidad_global_imagen`: variables `image_intensity_*`;
- `area_visible_imagen`: `image_nonzero_area_frac`;
- `intensidad_zona_visible`: variables `image_nonzero_intensity_*`;
- `posicion_area_visible`: variables `image_nonzero_centroid_*`;
- `intensidad_tumor`: variables `tumor_intensity_*`;
- `intensidad_peritumor`: variables `peritumor_*`;
- `contexto_fuera_peritumor`: variables `outside_peritumor_*`;
- `contraste_tumor_vs_fuera`: variables `tumor_vs_outside_*`;
- `contraste_tumor_vs_peritumor`: variables `tumor_vs_peritumor_*`;
- `posicion_tumor_centro`: `mask_centroid_distance_center_norm`;
- `posicion_bbox_centro`: `mask_bbox_center_distance_center_norm`;
- `bbox_tumor`: resto de variables `mask_bbox_*`;
- `forma_tumor`: compacidad, perímetro, diámetro equivalente y área de máscara.

Los CSV principales de shortcuts son:

- `resultados_actualizados/analisis_dataset/brisc_train/resumen_shortcuts/shortcuts_clave_por_clase.csv`;
- `resultados_actualizados/analisis_dataset/brisc_train/resumen_shortcuts/shortcuts_clave_por_clase_plano.csv`;
- `resultados_actualizados/analisis_dataset/brisc_test/resumen_shortcuts_test/comparacion_top_train_vs_test_clase.csv`;
- `resultados_actualizados/analisis_dataset/brisc_test/resumen_shortcuts_test/comparacion_top_train_vs_test_clase_plano.csv`.

## Métricas espaciales

Para comparar mapas de explicabilidad con máscara tumoral se convierte cada mapa en una distribución espacial de relevancia normalizada. Las métricas se calculan para Multi y OVA de forma separada.

### Zonas anatómicas

Se definen tres zonas:

- `tumor`: máscara tumoral original.
- `peritumor_r5`: anillo alrededor del tumor obtenido dilatando la máscara con un disco de radio 5 píxeles y restando la máscara original.
- `outside_peritumor_r5`: resto de la imagen fuera de tumor y fuera del anillo peritumoral de radio 5.

En las métricas XAI actuales el peritumor se mide con `r5`. En oclusión sí se estudian varios radios (`0`, `5`, `10`, `15`) para evaluar el efecto causal de tapar tumor y contexto progresivamente mayor.

### Precision y recall en píxeles

Cuando se binariza el mapa de calor, se define:

- TP: píxeles activos del mapa que caen dentro del tumor.
- FP: píxeles activos del mapa que caen fuera del tumor.
- FN: píxeles del tumor que no quedan activos en el mapa.

Por tanto:

- `precision = TP / (TP + FP)`: de lo que el mapa marca como importante, qué fracción cae en tumor.
- `recall = TP / (TP + FN)`: de todo el tumor real, qué fracción queda cubierta por el mapa.

En este contexto, una precisión baja indica que mucha activación cae fuera de la máscara tumoral. Un recall bajo indica que el mapa cubre poca parte del tumor.

### Métricas principales

Las métricas principales para el texto del TFM son:

- `top_mask_area_dice`: selecciona tantos píxeles calientes como área tiene la máscara tumoral y calcula Dice frente a la máscara.
- `tumor_activation_frac`: fracción de activación total del mapa que cae dentro del tumor.
- `peritumor_r5_activation_frac`: fracción de activación total que cae en el anillo peritumoral de radio 5.
- `outside_peritumor_r5_activation_frac`: fracción de activación total que cae fuera de tumor y peritumor r5.
- `pointing_game_hit`: vale 1 si el máximo del mapa cae dentro del tumor y 0 si cae fuera.
- `mask_centroid_distance_norm`: distancia normalizada entre el centroide del mapa de calor y el centroide de la máscara tumoral.
- `gini`: concentración del mapa; valores más altos indican mapas más concentrados en pocos píxeles.

El centroide tumoral es la media de las coordenadas de todos los píxeles de la máscara. El centroide del mapa de calor es una media ponderada por la activación: los píxeles con mayor relevancia pesan más. La distancia se normaliza por la diagonal de la imagen para poder comparar entre mapas.

### Métricas complementarias

También se calculan métricas top-k y por umbral:

- `top5`, `top10`, `top15`, `top20`: seleccionan el 5%, 10%, 15% o 20% de píxeles más calientes del mapa.
- `top_2x_mask_area`: selecciona el doble de píxeles que ocupa la máscara tumoral.
- `thr50`: selecciona píxeles con activación normalizada mayor o igual que 0.50.
- `thr75`: selecciona píxeles con activación normalizada mayor o igual que 0.75.

Para responder a la pregunta "de los píxeles con activación mayor que 0.50 o 0.75, cuántos caen en tumor", se usan:

- `thr50_precision`
- `thr75_precision`

Para medir qué parte del tumor queda cubierta por esos píxeles activos, se usan:

- `thr50_recall`
- `thr75_recall`

Estas métricas complementarias son útiles como análisis de sensibilidad o anexo. En el texto principal se priorizan `top_mask_area_dice`, activación por zonas, `pointing_game_hit`, distancia al centroide y `gini` para evitar una tabla demasiado redundante.

Dice, IoU, precisión y recall se calculan con `sklearn.metrics`. Las métricas específicas del estudio, como `top_mask_area`, activación por zonas o distancia entre centroides, se mantienen implementadas en el código porque dependen directamente de la pregunta del TFM.

## Hilo de interpretación

### 1. Mapas frente a máscara

Primero se evalúa si Grad-CAM, Grad-CAM++ y LRP concentran relevancia en la máscara tumoral.

Lectura esperada:

- si el mapa cae en tumor/peritumor, hay evidencia espacial compatible con dependencia lesional;
- si el mapa cae fuera, no se puede concluir automáticamente shortcut, pero sí aparece una alerta.

### 2. Oclusión

Después se ocluye tumor y peritumor.

Lectura esperada:

- si al quitar tumor baja la confianza, el tumor tiene efecto causal sobre la predicción;
- si no baja, la decisión puede depender de contexto, anatomía global, textura o señales no capturadas por la máscara.

### 3. Análisis de shortcuts en train

Con las imágenes de entrenamiento se buscan regularidades discriminativas por:

- clase;
- plano;
- clase + plano.

Esto permite formular hipótesis. Por ejemplo:

- glioma podría estar asociado a imágenes más oscuras o menor contraste tumor-peritumor;
- meningioma podría estar asociado a mayor contraste tumoral;
- pituitary podría estar asociado a tumor más centrado, mayor área visible o contexto anatómico.

### 4. Comprobación en test

Luego se calculan las mismas métricas en test para ver si las señales detectadas en train se mantienen.

Si una señal aparece en train pero no en test, es débil como explicación del rendimiento. Si aparece en ambos, es una regularidad estable del dataset y gana plausibilidad como shortcut candidato.

### 5. Cruce con predicciones

El paso final es cruzar las métricas de test con las predicciones por imagen y semilla.

Para ello `run_tfm_brisc_vgg_slurm.sh` guarda un CSV opcional de predicciones por muestra:

- `seed`;
- `test_index`;
- `image_path`;
- `true_label`;
- `plane`;
- `multi_pred`;
- `ova_pred`;
- probabilidades por clase;
- acierto/error;
- tipo de resultado (`both_correct`, `multi_correct_ova_wrong`, `multi_wrong_ova_correct`, `both_wrong`).

El cruce se hace con:

- `scripts/analisis_errores_dataset_explicabilidad.py`

La pregunta es:

> ¿Las imágenes que el modelo predice como una clase tienen precisamente las características candidatas a shortcut de esa clase?

Ejemplos:

- falsos positivos de pituitary con tumor muy centrado o área visible alta;
- falsos positivos de glioma en imágenes más oscuras;
- falsos negativos de meningioma con menor contraste tumoral;
- casos donde OVA acierta y Multi falla asociados a señales clase-específicas.

## Resultados agregados actuales

### Rendimiento predictivo

En BRISC, OVA mejora el rendimiento predictivo frente a Multi en la tanda de 10 semillas:

- `accuracy`: Multi `0.9066`, OVA `0.9303`;
- `f1_macro`: Multi `0.9054`, OVA `0.9306`.

Por clase, la mejora se observa como aumento de sensibilidad/recall:

- `glioma`: Multi `0.876`, OVA `0.892`;
- `meningioma`: Multi `0.826`, OVA `0.876`;
- `no_tumor`: Multi `0.996`, OVA `0.996`;
- `pituitary`: Multi `0.973`, OVA `0.988`.

Además, OVA reduce errores tumorales hacia `no_tumor`, un punto clínicamente relevante:

- Multi: `304` errores tumor -> `no_tumor`;
- OVA: `179` errores tumor -> `no_tumor`.

Estos conteos están agregados sobre 10 semillas.

### Explicabilidad espacial

El resultado global de XAI no indica que OVA localice mejor el tumor. Al contrario, LRP, que fue el método con mayor alineación tumoral, favorece generalmente a Multi:

| Método | Modelo | `top_mask_area_dice` | `tumor_activation_frac` | `outside_peritumor_r5_activation_frac` | `pointing_game_hit` |
|---|---|---:|---:|---:|---:|
| Grad-CAM | Multi | 0.0498 | 0.0376 | 0.9200 | 0.0636 |
| Grad-CAM | OVA | 0.0286 | 0.0220 | 0.9549 | 0.0363 |
| Grad-CAM++ | Multi | 0.0412 | 0.0259 | 0.9470 | 0.0538 |
| Grad-CAM++ | OVA | 0.0410 | 0.0294 | 0.9409 | 0.0495 |
| LRP | Multi | 0.1134 | 0.0803 | 0.9003 | 0.1793 |
| LRP | OVA | 0.0856 | 0.0740 | 0.9068 | 0.1345 |

Por clase, LRP muestra:

- `meningioma`: mayor alineación tumoral, con `top_mask_area_dice` aproximado de `0.262` en Multi y `0.210` en OVA;
- `glioma`: baja alineación tumoral, con Dice cercano a `0.03`;
- `pituitary`: alineación muy baja en OVA, con Dice cercano a `0.005`.

### Oclusión

La oclusión tumoral/peritumoral confirma que la dependencia de la región tumoral no es homogénea:

| Clase | Multi drop clase real | OVA drop clase real | Lectura |
|---|---:|---:|---|
| glioma | -0.021 | -0.012 | Ocluir tumor/peritumor apenas perjudica |
| meningioma | 0.314 | 0.220 | Fuerte sensibilidad a la región tumoral |
| pituitary | -0.001 | -0.006 | Ocluir tumor/peritumor casi no afecta |

Globalmente, OVA es más robusto a la oclusión:

- Multi: caída media `0.105`, cambio de predicción `15.3%`;
- OVA: caída media `0.073`, cambio de predicción `12.1%`.

Esta robustez tiene doble lectura: puede ser estabilidad, pero también puede indicar que el modelo no dependía principalmente de la máscara tumoral.

### Shortcuts en train y test

Las señales descriptivas principales son:

- `glioma`: menor contraste tumor-peritumor, tumor más oscuro, imagen global más oscura y contexto exterior más oscuro. El patrón es especialmente fuerte en `glioma axial`.
- `meningioma`: tumor más brillante y mayor contraste frente al exterior/peritumor. Estas señales son tumorales y encajan con XAI y oclusión.
- `pituitary`: mayor área visible, tumor/bbox más centrados y contexto exterior más intenso. Las señales son especialmente marcadas en `pituitary coronal` y `pituitary sagittal`.
- `no_tumor`: señales de intensidad global y posición/área visible, sin variables tumorales por ausencia de máscara.

Las regularidades más relevantes se mantienen en test en la misma dirección. Por ejemplo:

| Grupo | Señal | Train `d` | Test `d` | Lectura |
|---|---|---:|---:|---|
| glioma | `tumor_vs_peritumor_r5_mean_diff` | -1.77 | -1.58 | menor contraste tumor-peritumor |
| glioma | `image_intensity_mean` | -1.33 | -1.20 | imagen más oscura |
| meningioma | `tumor_vs_outside_r10_mean_diff` | 1.69 | 1.58 | tumor más contrastado |
| meningioma | `tumor_vs_peritumor_r15_mean_diff` | 1.39 | 1.45 | tumor más separado del peritumor |
| pituitary | `image_nonzero_area_frac` | 1.67 | 1.31 | mayor área visible |
| pituitary | `mask_centroid_distance_center_norm` | -1.46 | -1.53 | tumor más centrado |
| pituitary sagittal | `outside_peritumor_r15_intensity_p50` | 2.31 | 1.78 | contexto exterior más intenso |
| pituitary coronal | `image_nonzero_area_frac` | 2.09 | 1.94 | mayor área visible |

Estas señales no prueban por sí solas que el modelo las use. Se interpretan como candidatos a shortcut cuando coinciden con baja alineación tumoral y baja sensibilidad a la oclusión.

## Conclusión integrada

La conclusión principal no es que OVA sea más explicable, sino que OVA mejora el rendimiento predictivo mientras reduce la dependencia aparente de la región tumoral en algunas clases.

La lectura por clase es:

- `meningioma`: patrón compatible con dependencia tumoral real. Las señales discriminativas son tumorales, LRP se alinea mejor con la máscara y la oclusión produce una caída clara.
- `glioma`: patrón compatible con posible dependencia de señales globales o contextuales. Hay baja alineación tumoral, baja sensibilidad a la oclusión y señales de oscuridad global/contextual en train y test.
- `pituitary`: candidato más fuerte a shortcut contextual, sobre todo en OVA. Hay buen rendimiento, muy baja alineación tumoral, oclusión casi nula y señales estables de área visible, centrado y contexto exterior.

Formulación prudente para la memoria:

> OVA mejora el rendimiento predictivo en BRISC, pero esta mejora no implica una mejor focalización anatómica. La combinación de explicabilidad, oclusión y análisis del dataset sugiere que parte de la ventaja de OVA podría estar asociada a la explotación de señales discriminativas no estrictamente tumorales, especialmente en glioma y pituitary. En cambio, meningioma presenta un patrón más coherente con dependencia tumoral real.

## Lectura actual por clase

### Glioma

En train y test aparecen señales compatibles con:

- menor contraste tumor-peritumor;
- tumor más oscuro;
- imagen global más oscura;
- contexto fuera del tumor/peritumor más oscuro;
- patrón especialmente marcado en axial.

Esto encaja con la hipótesis de que glioma podría depender de contexto, intensidad global o características de adquisición/plano, más que de una máscara tumoral claramente delimitada.

### Meningioma

En train y test aparecen señales más directamente tumorales:

- mayor contraste tumor-vs-fuera;
- tumor más brillante;
- mayor contraste tumor-vs-peritumor.

Esto encaja con los resultados de oclusión: meningioma es la clase donde quitar tumor/peritumor reduce más claramente la confianza.

### Pituitary

En train y test aparecen señales compatibles con contexto y posición:

- tumor y bbox más centrados;
- mayor área visible/no negra;
- patrones de contexto fuera del peritumor;
- señales fuertes en coronal y sagittal.

Esto no significa necesariamente un shortcut espurio: puede reflejar anatomía real de la región selar. Pero si la oclusión tumoral apenas cambia la predicción, estas variables son candidatas razonables para explicar por qué el modelo acierta sin depender mucho de la máscara tumoral estricta.

## Papel de OVA

La utilidad de OVA se plantea como auditoría clase-específica, no como garantía de mejor localización espacial.

En OVA hay clasificadores separados:

- glioma vs resto;
- meningioma vs resto;
- pituitary vs resto;
- no_tumor vs resto.

Cada clasificador puede explotar señales distintas. Por eso OVA permite preguntar:

- qué usa el clasificador de glioma para decir "glioma";
- qué usa el de pituitary para decir "pituitary";
- si esas señales coinciden con tumor/peritumor o con contexto;
- si los errores se explican por regularidades clase-específicas.

La tesis prudente es:

> OVA no muestra necesariamente mejor explicabilidad espacial que Multi. Su valor está en que permite auditar por separado las hipótesis de clase y detectar posibles shortcuts específicos.

## Comandos principales

Grad-CAM:

```bash
DATASET=brisc SEEDS="1 2 3 4 5 6 7 8 9 10" \
sbatch --nodelist=zeus scripts/run_explicabilidad_gradcam_slurm.sh
```

Grad-CAM++:

```bash
DATASET=brisc CAM_METHOD="gradcam++" SEEDS="1 2 3 4 5 6 7 8 9 10" \
sbatch --nodelist=zeus scripts/run_explicabilidad_gradcam_slurm.sh
```

LRP:

```bash
DATASET=brisc SEEDS="1 2 3 4 5 6 7 8 9 10" \
sbatch --nodelist=zeus scripts/run_explicabilidad_lrp_slurm.sh
```

Oclusión:

```bash
SEEDS="1 2 3 4 5 6 7 8 9 10" OCCLUSION_RADII="0 5 10 15" RING_RADIUS=15 \
sbatch --nodelist=zeus scripts/run_oclusion_tumor_brisc_slurm.sh
```

Análisis de dataset solo test:

```bash
OUTPUT_DIR="resultados_actualizados/analisis_dataset/brisc_test" \
SPLITS="test" \
PERITUMOR_RADII="5 10 15" \
sbatch scripts/run_analisis_dataset_brisc_slurm.sh
```

BRISC predictivo con predicciones por imagen:

```bash
RESULT_DIR="resultados_actualizados/secuencial_con_predicciones" \
PREDICTIONS_CSV="resultados_actualizados/secuencial_con_predicciones/brisc_test_predictions_10seeds.csv" \
SEEDS="1 2 3 4 5 6 7 8 9 10" \
COUPLING_MODES="multi-output ova" \
sbatch --nodelist=zeus scripts/run_tfm_brisc_vgg_slurm.sh
```

Cruce de predicciones con descriptores de test:

```bash
python scripts/analisis_errores_dataset_explicabilidad.py \
  --dataset-csv resultados_actualizados/analisis_dataset/brisc_test/brisc_dataset_por_imagen.csv \
  --prediction-csv resultados_actualizados/secuencial_con_predicciones/brisc_test_predictions_10seeds.csv \
  --output-dir resultados_actualizados/analisis_dataset/brisc_predicciones_shortcuts
```

Resumen de shortcuts desde contrastes:

```bash
python scripts/resumir_shortcuts_brisc.py \
  --input-dir resultados_actualizados/analisis_dataset/brisc_train
```

## Estado operativo

Los resultados finales deben interpretarse con las versiones actuales:

- Grad-CAM/Grad-CAM++ con `pytorch-grad-cam`;
- LRP con `zennit`;
- oclusión con dilatación `skimage.disk`;
- morfología de máscaras con `skimage.regionprops`;
- predicciones por imagen desde el flujo predictivo principal.

Las carpetas antiguas generadas antes de migrar a librerías validadas deben tratarse como resultados históricos o exploratorios, no como resultados finales del TFM.
