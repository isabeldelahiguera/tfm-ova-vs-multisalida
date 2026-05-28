# Resumen de Explicabilidad

Este documento resume el estado actual del análisis de explicabilidad, qué se está midiendo y cómo deben interpretarse los resultados.

## Objetivo

El objetivo es comparar las explicaciones de los modelos `multi-output` y `OVA` mediante mapas Grad-CAM/Grad-CAM++, evaluando si las regiones relevantes para la predicción se alinean con la lesión cuando existe máscara, y describiendo la compactación del mapa cuando no existe anotación espacial.

## Resumen Ejecutivo

Se han probado varias formas de explicar los modelos:

- Grad-CAM clásico.
- Grad-CAM++.
- Grad-CAM++ en varias capas (`last`, `features.7`, `features.2`).
- Grad-CAM++ en tres semillas (`1`, `2`, `3`).
- RISE, basado en perturbaciones aleatorias.
- Oclusión de máscara tumoral exacta.
- Oclusión de máscara dilatada con radios `5`, `10` y `15`.
- Métricas por zona: tumor, peritumor y fuera de tumor/peritumor.
- Métricas específicas para OVA por hipótesis de clase.

La conclusión principal es:

> La activación explicativa cae mayoritariamente fuera de la máscara tumoral y también fuera del peritumor inmediato. Esto ocurre con Grad-CAM, Grad-CAM++, varias capas, varias semillas y RISE. Por tanto, no se puede afirmar que los modelos, y en particular OVA, sean espacialmente explicables en el sentido de localizar bien el tumor.

Los resultados sí muestran una diferencia importante por clase:

- `meningioma`: es la clase donde más evidencia hay de dependencia tumoral/peritumoral. La oclusión produce caídas claras y RISE obtiene mejor solapamiento con la máscara.
- `glioma`: la activación suele caer fuera del tumor; con oclusión dilatada aparece algo de efecto, pero débil.
- `pituitary`: la activación y la oclusión sugieren que el modelo puede estar usando contexto anatómico/global más que la máscara tumoral exacta.

Respecto a Multi vs OVA:

> OVA no muestra una ventaja espacial robusta frente a Multi. Aunque OVA puede ser útil para auditar hipótesis por clase, la separación en clasificadores binarios no se traduce automáticamente en mapas mejor alineados con el tumor.

La lectura metodológica para el TFM es:

> La buena clasificación no implica buena localización explicativa. La explicabilidad espacial debe validarse empíricamente, porque los modelos pueden apoyarse en tumor, peritumor, contexto anatómico, textura global o sesgos del dataset.

## BRISC

En BRISC sí hay máscaras tumorales, por lo que se puede hacer una evaluación espacial de las explicaciones.

### Clase Objetivo Explicada

Grad-CAM se calcula respecto a una única salida objetivo, no usando todos los logits a la vez.

- En el modo `predicted`, se explica la clase predicha por cada modelo.
- En el modo `true`, se explica la etiqueta verdadera.

Para `multi-output`, se usa el logit de la clase objetivo. Para `OVA`, se usa el clasificador binario correspondiente a esa clase y su salida positiva.

Esto mantiene el análisis clase-específico. Como análisis complementario, se añadió la opción de calcular métricas para todas las clases sin agregarlas:

```bash
--include-all-class-metrics
```

Esto genera:

- `gradcam_all_class_metrics.csv`
- `gradcam_all_class_metrics_summary.csv`

La idea es comparar si el Grad-CAM de la clase verdadera localiza mejor que los Grad-CAM de clases no verdaderas, sin mezclar mapas de clases activas y no activas.

### Métricas Iniciales

Las métricas originales incluían:

- `cam_active_area_frac_50`
- `cam_active_area_frac_75`
- `cam_gini`
- `cam_inside_frac`
- `cam_outside_frac`
- `cam_pointing_game_hit`
- `cam_top10_*`
- `cam_top20_*`
- `cam_top30_*`

Las métricas `top10/top20/top30` seleccionan el 10%, 20% o 30% de píxeles más activados del Grad-CAM y comparan esa región con la máscara tumoral.

Problema detectado: las máscaras tumorales ocupan muy poca área. En el estudio de tamaño tumoral:

- mediana global: 1.43% de la imagen;
- percentil 95: 6.11%;
- meningioma es la clase con mayor área relativa, pero aun así la mediana es 2.31%.

Por tanto, `top10` y especialmente `top20` seleccionan regiones bastante mayores que el tumor en la mayoría de imágenes.

### Métrica Adaptativa

Para corregir ese sesgo se añadió una métrica adaptativa:

- `cam_top_mask_area_dice`
- `cam_top_mask_area_iou`
- `cam_top_mask_area_precision`
- `cam_top_mask_area_recall`
- `cam_top_mask_area_outside_precision`

Funcionamiento:

1. Se cuenta cuántos píxeles positivos tiene la máscara tumoral.
2. Se selecciona el mismo número de píxeles más calientes del Grad-CAM.
3. Se compara esa región con la máscara.

Así, si la máscara ocupa 39 píxeles, se seleccionan los 39 píxeles más activados del Grad-CAM. Esta métrica es más justa para lesiones pequeñas que usar un porcentaje fijo.

### Activación Fija 0.5 y 0.75

También se añadieron métricas para evaluar si la zona activa del mapa cae dentro del tumor:

- `cam_thr50_dice`
- `cam_thr50_iou`
- `cam_thr50_precision`
- `cam_thr50_recall`
- `cam_thr50_outside_precision`
- `cam_thr75_dice`
- `cam_thr75_iou`
- `cam_thr75_precision`
- `cam_thr75_recall`
- `cam_thr75_outside_precision`

Estas métricas binarizan el Grad-CAM con umbral fijo:

- `CAM >= 0.5`
- `CAM >= 0.75`

y comparan esa región activa con la máscara.

Esto complementa `cam_active_area_frac_50`, que por sí sola solo mide qué porcentaje de la imagen tiene activación mayor o igual a 0.5, sin saber si cae dentro o fuera del tumor.

### Activación Por Zonas

Para estudiar si la decisión puede depender de contexto anatómico o región peritumoral, se añadieron métricas por zonas:

- `cam_tumor_activation_frac`
- `cam_peritumor_r5_activation_frac`
- `cam_outside_peritumor_r5_activation_frac`
- `cam_tumor_mean`
- `cam_peritumor_r5_mean`
- `cam_outside_peritumor_r5_mean`

Las zonas son:

- `tumor`: máscara original.
- `peritumor_r5`: anillo alrededor de la máscara obtenido dilatando 5 píxeles y restando la máscara original.
- `outside_peritumor_r5`: resto de la imagen fuera de tumor + peritumor.

Estas métricas permiten comprobar si la activación se concentra en la lesión estricta, en su entorno inmediato o en zonas más alejadas. Esto es relevante porque la diferenciación entre tumores puede depender también de información contextual, no solo del interior segmentado.

### Resultados Con Grad-CAM

Con Grad-CAM clásico y la métrica adaptativa `top_mask_area`, el resultado global en modo predicción fue:

| Métrica | Multi | OVA | Mejor |
| --- | ---: | ---: | --- |
| `top_mask_area_dice` | 0.078 | 0.052 | Multi |
| `top_mask_area_iou` | 0.050 | 0.031 | Multi |
| `top_mask_area_outside_precision` | 0.922 | 0.948 | Multi |

Como `outside_precision` cuanto menor mejor, multi-output muestra menor activación fuera de la máscara.

Por clase:

- Glioma: ambos modelos localizan muy mal, aunque multi queda algo por encima.
- Meningioma: multi suele mejorar a OVA, aunque OVA es competitivo.
- Pituitary: multi queda claramente por encima de OVA.

Interpretación:

> Multi-output no clasifica necesariamente mejor en todos los casos, pero sus máximas activaciones se alinean algo más con las máscaras tumorales que las de OVA. OVA produce mapas más compactos en algunas métricas, pero esa compactación no implica necesariamente mejor alineación con el tumor.

### Resultados Con Grad-CAM++

Se añadió Grad-CAM++ mediante:

```bash
CAM_METHOD=gradcam++
```

Los resultados se guardan en:

- `resultados_actualizados/explicabilidad/brisc/seed_1_gradcampp/`
- `resultados_actualizados/explicabilidad/brisc/seed_1_true_target_gradcampp/`

Grad-CAM++ produce mapas más compactos y reduce mucho la activación total fuera de la máscara.

Ejemplo global en modo predicción:

| Métrica | Grad-CAM Multi | Grad-CAM++ Multi |
| --- | ---: | ---: |
| `outside_frac` | 0.953 | 0.773 |
| `inside_frac` | 0.047 | 0.087 |
| `active_area_frac_50` | 0.073 | 0.003 |
| `gini` | 0.730 | 0.851 |

Sin embargo, la mejora en solapamiento adaptativo es limitada:

| Métrica | Multi | OVA | Mejor |
| --- | ---: | ---: | --- |
| `top_mask_area_dice` | 0.082 | 0.052 | Multi |
| `top_mask_area_iou` | 0.050 | 0.031 | Multi |
| `top_mask_area_outside_precision` | 0.918 | 0.948 | Multi |

Interpretación:

> Grad-CAM++ reduce la dispersión global del mapa, pero no cambia la conclusión principal: la alineación exacta con la máscara sigue siendo baja y multi-output continúa siendo globalmente más favorable que OVA en localización.

### Oclusión

El análisis de oclusión intenta responder una pregunta causal:

> Si se tapa el tumor, ¿baja la probabilidad de la clase verdadera o cambia la predicción?

El script `scripts/oclusion_tumor_brisc.py` sustituye la región tumoral por la media local de un anillo alrededor de la lesión.

Resultados previos con oclusión exacta de máscara:

| Clase | Drop Multi | Drop OVA | Interpretación |
| --- | ---: | ---: | --- |
| Glioma | -0.028 | -0.009 | No hay caída clara |
| Meningioma | 0.315 | 0.305 | Efecto fuerte |
| Pituitary | -0.000 | -0.007 | No hay caída clara |

Esto sugiere que la oclusión exacta de la máscara solo muestra dependencia clara de la región tumoral en meningioma. Para glioma y pituitary, el modelo podría estar usando información contextual, peritumoral o la oclusión puede ser demasiado localizada.

Por ello se añadió una oclusión más fuerte con máscara dilatada:

```bash
OCCLUSION_RADII="0 5 10 15"
```

Ahora se guardan:

- `occlusion_radius`
- `mask_area_frac`
- `occlusion_area_frac`

La finalidad es comprobar si al tapar tumor + entorno peritumoral empieza a caer la confianza en clases donde la máscara estricta no producía efecto.

Resultados con radios `0, 5, 10, 15`:

| Radio | Drop Multi | Drop OVA | Cambio pred. Multi | Cambio pred. OVA |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.104 | 0.104 | 13.6% | 10.5% |
| 5 | 0.101 | 0.104 | 15.0% | 12.7% |
| 10 | 0.107 | 0.142 | 16.4% | 15.2% |
| 15 | 0.123 | 0.151 | 18.4% | 15.9% |

Al aumentar la región ocluida, el efecto global aumenta, especialmente en OVA. Esto sugiere que el entorno tumoral o contexto inmediato aporta información para la decisión.

Resultados por clase:

| Clase | Radio | Drop Multi | Drop OVA | Cambio pred. Multi | Cambio pred. OVA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Glioma | 0 | -0.028 | -0.009 | 1.6% | 2.8% |
| Glioma | 5 | -0.037 | -0.016 | 4.7% | 3.9% |
| Glioma | 10 | -0.020 | 0.028 | 7.1% | 5.1% |
| Glioma | 15 | -0.001 | 0.051 | 10.6% | 7.9% |
| Meningioma | 0 | 0.315 | 0.305 | 35.6% | 26.8% |
| Meningioma | 5 | 0.315 | 0.313 | 37.3% | 31.7% |
| Meningioma | 10 | 0.317 | 0.384 | 38.6% | 38.2% |
| Meningioma | 15 | 0.348 | 0.393 | 40.8% | 37.3% |
| Pituitary | 0 | -0.000 | -0.007 | 1.3% | 0.3% |
| Pituitary | 5 | 0.000 | -0.008 | 1.0% | 0.7% |
| Pituitary | 10 | -0.001 | -0.009 | 1.7% | 0.3% |
| Pituitary | 15 | -0.001 | -0.012 | 2.0% | 1.0% |

Lectura:

- En meningioma, la región tumoral/peritumoral tiene un efecto causal claro: al taparla baja mucho la probabilidad de la clase verdadera y cambian muchas predicciones.
- En glioma, la máscara exacta no afecta, pero al dilatar la región aparece cierto efecto, sobre todo en OVA. Esto sugiere que puede haber información relevante en el entorno de la lesión.
- En pituitary, la oclusión apenas afecta incluso con radio 15. Esto puede indicar que la decisión depende de señales anatómicas o contextuales más globales, por ejemplo la localización selar/supraselar, no capturadas por una máscara estricta o una dilatación pequeña.

Estos resultados sí son útiles para el estudio de explicabilidad porque complementan las métricas CAM con una prueba de perturbación. Grad-CAM/Grad-CAM++ describen dónde aparece activación, mientras que la oclusión evalúa si retirar una región cambia la decisión. La combinación permite distinguir entre mapas poco alineados con la máscara y dependencia causal de la región tumoral/peritumoral.

## Posible Ganancia Interpretativa De OVA

Los resultados no apoyan que OVA localice globalmente mejor la máscara tumoral que multi-output. De hecho, en métricas de solapamiento espacial (`top_mask_area`, Dice/IoU, outside), multi-output suele quedar por encima.

La posible ganancia de OVA debe buscarse en otra dimensión:

> OVA permite analizar explicaciones independientes por hipótesis diagnóstica.

En un problema multiclase, el modelo multi-output produce una salida conjunta. En OVA, en cambio, cada clase tiene su propio clasificador binario:

- glioma vs resto;
- meningioma vs resto;
- pituitary vs resto.

Esto permite estudiar para una misma imagen:

- qué región apoya la hipótesis glioma;
- qué región apoya la hipótesis meningioma;
- qué región apoya la hipótesis pituitary;
- si la clase verdadera activa zonas distintas de las clases alternativas;
- si los clasificadores erróneos miran a regiones no plausibles.

Por tanto, la ventaja interpretativa de OVA no sería necesariamente una mejor localización media, sino una explicación más descomponible por clase. Esto es especialmente útil para análisis caso a caso, porque permite comparar hipótesis diagnósticas alternativas.

Para buscar esta ganancia de forma cuantitativa, se añadieron:

- métricas por todas las clases (`gradcam_all_class_metrics.csv`);
- métricas por zona (`tumor`, `peritumor_r5`, `outside_peritumor_r5`);
- oclusión dilatada por radio.

Preguntas que se pueden responder:

- En OVA, ¿el clasificador de la clase verdadera concentra más activación en tumor/peritumor que los clasificadores de clases falsas?
- En los casos donde OVA acierta y multi falla, ¿OVA muestra mayor caída por oclusión o mejor activación peritumoral?
- ¿OVA es más sensible a la oclusión dilatada en clases concretas, como meningioma o glioma?

Una formulación prudente sería:

> Aunque OVA no muestra mejor localización global de la máscara, ofrece una estructura más interpretable para el análisis por hipótesis diagnóstica. Cada clasificador OVA permite inspeccionar qué evidencia espacial apoya o descarta una clase concreta, lo que puede aportar valor en análisis cualitativo caso a caso y en comparaciones entre clase verdadera y clases alternativas.

## TB Chest X-Ray

En TB chest X-ray no hay máscaras ni anotaciones espaciales de lesión. Por tanto, no se puede cuantificar localización patológica.

No se debe afirmar:

- que un modelo mira mejor la lesión;
- que localiza mejor la tuberculosis;
- que la activación cae en una región clínicamente correcta.

Solo se pueden describir propiedades globales del mapa:

- área activa;
- compactación;
- dispersión.

Resultados observados:

| Métrica | Multi | OVA | Lectura |
| --- | ---: | ---: | --- |
| `cam_active_area_frac_50` | 0.0128 | 0.0215 | OVA activa más área |
| `cam_active_area_frac_75` | 0.0044 | 0.0062 | OVA activa más área fuerte |
| `cam_gini` | 0.8749 | 0.8316 | Multi más compacto |

Interpretación:

> En TB, multi-output produce mapas más compactos/concentrados, mientras que OVA tiende a activar una mayor proporción de la radiografía. Al no existir máscaras, este análisis es descriptivo y no valida localización clínica.

## Posibles Atajos En OVA

Se analizó si los clasificadores OVA podían estar acertando aunque su explicación espacial no se apoyase principalmente en tumor/peritumor.

Criterio usado:

- se toman solo casos donde OVA acierta la clase;
- se mira el Grad-CAM++ del clasificador OVA de la clase verdadera;
- se mide cuánta activación cae en tumor, peritumor de radio 5 y fuera de tumor/peritumor;
- si la activación fuera de tumor/peritumor es alta y la activación en tumor es baja, se considera un patrón sospechoso de atajo o contexto.

Resultados en casos donde OVA acierta:

| Clase verdadera | n | Activación tumor | Activación peritumor r5 | Activación fuera tumor/peritumor | Dice top-mask-area |
| --- | ---: | ---: | ---: | ---: | ---: |
| glioma | 237 | 0.0003 | 0.0023 | 0.9974 | 0.0005 |
| meningioma | 281 | 0.1757 | 0.0990 | 0.7005 | 0.1342 |
| pituitary | 295 | 0.0198 | 0.0084 | 0.5888 | 0.0185 |
| global | 813 | 0.0680 | 0.0379 | 0.7465 | 0.0532 |

Además, usando un criterio simple de sospecha (`outside_peritumor_r5 >= 0.8` y `tumor_activation < 0.05`), aparecen 538 de 813 casos correctos de OVA, es decir un 66.2%.

Por clase:

| Clase | Casos sospechosos |
| --- | ---: |
| glioma | 237 |
| meningioma | 128 |
| pituitary | 173 |

Lectura:

> El caso más problemático es glioma: OVA acierta muchos casos, pero el mapa de la clase verdadera prácticamente no cae en tumor ni peritumor. Esto sugiere que el clasificador puede estar usando señales de contexto, textura, localización global o artefactos correlacionados con la clase. En meningioma, en cambio, sí hay más evidencia tumoral/peritumoral, coherente con los resultados de oclusión, donde tapar tumor/peritumor afecta más a la predicción.

Esto no invalida OVA como clasificador, pero sí limita la afirmación de que sea más explicable espacialmente. La ganancia de OVA debe formularse como análisis por hipótesis de clase, no como mejor localización tumoral global.

## Robustez Por Semilla Y Capa

Se lanzó un estudio con tres semillas (`1`, `2`, `3`) y tres capas Grad-CAM++:

- `last`: última convolucional (`features.12`);
- `features.7`: capa intermedia;
- `features.2`: capa temprana.

El objetivo era comprobar si las conclusiones dependían de usar solo `seed_1` y la última capa.

Resultados globales, media ± desviación entre semillas:

| Capa | Multi Dice top-mask | OVA Dice top-mask | Multi outside top-mask | OVA outside top-mask | Multi outside peritumor | OVA outside peritumor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `features.2` | 0.0760 ± 0.0468 | 0.0389 ± 0.0147 | 0.9240 | 0.9611 | 0.8539 | 0.8736 |
| `features.7` | 0.0332 ± 0.0016 | 0.0318 ± 0.0085 | 0.9668 | 0.9682 | 0.8806 | 0.8895 |
| `last` | 0.0424 ± 0.0370 | 0.0506 ± 0.0062 | 0.9576 | 0.9494 | 0.4902 | 0.8185 |

Lecturas:

- La capa temprana `features.2` mejora el Dice de Multi, pero los mapas son más difusos y siguen con mucha activación fuera de tumor/peritumor.
- La capa intermedia `features.7` es más estable entre semillas, pero ofrece poco solapamiento con máscara.
- La última capa no es estable: en `seed_3`, Multi produce muchos mapas Grad-CAM++ casi vacíos.

Número de mapas casi vacíos (`active_area_frac_50 == 0`):

| Configuración | Multi | OVA |
| --- | ---: | ---: |
| `seed_1`, `last` | 120 / 860 | 121 / 860 |
| `seed_2`, `last` | 128 / 860 | 62 / 860 |
| `seed_3`, `last` | 719 / 860 | 1 / 860 |
| `seed_1`, `features.7` | 5 / 860 | 91 / 860 |
| `seed_2`, `features.7` | 28 / 860 | 21 / 860 |
| `seed_3`, `features.7` | 144 / 860 | 10 / 860 |
| `seed_1`, `features.2` | 1 / 860 | 56 / 860 |
| `seed_2`, `features.2` | 25 / 860 | 15 / 860 |
| `seed_3`, `features.2` | 135 / 860 | 88 / 860 |

Por clase, el patrón sigue siendo heterogéneo:

| Capa | Clase | Multi Dice | OVA Dice | Multi outside peritumor | OVA outside peritumor |
| --- | --- | ---: | ---: | ---: | ---: |
| `features.2` | glioma | 0.0390 | 0.0398 | 0.8831 | 0.8913 |
| `features.2` | meningioma | 0.1504 | 0.0672 | 0.8095 | 0.8843 |
| `features.2` | pituitary | 0.0314 | 0.0093 | 0.8746 | 0.8476 |
| `features.7` | glioma | 0.0251 | 0.0399 | 0.8738 | 0.8197 |
| `features.7` | meningioma | 0.0565 | 0.0490 | 0.8703 | 0.9137 |
| `features.7` | pituitary | 0.0162 | 0.0074 | 0.8969 | 0.9239 |
| `last` | glioma | 0.0243 | 0.0218 | 0.5471 | 0.8761 |
| `last` | meningioma | 0.0649 | 0.1151 | 0.3069 | 0.7386 |
| `last` | pituitary | 0.0347 | 0.0091 | 0.6289 | 0.8511 |

Conclusión:

> La explicación espacial es sensible tanto a la semilla como a la capa usada. Usar solo la última capa y una única semilla no es suficiente para defender una conclusión fuerte. Las capas tempranas/intermedias evitan parte de la inestabilidad de la última capa, pero no resuelven el problema principal: la activación sigue cayendo mayoritariamente fuera de tumor/peritumor. No aparece evidencia robusta de que OVA sea más explicable espacialmente que Multi; la posible utilidad de OVA sigue siendo la auditoría por hipótesis de clase.

## Interpretación General

Los resultados muestran que clasificación y localización explicativa no son equivalentes.

Un modelo puede clasificar bien usando:

- la lesión;
- bordes o región peritumoral;
- contexto anatómico;
- textura global;
- patrones correlacionados con la clase;
- sesgos del conjunto de datos.

Por ello, buen rendimiento predictivo no implica necesariamente alto solapamiento Grad-CAM con la máscara tumoral.

La conclusión metodológica prudente es:

> Las métricas CAM muestran una alineación espacial débil con las máscaras tumorales, especialmente porque las lesiones ocupan una fracción muy pequeña de la imagen. Grad-CAM++ mejora la compactación del mapa, pero el solapamiento exacto sigue siendo limitado. La oclusión complementa este análisis evaluando si la región tumoral tiene efecto causal sobre la predicción, aunque los resultados iniciales indican que este efecto es claro principalmente en meningioma.

## Conclusión Actual Para El TFM

La hipótesis inicial era que OVA, al separar el problema en clasificadores binarios independientes, podía aportar explicaciones más claras por clase. Los resultados no apoyan una mejora espacial global de OVA.

Lo que sí se puede defender es:

1. Se ha evaluado la explicabilidad con varios métodos, no solo con un mapa aislado.
2. La localización de las explicaciones es débil respecto a la máscara tumoral exacta.
3. La mayor parte de la activación cae fuera de tumor/peritumor:
   - Grad-CAM clásico: `outside` muy alto, alrededor de 0.92-0.95 en `top_mask_area`.
   - Grad-CAM++: mejora la compactación, pero mantiene `outside` alto, alrededor de 0.92 para Multi y 0.95 para OVA en `top_mask_area` de `seed_1`.
   - Robustez por capas/semillas: el patrón sigue siendo inestable y con mucha activación fuera.
   - RISE: mejora el Dice top-mask, pero es muy difuso y mantiene activación fuera de peritumor alrededor de 0.93.
4. La clase `meningioma` es la excepción parcial: ahí la oclusión y RISE sí indican que la lesión/peritumor tiene más peso.
5. En `glioma` y `pituitary`, los modelos parecen depender más de contexto, textura, posición anatómica o señales no capturadas por la máscara exacta.

La frase más honesta sería:

> Aunque OVA es una formulación más descomponible por clase, esta descomposición no garantiza explicaciones espacialmente más alineadas con el tumor. En este estudio, la explicabilidad espacial es limitada y altamente dependiente de la clase, la semilla, la capa y el método usado. Los resultados apuntan a que los modelos no se apoyan exclusivamente en la región tumoral segmentada, sino también en información fuera del tumor/peritumor.

Por tanto, el valor del análisis no es demostrar que OVA localiza mejor, sino mostrar que:

- la interpretabilidad no se puede asumir por arquitectura;
- hay que validarla con métricas espaciales y perturbaciones;
- el comportamiento explicativo depende de la clase;
- meningioma parece más apoyado en lesión, mientras que glioma y pituitary no.

## Comandos Relevantes

Grad-CAM++ con todas las métricas:

```bash
DATASET=brisc CAM_METHOD=gradcam++ CAM_TARGET=predicted SELECTION=all NUM_IMAGES=860 REQUIRE_MASK=1 INCLUDE_ALL_OVA_CAMS=1 INCLUDE_ALL_CLASS_METRICS=1 sbatch --nodelist=zeus scripts/run_explicabilidad_gradcam_slurm.sh
```

```bash
DATASET=brisc CAM_METHOD=gradcam++ CAM_TARGET=true SELECTION=all NUM_IMAGES=860 REQUIRE_MASK=1 INCLUDE_ALL_OVA_CAMS=1 INCLUDE_ALL_CLASS_METRICS=1 sbatch --nodelist=zeus scripts/run_explicabilidad_gradcam_slurm.sh
```

Oclusión con radios:

```bash
OCCLUSION_RADII="0 5 10 15" RING_RADIUS=15 sbatch --nodelist=zeus scripts/run_oclusion_tumor_brisc_slurm.sh
```

Estudio de robustez por semilla y capa, sin guardar PNGs:

```bash
DATASET=brisc SEEDS="1 2 3" TARGET_LAYERS="last features.7 features.2" CAM_METHOD=gradcam++ CAM_TARGET=predicted SAVE_IMAGES=0 bash scripts/submit_gradcam_seed_layer_grid.sh
```

Las capas corresponden a:

- `last`: última convolucional (`features.12`), más semántica y menos precisa espacialmente;
- `features.7`: bloque convolucional intermedio;
- `features.2`: bloque temprano, con más detalle espacial pero menos semántica de clase.

Cuando acaben los jobs:

```bash
python scripts/analisis_gradcam_semillas_capas.py --dataset brisc
```

Esto genera:

- `resultados_actualizados/analisis_explicabilidad/brisc/gradcam_seed_layer_runs.csv`;
- `resultados_actualizados/analisis_explicabilidad/brisc/gradcam_seed_layer_aggregate.csv`.

RISE como método alternativo basado en perturbaciones aleatorias:

```bash
DATASET=brisc CAM_METHOD=rise CAM_TARGET=predicted SELECTION=all NUM_IMAGES=860 REQUIRE_MASK=1 SAVE_IMAGES=0 RISE_SAMPLES=200 RISE_MASK_SIZE=8 RISE_BATCH_SIZE=64 sbatch --nodelist=zeus scripts/run_explicabilidad_gradcam_slurm.sh
```

RISE no usa capa interna. La idea es generar muchas máscaras aleatorias, ocultar partes de la imagen y estimar qué zonas hacen bajar/subir la confianza. Es más caro que Grad-CAM, pero sirve como comprobación independiente porque no depende de la última capa convolucional.

Para tres semillas:

```bash
for SEED in 1 2 3; do DATASET=brisc SEED=$SEED CAM_METHOD=rise CAM_TARGET=predicted SELECTION=all NUM_IMAGES=860 REQUIRE_MASK=1 SAVE_IMAGES=0 RISE_SAMPLES=200 RISE_MASK_SIZE=8 RISE_BATCH_SIZE=64 sbatch --nodelist=zeus scripts/run_explicabilidad_gradcam_slurm.sh; done
```

### Resultados Con RISE

Se lanzó RISE con tres semillas (`1`, `2`, `3`), `RISE_SAMPLES=200`, `RISE_MASK_SIZE=8`, `CAM_TARGET=predicted` y sin guardar imágenes.

Resultados globales:

| Método | Multi Dice top-mask | OVA Dice top-mask | Multi outside top-mask | OVA outside top-mask | Multi outside peritumor | OVA outside peritumor | Multi Gini | OVA Gini |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RISE | 0.1207 ± 0.0283 | 0.0681 ± 0.0316 | 0.8793 | 0.9319 | 0.9327 | 0.9377 | 0.1910 | 0.1922 |

Comparado con Grad-CAM++, RISE mejora el solapamiento `top_mask_area`, especialmente en Multi, pero produce mapas mucho más difusos. Esto se ve en el Gini bajo (`~0.19`) y en la alta fracción de activación total fuera de tumor/peritumor (`~0.93`).

Por clase:

| Clase | Multi Dice top-mask | OVA Dice top-mask | Multi outside top-mask | OVA outside top-mask | Multi outside peritumor | OVA outside peritumor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| glioma | 0.0177 | 0.0191 | 0.9823 | 0.9809 | 0.9400 | 0.9402 |
| meningioma | 0.3129 | 0.1734 | 0.6871 | 0.8266 | 0.8984 | 0.9101 |
| pituitary | 0.0118 | 0.0022 | 0.9882 | 0.9978 | 0.9614 | 0.9636 |

Lectura:

- RISE refuerza que `meningioma` es la clase donde la región tumoral tiene más peso explicativo.
- En `meningioma`, Multi obtiene un Dice top-mask mucho mayor que OVA.
- En `glioma` y `pituitary`, incluso RISE apenas alinea las zonas importantes con la máscara.
- RISE no resuelve el problema de activación fuera de tumor/peritumor porque genera mapas más suaves y extensos.

Conclusión:

> RISE aporta una comprobación independiente de Grad-CAM/Grad-CAM++. Sus resultados apoyan que la evidencia tumoral/peritumoral es más clara en meningioma, pero no en glioma ni pituitary. Además, tampoco muestra una ventaja espacial global de OVA frente a Multi.
