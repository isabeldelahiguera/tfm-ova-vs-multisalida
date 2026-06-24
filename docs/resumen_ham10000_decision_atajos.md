# HAM10000: decision tras sampler balanced, explicabilidad y atajos

Este documento resume las tres piezas que conviene llevar al TFM tras los ultimos experimentos con HAM10000:

1. fijar que configuracion predictiva es mas informativa;
2. resumir la explicabilidad global con Grad-CAM, Grad-CAM++ y LRP;
3. interpretar CRP y el analisis de posibles atajos de OVA.

## 1. Configuraciones predictivas

Los resultados siguientes son medias en test interno salvo que se indique lo contrario.

| Configuracion | Modelo | Accuracy | Balanced accuracy | F1 macro | Lectura |
|---|---:|---:|---:|---:|---|
| VGG scratch [32,64,128] | Multi | 0.743 | 0.435 | 0.450 | Baseline comparable con el resto del trabajo |
| VGG scratch [32,64,128] | OVA | 0.737 | 0.392 | 0.409 | OVA queda por debajo en macro |
| VGG16 block5 sin pesos | Multi | 0.793 | 0.578 | 0.580 | Transfer mejora claramente el baseline |
| VGG16 block5 sin pesos | OVA | 0.798 | 0.566 | 0.594 | OVA compite e incluso mejora F1, pero con menor balanced accuracy |
| VGG16 block5 loss balanced | Multi | 0.709 | 0.681 | 0.573 | La ponderacion mejora sensibilidad macro, penalizando accuracy |
| VGG16 block5 loss balanced | OVA | 0.700 | 0.615 | 0.497 | La ponderacion perjudica a OVA |
| VGG16 block5 sampler balanced | Multi | 0.749 | 0.650 | 0.620 | Mejor compromiso justo para comparar en HAM10000 |
| VGG16 block5 sampler balanced | OVA | 0.773 | 0.648 | 0.622 | OVA queda practicamente empatado en macro y mejor en accuracy/F1 |

La configuracion mas util para el analisis final es **VGG16 block5 + sampler balanced + sin pesos en la perdida**. Motivo: aplica el mismo mecanismo de balanceo a Multi y OVA, evita que cada clasificador OVA tenga una escala de perdida distinta, y deja a ambas formulaciones casi empatadas predictivamente.

Esto es mejor para la pregunta del TFM que la perdida `balanced`: con `class_weighting=balanced`, Multi mejora la sensibilidad macro, pero OVA queda claramente penalizado. Con sampler balanced, en cambio, la comparacion es mas limpia porque ambos ven batches con mayor presencia de clases minoritarias sin cambiar la funcion de perdida de forma distinta para cada salida binaria.

## 2. Explicabilidad global en la configuracion seleccionada

Configuracion: VGG16 block5, `TRAIN_SAMPLER=balanced`, `CLASS_WEIGHTING=none`, seed 1, test completo de 1505 imagenes.

| Metodo | Modelo | Inside lesion | Outside lesion | Pointing game | Dice | IoU |
|---|---:|---:|---:|---:|---:|---:|
| Grad-CAM | Multi | 0.606 | 0.394 | 0.886 | 0.686 | 0.554 |
| Grad-CAM | OVA | 0.318 | 0.682 | 0.314 | 0.331 | 0.251 |
| Grad-CAM++ | Multi | 0.697 | 0.303 | 0.940 | 0.735 | 0.605 |
| Grad-CAM++ | OVA | 0.399 | 0.601 | 0.423 | 0.404 | 0.311 |
| LRP | Multi | 0.588 | 0.412 | 0.635 | 0.621 | 0.470 |
| LRP | OVA | 0.375 | 0.625 | 0.323 | 0.415 | 0.300 |

La conclusion principal es fuerte: **aunque Multi y OVA quedan casi empatados en rendimiento predictivo con sampler balanced, Multi produce mapas mucho mas alineados con la lesion en los tres metodos de explicabilidad**.

Por tanto, HAM10000 si aporta algo distinto al TFM: no solo pregunta si OVA y Multi predicen parecido, sino si llegan a esas predicciones apoyandose en regiones visuales similares. Aqui la respuesta parece ser que no siempre.

## 3. CRP y posibles atajos de OVA

Se uso CRP oficial mediante `zennit-crp`, no una implementacion manual. La seleccion de casos se hizo a partir de Grad-CAM++ porque es el metodo que dio mapas mas limpios y mejor alineados globalmente con la mascara.

Configuracion CRP: VGG16 block5 + sampler balanced, seed 1, `PER_GROUP=20`, `TOP_CHANNELS=5`.

| Grupo CRP | Modelo | Inside canal | Pointing | Dice | Lectura |
|---|---:|---:|---:|---:|---|
| both_correct_ova_low_inside | Multi | 0.332 | 0.390 | 0.363 | Ambos aciertan, pero Multi usa canales mas lesionarios |
| both_correct_ova_low_inside | OVA | 0.106 | 0.090 | 0.118 | OVA acierta con canales mucho menos centrados en lesion |
| minority_ova_low_inside | Multi | 0.394 | 0.570 | 0.343 | En minoritarias, Multi mantiene mas foco tumoral |
| minority_ova_low_inside | OVA | 0.137 | 0.290 | 0.132 | OVA muestra baja alineacion con lesion |
| multi_correct_ova_wrong | Multi | 0.786 | 0.710 | 0.693 | Multi acierta mirando dentro |
| multi_correct_ova_wrong | OVA | 0.820 | 0.820 | 0.699 | OVA tambien mira dentro, pero se equivoca |
| ova_correct_multi_wrong | Multi | 0.467 | 0.460 | 0.437 | Diferencia pequena |
| ova_correct_multi_wrong | OVA | 0.455 | 0.440 | 0.423 | Diferencia pequena |
| ova_low_inside | Multi | 0.329 | 0.410 | 0.363 | Multi mejor que OVA en casos donde OVA localiza mal |
| ova_low_inside | OVA | 0.099 | 0.090 | 0.112 | OVA usa canales muy poco alineados con la lesion |

CRP matiza la interpretacion. No todos los errores de OVA se explican por mirar fuera de la lesion: en `multi_correct_ova_wrong`, OVA tambien cae dentro de la mascara pero se equivoca. Esto sugiere que OVA puede usar informacion lesionaria, pero posiblemente menos discriminativa o mas confundible entre clases.

En cambio, en grupos como `both_correct_ova_low_inside`, `minority_ova_low_inside` y `ova_low_inside`, OVA acierta o compite usando canales mucho menos centrados en tumor. Esa es la evidencia mas interesante para discutir posibles atajos.

## Analisis espacial de posibles atajos

El analisis de atajos compara cuanta activacion cae dentro de la mascara, fuera de la mascara y en la region exterior/peritumoral. Los resultados globales muestran que OVA concentra bastante mas activacion fuera de la lesion.

| Metodo | Modelo | Inside lesion | Outside lesion | Outside peritumor r=5 |
|---|---:|---:|---:|---:|
| Grad-CAM | Multi | 0.606 | 0.394 | 0.336 |
| Grad-CAM | OVA | 0.318 | 0.682 | 0.643 |
| Grad-CAM++ | Multi | 0.697 | 0.303 | 0.243 |
| Grad-CAM++ | OVA | 0.399 | 0.601 | 0.559 |
| LRP | Multi | 0.588 | 0.412 | 0.373 |
| LRP | OVA | 0.375 | 0.625 | 0.598 |

La diferencia por clase es heterogenea. Las mayores diferencias Multi-OVA aparecen sobre todo en `bcc`, `nv` y `akiec`. En `vasc` la diferencia es pequena e incluso Grad-CAM++ da a OVA algo mas dentro de lesion.

Esto no demuestra automaticamente que OVA use atajos, pero si deja candidatos razonables: piel sana, contexto perilesional, bordes, iluminacion, pelos o artefactos de adquisicion. La hoja visual generada para revisar casos candidatos esta en:

`resultados_actualizados/explicabilidad/ham10000/analisis_atajos_sampler_balanced/ova_shortcut_candidate_contact_sheet_top25.png`

## Decision para el TFM

Usaria HAM10000 como bloque complementario de explicabilidad, no como tercer dataset principal de equivalencia estadistica.

Configuraciones a mantener:

| Uso | Configuracion |
|---|---|
| Baseline historico | VGG scratch [32,64,128] |
| Transfer learning directa | VGG16 block5 sin pesos |
| Configuracion principal para explicabilidad | VGG16 block5 + sampler balanced |

La narrativa recomendada es:

- En HAM10000, entrenar desde cero no fue suficiente para obtener mapas y predicciones solidas.
- VGG16 preentrenada permite un rendimiento mas razonable sin introducir conexiones residuales.
- La perdida balanceada mejora Multi pero perjudica OVA, probablemente porque altera de forma distinta cada problema binario.
- El sampler balanced es la alternativa mas limpia: mejora la presencia de minoritarias en entrenamiento sin cambiar de forma asimetrica las perdidas.
- Con sampler balanced, Multi y OVA predicen de forma muy parecida, pero sus mapas no son equivalentes.
- Multi se alinea mucho mas con la mascara de lesion; OVA mantiene rendimiento parecido usando con mas frecuencia regiones externas o peritumorales.

Conclusiones prudentes:

- No afirmar que OVA "usa atajos" de forma definitiva.
- Si afirmar que en HAM10000 hay evidencia de **desacoplamiento entre rendimiento predictivo y alineacion espacial**.
- Si afirmar que OVA puede alcanzar metricas predictivas parecidas a Multi sin apoyarse visualmente en la lesion con la misma intensidad.
- Proponer como extension revisar manualmente los casos candidatos y estudiar si las zonas externas contienen artefactos, pelos, piel sana, bordes o patrones de adquisicion.
