# HAM10000: atajos espaciales separados por artefacto y por clase

Este analisis separa dos preguntas que no conviene mezclar:

1. si OVA y Multi se apoyan en regiones espaciales distintas de la imagen;
2. si esa diferencia puede asociarse a artefactos visuales concretos o a clases concretas.

La configuracion usada es `VGG16 block5 + TRAIN_SAMPLER=balanced + CLASS_WEIGHTING=none`, porque es la configuracion donde Multi y OVA quedan mas cerca predictivamente. Asi, si la explicabilidad difiere, no se puede atribuir simplemente a que un modelo predice mucho peor que el otro.

Los ficheros generados estan en:

`resultados_actualizados/explicabilidad/ham10000/analisis_atajos_artifacts_sampler_balanced/`

## Ficheros generados

| Fichero | Que contiene |
|---|---|
| `artifact_metrics_by_image.csv` | Proxies de artefactos por imagen de test |
| `artifact_prevalence_by_class.csv` | Frecuencia/media de artefactos por clase |
| `spatial_shortcuts_by_class_contrast.csv` | Diferencias espaciales Multi vs OVA por clase |
| `spatial_shortcuts_by_artifact_contrast.csv` | Diferencias espaciales Multi vs OVA segun artefacto alto/bajo |
| `spatial_shortcuts_by_class_and_artifact.csv` | Cruce por clase y artefacto |

## Proxies de artefactos

HAM10000 no trae etiquetas oficiales de artefactos como pelo, regla, borde negro o reflejo. Por eso no se puede afirmar "esta imagen tiene artefacto X" como ground truth. Lo que se calcula son proxies reproducibles con librerias estandar de imagen:

| Proxy | Interpretacion |
|---|---|
| `artifact_dark_border_frac` | Fraccion de borde oscuro; aproxima marcos negros o viñeteado |
| `artifact_dark_corner_frac` | Fraccion de esquinas oscuras; aproxima esquinas negras/marco |
| `artifact_hair_line_score` | Respuesta de filtro Frangi sobre lineas oscuras fuera de lesion; proxy de pelos/lineas finas |
| `artifact_specular_frac` | Pixeles muy brillantes y poco saturados fuera de lesion; proxy de reflejos |
| `artifact_high_saturation_frac` | Pixeles muy saturados fuera de lesion; proxy de marcas/coloraciones/artefactos cromaticos |
| `artifact_edge_density_outside_lesion` | Densidad de bordes fuera de lesion; proxy general de textura, pelo, marcas o estructuras externas |

Los grupos `high_q75` indican imagenes por encima del percentil 75 de ese proxy. No son etiquetas clinicas ni anotaciones humanas.

## Resultado por clase

Con Grad-CAM++, la diferencia principal por clase queda asi:

| Clase | n | Multi inside | OVA inside | Delta Multi-OVA | Multi Dice | OVA Dice | Lectura |
|---|---:|---:|---:|---:|---:|---:|---|
| akiec | 55 | 0.801 | 0.517 | 0.283 | 0.720 | 0.519 | OVA menos centrado en lesion |
| bcc | 79 | 0.773 | 0.364 | 0.409 | 0.750 | 0.349 | Diferencia muy fuerte |
| bkl | 167 | 0.776 | 0.613 | 0.163 | 0.737 | 0.612 | Diferencia moderada |
| df | 18 | 0.712 | 0.587 | 0.126 | 0.696 | 0.635 | Muestra pequena; diferencia menor |
| mel | 161 | 0.792 | 0.670 | 0.122 | 0.730 | 0.683 | Diferencia menor |
| nv | 1003 | 0.658 | 0.307 | 0.351 | 0.735 | 0.310 | Diferencia muy fuerte y con mucho peso muestral |
| vasc | 22 | 0.612 | 0.643 | -0.031 | 0.829 | 0.779 | OVA no parece peor en inside |

Conclusion por clase: la evidencia de posible dependencia no lesional de OVA no es uniforme. Es especialmente clara en `bcc` y `nv`, tambien visible en `akiec`, y bastante menos clara en `mel`, `df` o `vasc`.

## Resultado por artefacto

Con Grad-CAM++, el contraste por artefacto alto/bajo queda asi:

| Artefacto | Nivel | n | Multi inside | OVA inside | Delta Multi-OVA | OVA outside peritumor | Lectura |
|---|---|---:|---:|---:|---:|---:|---|
| Borde oscuro | bajo/medio | 1272 | 0.707 | 0.384 | 0.323 | 0.576 | OVA mucho mas externo |
| Borde oscuro | alto | 233 | 0.639 | 0.481 | 0.157 | 0.466 | La brecha baja; no explica por si solo el problema |
| Esquinas oscuras | bajo/medio | 1286 | 0.707 | 0.384 | 0.323 | 0.575 | OVA mucho mas externo |
| Esquinas oscuras | alto | 219 | 0.634 | 0.485 | 0.149 | 0.460 | Similar a borde oscuro |
| Lineas tipo pelo | bajo/medio | 1129 | 0.686 | 0.386 | 0.300 | 0.573 | Brecha clara |
| Lineas tipo pelo | alto | 376 | 0.728 | 0.438 | 0.290 | 0.516 | La brecha permanece, pero no aumenta |
| Reflejos | bajo/medio | 1129 | 0.677 | 0.377 | 0.300 | 0.580 | Brecha clara |
| Reflejos | alto | 376 | 0.755 | 0.464 | 0.291 | 0.496 | La brecha permanece |
| Alta saturacion | bajo/medio | 1428 | 0.697 | 0.400 | 0.297 | 0.558 | Brecha clara |
| Alta saturacion | alto | 77 | 0.682 | 0.371 | 0.311 | 0.579 | Pocos casos; no concluyente |
| Bordes fuera de lesion | bajo/medio | 1129 | 0.688 | 0.372 | 0.316 | 0.588 | Brecha fuerte |
| Bordes fuera de lesion | alto | 376 | 0.721 | 0.479 | 0.242 | 0.471 | OVA mejora algo, pero sigue por debajo |

Conclusion por artefacto: no aparece un unico artefacto que explique por si solo la peor localizacion de OVA. La diferencia Multi-OVA existe tanto en imagenes con artefacto bajo/medio como alto. Esto sugiere que el fenomeno es mas general que "OVA mira pelos" o "OVA mira bordes negros".

## Prevalencia de artefactos por clase

Los proxies si varian por clase. Por ejemplo:

| Clase | n | Borde oscuro alto | Pelo/lineas alto | Reflejo alto | Bordes externos alto |
|---|---:|---:|---:|---:|---:|
| akiec | 55 | 0.127 | 0.345 | 0.400 | 0.582 |
| bcc | 79 | 0.215 | 0.380 | 0.304 | 0.342 |
| bkl | 167 | 0.311 | 0.251 | 0.138 | 0.371 |
| df | 18 | 0.222 | 0.222 | 0.556 | 0.389 |
| mel | 161 | 0.354 | 0.211 | 0.261 | 0.342 |
| nv | 1003 | 0.094 | 0.245 | 0.242 | 0.189 |
| vasc | 22 | 0.091 | 0.045 | 0.545 | 0.136 |

Esto es importante: si una clase tiene mas artefactos, un modelo podria aprender correlaciones espurias de esa clase. Pero en los resultados actuales `nv` muestra una brecha espacial grande aunque no tiene especialmente altos los proxies de borde o pelo. Por tanto, en `nv` el problema podria estar mas relacionado con piel/contexto/peritumor o patrones no capturados por estos proxies simples.

## Lectura recomendada

La conclusion prudente seria:

- OVA y Multi son parecidos predictivamente con sampler balanced, pero no son equivalentes espacialmente.
- La diferencia espacial es especialmente fuerte en `bcc`, `nv` y `akiec`.
- No hay evidencia suficiente para atribuirlo a un unico artefacto concreto.
- Si hay evidencia de que OVA usa mas region externa o peritumoral que Multi.
- Los artefactos deben tratarse como hipotesis/candidatos, no como causa demostrada.

Para el TFM, yo lo formularia asi:

> El analisis por clase muestra que la menor alineacion espacial de OVA no se reparte de forma uniforme entre diagnosticos, sino que se concentra especialmente en `bcc`, `nv` y `akiec`. Al cruzarlo con proxies de artefactos visuales, no se observa que un unico tipo de artefacto explique por si solo el fenomeno. La diferencia Multi-OVA persiste tambien en imagenes con bajo nivel de artefactos estimados. Por tanto, los resultados apuntan mas a una dependencia espacial/contextual general de OVA que a un atajo aislado facilmente identificable.

Siguiente paso razonable: revisar manualmente los casos candidatos de Grad-CAM++ por clase, sobre todo `bcc` y `nv`, y etiquetar cualitativamente si el foco externo corresponde a piel sana, borde de lesion, pelo, reflejo, marco oscuro u otro patron.
