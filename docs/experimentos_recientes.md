# Experimentos recientes y análisis derivados

Este documento resume las pruebas añadidas después de la comparación secuencial base entre `multi-output` y `OVA`. Los CSV y figuras se generan localmente bajo `resultados_actualizados/` y no se versionan en GitHub.

## Eficiencia de OVA

Tras comprobar la equivalencia práctica entre `multi-output` y `OVA` en varios datasets, se añadieron tandas centradas en el principal inconveniente de OVA: el coste de entrenar un clasificador por clase. Estas tandas no proponen un algoritmo nuevo, sino que evalúan si el coste adicional puede mitigarse mediante:

- entrenamiento paralelo por clase;
- reducción arquitectónica;
- cambios simples de `batch_size`, `patience`, `epochs` y `learning_rate`;
- selección de configuraciones que mantengan rendimiento predictivo.

La comparación temporal usa:

- `train_time_seconds` para `multi-output`;
- `train_time_seconds` acumulado para OVA secuencial;
- `parallel_train_time_seconds` como estimación de tiempo OVA paralelo ideal;
- `ova_model_train_time_seconds_mean` como coste medio por clasificador binario.

Scripts principales:

- `run_parallel_ova.py`;
- `scripts/analisis_tiempos_paralelo_ova.py`;
- `scripts/analisis_tiempos_arquitecturas_reducidas.py`;
- `scripts/analisis_arquitecturas_ova.py`;
- `scripts/analisis_configuraciones_seleccionadas_estadistico.py`.

## Arquitecturas reducidas

Para datasets clásicos MLP se comparan:

- referencia `[32, 16]`;
- OVA `[24, 12]`;
- OVA `[16, 8]`.

Para datasets de imagen con VGG compacta se probaron reducciones como:

- referencia `[32, 64, 128]`;
- OVA `[24, 48, 96]`;
- OVA `[16, 32, 64]`.

La finalidad es comprobar si OVA mantiene rendimiento al usar modelos más pequeños, reduciendo parte del coste de entrenar una red por clase.

## Configuraciones destacadas

Las pruebas de eficiencia se interpretan de forma matizada. OVA no elimina su coste computacional, porque el coste secuencial acumulado sigue siendo mayor. Sin embargo, cuando se estima el escenario paralelo por clase, el coste efectivo puede ser comparable o menor que el de `multi-output` en varios datasets, manteniendo equivalencia o mejora predictiva.

Casos destacados de las tandas locales:

| Dataset | Configuración OVA destacada | Lectura |
|---|---|---|
| TB Chest X-ray | VGG reducida `[16, 32, 64]` | OVA mantiene o mejora rendimiento con menor tiempo paralelo estimado. |
| MNIST | VGG `[32, 64, 128]`, `batch_size=128` | Rendimiento prácticamente equivalente y tiempo paralelo competitivo. |
| CIFAR-10 | VGG `[32, 64, 128]`, `batch_size=128` | Caso de equivalencia predictiva con reducción temporal moderada. |
| BRISC | VGG `[32, 64, 128]`, `patience` ajustado | OVA mejora rendimiento; la ventaja espacial no queda demostrada por XAI. |
| Digits | MLP `[32, 16]`, `batch_size=64` | OVA mejora rendimiento con coste paralelo comparable. |

En datasets pequeños como `iris`, `wine` y `breast_cancer`, el beneficio temporal es menos estable porque el entrenamiento `multi-output` ya dura muy poco y el overhead de lanzar varios clasificadores OVA pesa más.

## Explicabilidad en BRISC

El análisis de explicabilidad se añadió para comprobar si la mejora predictiva de OVA en imagen biomédica coincide con una mejor focalización anatómica. El hilo final combina:

- Grad-CAM y Grad-CAM++;
- LRP;
- oclusión de tumor y peritumor;
- análisis de regularidades del dataset;
- cruce con predicciones y errores por imagen.

Conclusión operativa:

> OVA puede mejorar el rendimiento predictivo en BRISC, pero esa mejora no implica automáticamente una mejor localización tumoral. La evidencia espacial es baja o inestable en varias clases, y parte de la decisión puede apoyarse en señales contextuales o de adquisición.

El protocolo completo está en `docs/protocolo_explicabilidad_actual.md`.

## HAM10000

HAM10000 se incorporó como extensión dermatológica para probar una tarea más compleja y desbalanceada. Se añadieron:

- split interno fijo por `lesion_id`;
- evaluación opcional con test oficial de ISIC 2018 Task 3;
- VGG16 preentrenada con ajuste de `block5`;
- sampler balanceado;
- pesos de clase;
- focal loss para OVA;
- calibración auxiliar de OVA;
- exclusión de clases;
- modo binario maligno/no maligno;
- análisis de atajos visuales y artefactos.

La configuración más útil para el análisis exploratorio fue VGG16 `block5` con sampler balanceado, sin tratar HAM10000 como resultado central de la memoria. Los detalles están en:

- `docs/resumen_ham10000_decision_atajos.md`;
- `docs/ham10000_atajos_por_artefacto_y_clase.md`.

## Figuras y tablas para memoria

Se añadieron scripts para generar figuras y tablas directamente desde los CSV locales:

- `scripts/figuras_tfm_configuraciones_seleccionadas.py`;
- `scripts/figuras_tfm_delta_sensibilidad_configuraciones.py`;
- `scripts/figuras_caepia_resultados.py`;
- `scripts/resumen_arquitecturas_referencias_pruebas.py`.

Algunos de estos scripts escriben por defecto en carpetas locales como `Memoria TFM/img/` o `figuras_caepia/`. Esas salidas no se versionan porque son artefactos derivados.
