# Resultados de clasificación

Este documento resume los resultados obtenidos hasta ahora en los experimentos de clasificación. La comparación principal se realiza entre el modelo `multi-output` y la descomposición `OVA`.

La tabla se ha construido a partir de los ficheros `exp_*_summary.csv`. Por tanto, los valores presentados son medias por configuración sobre las semillas ejecutadas, no resultados de una única ejecución aislada.

La interpretación se centra en dos cuestiones:

- si `OVA` y `multi-output` muestran un comportamiento funcional o predictivo parecido;
- si la descomposición `OVA` puede igualar o incluso mejorar el resultado de la red `multi-output`.

El tiempo de entrenamiento se interpreta como una medida complementaria de coste computacional. No forma parte de la hipótesis principal del TFM, pero permite distinguir entre equivalencia funcional y equivalencia computacional.

## Tabla resumen

| Dataset | Clases | Acc. multi | Acc. OVA | Delta acc. | F1 multi | F1 OVA | Delta F1 | FPR multi | FPR OVA | FNR multi | FNR OVA | Tiempo multi (s) | Tiempo OVA (s) | Ratio tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iris | 3 | 0.8778 | 0.9000 | 0.0222 | 0.8768 | 0.8996 | 0.0228 | 0.0611 | 0.0500 | 0.1222 | 0.1000 | 0.4 | 1.1 | x2.67 |
| wine | 3 | 1.0000 | 0.9907 | -0.0093 | 1.0000 | 0.9906 | -0.0094 | 0.0000 | 0.0043 | 0.0000 | 0.0079 | 0.5 | 1.2 | x2.50 |
| breast_cancer | 2 | 0.9795 | 0.9825 | 0.0029 | 0.9778 | 0.9811 | 0.0033 | 0.0261 | 0.0205 | 0.0261 | 0.0205 | 1.1 | 2.4 | x2.19 |
| dermatology | 6 | 0.9685 | 0.9640 | -0.0045 | 0.9645 | 0.9594 | -0.0051 | 0.0062 | 0.0071 | 0.0352 | 0.0407 | 1.3 | 7.2 | x5.74 |
| heart_disease | 5 | 0.5902 | 0.5738 | -0.0164 | 0.3314 | 0.3130 | -0.0184 | 0.1180 | 0.1225 | 0.6508 | 0.6753 | 0.8 | 4.7 | x5.71 |
| digits | 10 | 0.9694 | 0.9843 | 0.0148 | 0.9692 | 0.9842 | 0.0150 | 0.0034 | 0.0017 | 0.0308 | 0.0158 | 3.4 | 34.2 | x10.06 |
| mnist | 10 | 0.9607 | 0.9782 | 0.0174 | 0.9604 | 0.9780 | 0.0176 | 0.0044 | 0.0024 | 0.0397 | 0.0220 | 183.9 | 1813.2 | x9.86 |

## Lectura por dataset

### iris

`iris` es un dataset pequeño de 3 clases. En este caso, `OVA` mejora ligeramente la `accuracy` y el `f1_macro` frente a `multi-output`. La diferencia es reducida, por lo que el resultado puede interpretarse como una equivalencia práctica aproximada, con una ligera ventaja para `OVA`.

### wine

`wine` es un dataset de 3 clases en el que `multi-output` alcanza resultados perfectos en la media de las semillas. `OVA` queda muy cerca, aunque ligeramente por debajo. Este caso es importante porque muestra que `OVA` no domina siempre: la equivalencia funcional es muy alta, pero el mejor resultado lo obtiene `multi-output`.

### breast_cancer

`breast_cancer` es un problema de clasificación binaria. `OVA` obtiene una mejora pequeña en `accuracy`, `f1_macro`, `FPR` y `FNR`. Al haber solo dos clases, la comparación es especialmente directa: se compara una formulación multi-salida con una descomposición binaria muy cercana funcionalmente.

### dermatology

`dermatology` es un dataset de clasificación dermatológica con 6 clases. En este caso, `multi-output` obtiene resultados ligeramente mejores que `OVA` en `accuracy`, `f1_macro`, `FPR` y `FNR`. La diferencia es pequeña, por lo que sigue observándose una equivalencia predictiva aproximada. Sin embargo, este resultado refuerza que `OVA` no es sistemáticamente superior. Además, `OVA` requiere alrededor de 5.7 veces más tiempo de entrenamiento, algo coherente con el hecho de entrenar una red binaria por clase.

### heart_disease

`heart_disease` es un dataset clínico de UCI con 5 clases en la variable objetivo `num`. En este caso, `multi-output` obtiene mejores resultados que `OVA` en `accuracy`, `balanced_accuracy`, `f1_macro`, `FPR` y `FNR`. Además, las métricas absolutas son bastante más bajas que en otros datasets, lo que sugiere que este problema es más difícil para la configuración base usada. Aun así, la comparación sigue siendo informativa: `OVA` no mejora el rendimiento y requiere alrededor de 5.7 veces más tiempo de entrenamiento.

### digits

`digits` es un dataset de 10 clases basado en imágenes pequeñas representadas como variables tabulares. En este caso, `OVA` mejora de forma clara las métricas predictivas y reduce tanto los falsos positivos como los falsos negativos. La contrapartida es que el coste temporal aumenta mucho, ya que se entrena una red por clase.

### mnist

`mnist` también es un problema de 10 clases, pero de mayor escala. Es el caso más claro a favor de `OVA`: mejora a `multi-output` en `accuracy`, `f1_macro`, `TPR`, `FPR` y `FNR`. Esto sugiere que descomponer el problema en clasificadores binarios puede mejorar la predicción en algunos problemas multiclase, aunque el coste computacional sea casi diez veces mayor.

## Conclusión global provisional

Los resultados apoyan una lectura matizada de la hipótesis del TFM. En todos los datasets probados, `OVA` y `multi-output` producen resultados funcionalmente cercanos. Por tanto, la descomposición en problemas por clase no parece romper la capacidad predictiva del modelo.

Además, en varios datasets (`iris`, `breast_cancer`, `digits` y especialmente `mnist`), `OVA` no solo iguala a `multi-output`, sino que mejora sus métricas predictivas. Esta mejora también aparece en las tasas de error por clase, con menor `FPR` y menor `FNR`. Esto sugiere que especializar una red binaria para cada clase puede facilitar fronteras de decisión más favorables en algunos problemas.

Los casos `wine`, `dermatology` y `heart_disease` evitan una conclusión excesivamente fuerte. En estos datasets, `multi-output` supera a `OVA`, aunque con magnitudes distintas. Por tanto, no se puede afirmar que `OVA` sea siempre superior. Lo que sí muestran los resultados actuales es que `OVA` puede ser funcionalmente equivalente y, en determinados problemas, mejorar el rendimiento.

La contrapartida es clara: `OVA` tiene mayor coste computacional. En datasets con varias clases, como `dermatology`, `heart_disease`, `digits` y `mnist`, el tiempo de entrenamiento aumenta mucho porque se entrena una red por clase. Esto confirma que la equivalencia funcional, o incluso una mejora predictiva, no implica equivalencia computacional.

## Pendientes

- Revisar si las conclusiones se mantienen con otras arquitecturas o hiperparámetros.
- Incorporar medidas de variabilidad entre semillas, como desviación típica o intervalos de confianza.
- Decidir si `mnist` debe incluirse en la misma notebook que los datasets pequeños o en una sección separada, debido a su mayor coste y escala.
- Repetir algunos experimentos con distinto número de épocas para comprobar la sensibilidad al entrenamiento.
