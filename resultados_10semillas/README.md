# Resultados Curados

Esta carpeta contiene la selección curada de resultados obtenidos en SLURM. La base experimental usa 10 semillas por configuración, pero se conservan también ampliaciones puntuales cuando están disponibles.

Cada dataset tiene dos ficheros:

- `*_10semillas.csv`: resultados detallados, con una fila por semilla y tipo de modelo.
- `*_10semillas_summary.csv`: medias agregadas por configuración.

Además, algunos datasets incluyen ficheros con más semillas, como `iris_mlp_22semillas.csv` o `brisc_vgg_128_12semillas.csv`. Estos ficheros permiten contrastar la estabilidad de los tests estadísticos cuando se amplía el número de repeticiones.

Los ficheros proceden de `resultados_slurm/`, pero se han copiado aquí con nombres estables y sin identificadores internos de jobs de SLURM. La carpeta `resultados_slurm/` queda como zona local de trabajo y no se versiona.
