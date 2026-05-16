# Resultados con 10 semillas

Esta carpeta contiene la selección curada de resultados finales obtenidos en SLURM con 10 semillas por configuración.

Cada dataset tiene dos ficheros:

- `*_10semillas.csv`: resultados detallados, con una fila por semilla y tipo de modelo.
- `*_10semillas_summary.csv`: medias agregadas por configuración.

Los ficheros proceden de `resultados_slurm/`, pero se han copiado aquí con nombres estables y sin ids internos de jobs de SLURM. La carpeta `resultados_slurm/` queda como zona local de trabajo y no se versiona.
