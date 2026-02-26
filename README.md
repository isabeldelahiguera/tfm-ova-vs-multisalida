De momento solo tenemos

trainer.py
prueba.sh

Respecto a trainer.py

    - Le indicas una base de datos para usarla en entrenamiento, de momento solo soporta redes totalmente lineales. Al ser solo redes lineales hay tiene como parametros de entrada, si hay capas de normalizacion y el tamaño de las capas ocultas.
    - El codigo entrena primero una red para clasificar cada clase del dataset, se usa la como perdida la entropía cruzada. Después entrene una red que sabe clasificar solo una clase y no las demás, para ellos se usa la función de perdida BCEWithLogitsLoss().

Respecto a prueba.sh

    - No tiene parametros de entrada, lo único que hace es lanzar los experimentos indicados en el script.

