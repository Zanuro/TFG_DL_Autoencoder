
Anomalias:
    -Point: puntos muy diferentes al resto de puntos.
    -Anomalias en grupo/colectivas: Anomalas si estan juntas
    -Contextuales: cierto contexto, tiempo/espacio.
    -Puntos de cambio: solo para los timeseries, puntos donde el pattern cambia o evoluciona

Cabe destacar que en algunos campos, es suficiente con poner ciertos niveles de tolerancia y asi cualquier valor fuera de estos niveles se podrian considerar como anomalias.
Sin embargo, en muchos casos el proceso de etiquetar una anomalia es una tarea que conlleva tiempo y recursos humanos y conocimiento por parte de los humanos entender el proceso de exclusion de dichos datos y categorizarlos como anomalias.
 
Normalmente dado que los datasets no estan etiquetados conviene que la deteccion de anomalias sea no supervisado.

Una RNN es un tipo especial de red neuronal mas eficiente en caso del procesamiento de datos secuenciales.

LSTM puede aprender dependencias arbitrarias sobre un intervalo de tiempo grande. Con esta arquitectura se sustituye una neurona con una arquitectura compleja llamada bloque/unidad LSTM.
Componentes:
    -Constant error carousel: Unidad central con una conexion recurrente con una unidad de peso.Tiene el estado interno que actua como la memoria para la informacion pasada
    -Input gate: Unidad multiplicativa que protege la informacion de la unidad central de ruido de otros inputs
    -Output gate: Unidad multiplicativa que protege otras unidades de los ruidos de la informacion de la unidad central

RNN -- recibe el input y produce un output cada paso ya que solo tiene una sola capa non lineal entre el input y el output.
DNN -- recibe el input en la primer capa, procesa la informacion a traves de multiples capas antes de dar un output.
RNN no ofrece la posibilidad de un procesamiento de informacion de forma jerarquica como hace el DNN.
