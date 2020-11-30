AutoEncoder(Notes about autoencoders)

Componentes:

1.Encoder: reduce la dimension del input y comprime los datos del input en una representacion codificada.
2.Bottleneck(capa oculta): capa que contiene la representacion comprimida de los datos de entrada.
3.Decoder: reconstruir los datos a partir de la representacion codificada de forma mas cercana al original.
4.Reconstruction Loss: metodo que mide como de bien se ha hecho la reconstruccion con respecto al original.


Arquitecturas posibles son:
FeedForward Network
Long-short term memory Network
Convolutional Neural Network

Usos:

1.Deteccion de anomalias/outliers.
Datos relacionados basandose en la correlacion entre los datos presentes del input para comprimir los datos.

Si el error de reconstruccion es muy grande significa que la imagen no pertenece/ no estan correlacionados con las otras imagenes utilizadas para entrenar la red.

2.Image Denoising.
Eliminar ruido de una senal(imagenes,audio,documentos).
Las imagenes con ruido tienen cierta dimension simetrica y se va reduciendo a la mitada pasando por diferentes capas.
En la reconstruccion se elimina el ruido

Aun cuanda haya muchos nodos en la capa oculta se puede extraer informacion interesante de los datos de entrada.
El nodo es activo si su valor de salida es cercano al 1 o inactivo si esta cerca del 0.
Queremos que los nodos esten inactivos la mayor parte del tiempo.
Funcion sigmoid de activacion o funcion tanh de activacion.

Con un parametro de escasez queremos que la activacion de cada nodo sea cercana al parametro de escasez y por lo tanto deberia estan cerca del 0.

Anomalias/outlier son puntos de datos que no pertenecen a una poblacion.

Como detectar anomalias:

1.Deviacion estandar.
El 68% de los datos estan a un punto de la deviacion estandar y el 99.7% a 3 puntos de la deviacion estandar.
Es decir si un dato es tres veces mayor que la deviacion estandar entonces es un dato anomalo.

2.Boxplots
Demonstracion grafica utilizando cuantiles. 
Muy utiles para ver los outliers.

Rango inter-cuantil: medir la dispersion estadistica dividiendo el dataset en cuantiles.

3.DBScan Clustering
Cluserizacion(agrupacion) de datos en grupos.
Se utiliza con datos de una o multiples dimensiones.(tambien se puede utiliza k-means o clustering jerarquico).


Core Points:
-min_samples: minimo numero de pountos de nucleo necesario para formar un cluster.
-eps:(maxima distancia entre dos muestras para que se puedan considera del mismo cluster).
Border Points: son puntos tambien del cluster pero mas alejados del centro del cluster.
Noise Points: otros puntos que no pertenecen a ningun cluster.(anomalias o no)
Si los datos son de una dimension alta es menos precisa.

4.Isolation Forest:
Este metodo es diferentes de los otros.
Explicitamente identifica a los datos anomalos en vez de construir los rangos de datos normales y luego identificar el dato extrano.Para ello se basa en que los datos extranos son pocos y tienen atributos diferentes a los otros datos.
Es muy efectivo para detectar anomalias y para datasets de multi-dimensiones.

5.Robust Random Cut Forest
Algoritmo de Amazon para detectar anomalias,se basa en la deviacion estandar para detectar posbiles anomalias.
Tambien es mucho mas rapido que el algoritmo de Isolation Forest.

Building autoencoders: https://blog.keras.io/building-autoencoders-in-keras.html

Los autoencoders son data-specific, es decir, si la red se entrena con caras luego no va a ser posible pasar a deteccion de animales.

Es una tecnica self-supervised, el objetivo se genera a partir de los datos de input.