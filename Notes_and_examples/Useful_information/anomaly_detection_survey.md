Deep learning para deteccion de anomalias -- notes

1.Metodos para la deteccion de anomalias y su efectividad en diferentes campos.
Anomalias se pueden dar por errores o resultan de un proceso/pattern nuevo

Tecnicas de deep anomaly detection
Aprenden caractersticas de los jerarquicas de los datos sin la necesidad de emplear unas caractersiticas manuales.
Para ello existen metodos supervisados, no supervisados, hibridos, redes neuronales de una clase, semi-supervisados,factorizacion de matrices,variacional,autoencoder,aprendizaje reenforzado.



Datos secuenciales -- Video,TimeSeries,Texto : Arquitectura para la deteccion de anomalias: CNN,RNN,LSTM

Datos no-secuenciales -- Imagenes,Otros: CNN,AutoEncoders

Aspectos de la deteccion de anomalias basad en redes neuronales

-Datos de input: Secuenciales/no-secuenciales.Dependiendo del numero de caracteristicas se clasificarian en una categoria u otra.
-Etiquetas suelen indicar si un dato es normal o es un outlier.

Deteccion de anomalias supervisado: Entrenar un clasificador binario/multi-clase usando etiquetas tanto de datos normales como anomalias.Puede ser un problema el hecho de que las clases no esten balanceadas.

Deteccion de anomalias semi-supervisado: Son mas comunes que las anteriores.Estas tecnicas aprovechan las etiquetas positivas/comunes para separar los outliers.
Un metodo comun en los deep autoencoders en la deteccion de anomalias es entrenar el modelo usando los datos sin anomalias.Con suficientes datos tendria un error de reconstruccion bajo.

Deteccion de anomalias no supervisado: Detectan outliers basandose en propiedades intrinsecas de los datos. Se utilizan mas bien para etiquetar los datos cuando estos no vienen ya etiquetados o son dificiles de etiquetar.
Asumen que hay muchos mas datos normales que los datos anomalos.

Tambien existen otras dos categorias de redes de deteccion de anomalias basad en los objetivos del entrenamiento.

Deep Hybrid Models-- usados principalmente para la extraccion de caractersticas.
One-Class Neural Networks: inspirados en clasificadores one-class basados en kernel que pueden sacar una representacion de los datos con el objetivo de crear una one-class de los datos normales.Sacar informacion comun de una clase.

Tipo de anomalias:
-Point Anomalies: Anomalias aleatorias sin ninguna explicacion particular
-Anomalias basados en contexto: Datos que se pueden considerar como anomalias en determinados contextos,normalmente dependiendo del tiempo y el espacio.
-Anomalias colectivas o de grupo: Grupos de datos anomalos aunque de forma individual parece un dato normal pero en grupo tiene caracteristicas particulares.

Los outputs de las tecnicas de deteccion de anomalias:
-Anomaly score:Dependen de un threshold,normalmente los datos con una distancia mayor al centro de los datos son mas propensos a ser anomalos.
-Labels:Asignar una etiqueta si se trata de un dato anomalo o no.

Aplicaciones en la deteccion de anomalias:
-Deteccion de anomalias en redes sociales: Comportamiento diferente/anomalo de los individuos.Detectar esos patterns es fundamental ya que si no se detectan tendrian un grave efecto.
Para este tipo de problema se puede utilizar la tecnica del AutoEncoder: Zhang 2017, Castellini 2017

Modelos para la deteccion de anomalias

Supervisado: Son mejores en cuanto al rendimiento ya que estos utilizan ya datos etiquetados.
Los no supervisados intentan explicar las caractersticas de los datos.
Suelen tener dos sub-redes, una red de extraccion de caracteristicas y otra red de clasificacion.Sin embargo se necesitan una cantidad relativamente grande de datos: miles o millones.
La complejidad depende de la dimension de los datos de entrada y el numero de capas utilizadas en la red neuronal.

Ventajas:
    -Son mejores/mas precisos que los semi-supervisados o no supervisados
    -La fase de testing es rapida
Desventajas:
    -Tecnicas multi-clase necesita unas etiquetas para las diferentes clases e instancias anomalas.
    -Si los datos tienen un numero grande de dimensiones no lineales es dificil separar los datos normales de las anomalias.

Semi-supervisado:
Se asume que todos los datos de entreno tienen etiquetas de una clase.Aprenden un limite "discriminativo" a partir de los datos normales.Los datos de testing que no pertenecen a la clase de mayoria se etiquetan como anomalias.
Se suelen basar en proximidad y continuidad, es decir, datos de input que son cercanos en el espacio y en las caracteristicas son mas probables que tengan la misma etiqueta.
Las caracteristicas distintivas se aprenden de las capas de la red neuronal y se separan los atributos de los datos normales de las anomalias.
Complejidad del modelo es similar al supervisado dependiendo de la dimension y el numero de caracteristicas.
Ventajas:
    Las redes Generative Adversarial tienen un buen rendimiento
    Datos etiquetados tienen un rendimiento mucho mejor que los no supervisados.
Desventajas:
    Son muy susceptibles al overfitting.

No supervisados:
Los autoencoders son las arquitecturas no supervisadas mas usadas en la deteccion de anomalias.
Se asume que las regiones de los datos normales se pueden distinguir de las regiones anomalas de los datos,la mayoria de los datos son normales y producen una deteccion de las anomalias basandose en propiedades intrinsecas de los datos como distancia o densidad.
Complejidad computacional: Los autoencoders tienen un coste cuadratico, el problema de optimizacion siendo no-convexo.Depende del numero de operaciones,parametros/hiper-parametros de la red, capas ocultas.Tiene un coste mas grande que PCA ya que PCA se basa en descomposicion de matrices.

Ventajas:
    Aprende las caracateristicas intrinsecas de los datos y los separa de las anomalias.Caractersticas comunes de los datos
    Es una tecnica cost-effective ya que para encontrar anomalias no requiere datos de entreno ya etiquetados.
Desventajas:
    Con unos datos con muchas caracteristicas y dimensiones es mucho menos efectivo
    Para los hiperparametros del autoencoder a veces el ajuste de estos parametros es muy necesario para la obtencion de resultados
    Son menos efectivos que los no-supervisados o supervisados.


Arquitecturas redes neuronales para encontrar anomalias

Autoencoders: Con una sola capa con una funcion de activacion lineal son casi equivalentes a los PCA. Sin embargo los autoencoders permiten transformaciones tanto lineales como no lineales.
Uno de las aplicaciones mas comunes de los autoencoders son la deteccion de anomalias aunque tambien se usan tambien las Replicator Neural Network para la deteccion de estos.
Representan los datos con multiples capas reconstruyendo los datos de entrada mediante una funcion de identidad.
Normalmente, al haber mas datos normales que anomalias si se entrenan solamente con datos normales, el modelo fallara en reconstruir los datos anomalos produciendo un alto error de reconstruccion.
Los datos que suelen producir un alto error residual son normalmente los outliers.
Hay multiples arquitecturas de autoencoder dependiendo principalmente de los tipos de datos.
Para datos secuenciales se utilizarian los LSTM.
Sin embargo, aunque los autoencoders son simples y efectivos para la deteccion de anomalias, el rendimiento se ve afectado en parte por los datos de entrenamiento ruidosos. Zhou 2017.

Para datos secuenciales:

    LSTM-AE
    GRU-AE Gated Recurrent Unit - Autoencoder
    AE
    SDAE -- Stack Denoising Autoencoder

