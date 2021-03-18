Opinion Fraud Detection -- Autoencoder Decision Forest

Modelo que pueda evaluar la cualidad de las opiniones utilizando un autoencoder y el random forest.
Estos experimentos se hicieron sobre un dataset de Amazon.

Con la popularidad de los comercios electronicos, reservas online, y compartiendo las experiencias/opiniones sobre servicios y productos han aumentado tambien las opiniones/reviews fake para hacerle dano o mejorar el "prestigio" de un producto o servicio.
Tambien existen grupos dedicados a hacer este tipo de cosas.
Hasta ahora lo que se hacia era extraer informacion sobre el texto del review/rating o meta-datos del producto.
Tambien se ha visto un pattern en el comportamiento del usuario o distribucion en el comportamiento del usuario.

Rayana and Akoglu -- framework para pistas sobre decepcion,comportamiento extrano.

Algunas tecnicas utilizadas fueron: svm o naive bayes clasifiacion. 
Con algunas caracterisitcas de input se pueden detectar varias caracteristicas e intentar ver si es fake o no.
Tambien existe un autoencoder recursivo semi-supervisado para la deteccion de spam en blogs.

Autoencoder ha sido un algoritmo robusto utilizado como no supervisado para la deteccion de patterns.
Los arboles de decision tienen buen rendimiento.
El modelo creado se basa en la combinacion de estas dos arquitecturas.

Primero se utilizaran unas metricas para detectar comportamiento/patterns extranos que se usaran como indicadores discriminativos.

La mayoria de los metodos de deteccion de estos patterns se basan en caracteristicas particulares del lenguaje,relaciones,aspectos comportamentales,texto de la opinion,rating,votos positivos,etc.
Hay intentos de deteccion de las opiniones "fake" usando un modelo hibrido de red neuronal convolucionales o un modelo basado en un autoencoder semi-supervisado recursivo para la deteccion de spam.Wang 2016.

Para diferenciar los datos non-spam y los spam se han utilizado la entropia de las resenas y la entropia del tiempo de las opiniones. -- Wilcoxon signed-rank test.

Normalmente se ha demonstrado que los spammers tienen un valor mas alto(hay mas) en las resenas mas altas y las mas bajas pero estan en menor medida en las resenas con un rating "mediano".
Esto se da por el hecho de que una vez que un spammer pone un rating es muy probable que mantenga ese rating para siempre y asi se determina que es un spammer.

Otra estadistica es que los spammers utilizan menos palabras de media que los non-spammers y suelen comentar mucho antes que los non-spammers.

Para este modelo se ha utilizado informacion tanto sobre el usuario: numero de productos sobre los que ha opinado,estructura de la opinion,ratings de las opiones,numeros de votos positivos/negativos que ha dado, tiempo de actividad,etc. Pero tambien se han utilizado informacion de las opiniones que ha hecho: entropia de los ratings de todas las opiniones,ratings del usuario sobre los productos, votos positivos y negativos que ha recibido,tamano y semantica de las opiniones,etc.

La metodologia propuesta es inicializar todos los parametros y darselo como entrada al autoencoder,los nodos de las capas conectadas llegan a los nodos del arbol.Los nodos tomaran una decision y haran una prediccion en funcion de los resultados de las decisiones.
Luego se actualizan los parametros minimizando el coste entre la prediccion y el valor real y asi se continuara de forma iterativa.

Encoder recibe un input X y lo transforma en H = f(W*X + b)
donde f es la funcion de activacion y W y b son pesos.
Luego mediante un decoder: Xc = fd(Wd*H + bd), donde Xc es  la prediccion del valor inicial y Wd y bd son pesos.

Algoritmo 1: Proceso de entreno
    -Vector X input con las etiquetas fake/not fake, numero de epochs,numero de arboles y profundidad de los arboles
    -Inicializar parametros
    -1..n_epochs: Partir datos segun n_batch y batch_size
    -1..n_batch: Actualizar parametro mediante RMSProp

Complejidad:
    Consiste en tres partes:
        -parte autoencoder, capas conectadas y parte de random forest
        La parte de autoencoder depende del numero de capas desde el input hasta la capa oculta y los nodos ocultos de cada capa. Para las capas conectadas se tiene en cuenta el numero de capas y el numero de nodos de la capa.
        Para el random forest se ha tenido en cuenta el numero de nodos de salida, numero de arboles y nodos de cada arbol.
        Todo esto se multiplicara por el numero de epochs y batches que se hayan definido.

Tuning de los parametros:
    -numero de capas
    -nodos de los arboles del random forest
    -batch size
