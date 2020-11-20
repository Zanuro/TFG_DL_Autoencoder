Notes from Chapter 2 of the book

El primer modelo de dl es la clasificacion de numeros en 10 categorias:0-9.Para ello se utiliza el dataset de MNIST.
Los datos de entrenamiento son 60000 y los de testeo son 10000.

Los layers van a ser nuestras capas obteniendo una representacion mejor para ellos.
La segunda capa -softmax va a retornar la probabilidad de que un input pertenezca a una categoria.
Necesitamos tambien una funcion loss para medir el exito de nuestra red, un optimezer para actualizar la red en funcion de la loss function y los datos.
Tras esto antes de entrenar necesitamos reformar nuestros datos.Para entrenar la red se llamada a la funcion fit.
Con esto se representan dos valores:la loss function y la accuracy de la red sobre los datos entrenados.
Luego tambien lo comprobamos con los datos de testing.
Los datos de testing tienen un porcentaje menor de accuracy y por lo tanto occurre overfitting.

Tensor es un contenedor para datos/numeros en diferentes dimensiones.
Pueden ser scalares(0d), un float32 con dimensiones 0.
Pueden ser vectores,array de numeros de dimension 1.
Pueden ser matrices con dos axis de dimension 2.
Luego empaquetando datos de una cierta dimension se obtienen datos de dimensiones superiores.

Un tensor viene definido por:
-rango(numero de axis)- dimension del tensor.
-forma:dimensiones a lo largo de cada axis.
-tipos de datos:float32/uint8...

Normalmente los datos se separan en batches de 128.

Normalmente los vectores de datos son 2d: samples,features.
Los timeseries o secuencias de datos son 3d:samples,timestamps,features.
Imagenes: 4d- samples,height,width,channels.
Videos: 5d-samples,frames,height,width,channels.

Vector data:vector dos coordenadas(sample,features)
Cada persona-edad,sexo,altura.
Por lo tanto tenemos un dataset de 10.000 personas con tres atributos: (10000,3)

Timeseries data:tensor 3d, precio de las acciones:mayor precio del dia,menor precio y el precio actual.
Por lo tanto es un sensor(390,3).Por lo tanto un sensor (250,390,3).Serian 250 dias de 6.30h.

Image data:tambien 3d.
Un batch de 128 imagenes de 256x256 con los canales de altura,anchura y profundidad seria:(128,256,256,3).

Video data:5d,serian una composicion de frames:(samples,frames,height,width,color_depth).
Un video de 60 segundos con 4 frames por segundo de un video de 144x256 seria(4,240,144,256,3).

Tensor:operaciones

capa keras: keras.layers.Dense(512, activation='relu')
y retorna un tensor W 2d relu(dot(W, input) + b) y un vector b.

La primera operacion relu se realiza sobre un tensor 2d procurado obtener el max(x[i,j],0) de cada elemento.

Si se intentan sumar tensores de diferentes dimensiones, el tensor de dimensiona se va a redimensionar al tamano del tensor mas grande.
X(32,10)
Y(10,) -> Y(1,10) -> Y(32,10)
Como llegan a tener la misma forma se pueden sumar.

La operacion dot( . ), es compatible solo para vectores del mismo numero de elementos y se haria el producto entre elementos.
Al hacer el producto dot entre dos matrices: A(a x b), B(b x c) -> donde A[1] == B[0] -> C(a x c)

Tensor reshaping
Reordenar las filas y columnas para tener la forma de nuestro objetivo.
De un array([0.,1.],[2.,3.],[4.,5.]) -> x.reshape(2,3) -> *([0.,1.,2.],[3.,4.,5.])

Cada capa transforma su input del tipo:
output = relu(dot(W, input) + b ), los W y b se llaman peso o parametros de entreno.Al principio los parametros son random pero luego de forma gradual se ajusta(entreno).Todo esto se hace mediante el training loop,viendo el error entre el predecido y el real y actualizando de esta manera los pesos de la red.

Los pesos se actualizan dado que son diferenciables y se calcula el gradiente del loss en funcion de los coeficientes actuales de la red.

y_pred = dot(W,x)
loss_val = loss(y_pred,y)

Por lo tanto para recalcular los pesos necesarios para el entreno en las redes es necesario ver la combinacion de pesos que tienen el menor loss.

1.Input x e resultados a obtener y
2.Obtener predicciones y_pred
3.loss(y_pred, y)
4.backward pass para calcular el gradient del loss
5.modificar los parametros en la direccion contraria W -= step * gradient.

Para ello es importante obtener un buen factor del step.

Al hacer network.compile(optimezer='rmsprop',loss='categorical-crossentropy',metrics=['accuracy']), tenemos como funcion loss a la categorical_crossentropy para aprender los pesos de los tensores y la fase de entreno procurara minimizarlo.Las reglas para disminuir ese loss estan definidas por el optimizer rmsprop.
Por ultimo en la funcion fit se entrena los datos en grupos(batch) con unas iteraciones sobre todos los datos.
Como hay 60.000 datos de entreno habran 469 actualizaciones del gradient para la funcion loss es decir durante todas las iteraciones habran 2345 actualizaciones, aumentando progresivamente el accuracy.