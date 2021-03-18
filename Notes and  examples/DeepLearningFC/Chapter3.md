Notes from Chapter 3 of the book

datos de tipo vectores 2d -- (samples,features) se ven procesados por capas densas Dense.
datos de secuencia 3d -- (samples,timesteps,features) se procesa por capas recurrentes LSTM.
image data -- en tensores 4d con capas convolucionales 2d.

Las capas se pueden conectar por el tamano del output del tensor:

-- layer = layers.Dense(32, input_shape=(784,))
Esta capa retorna un tensor transformado de 784 a uno de 32.

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784,)))
model.add(layers.Dense(32))
Se interfiere de que el input de la segunda capa es el output de la primera.

Se utiliza binary crossentropy para problemas de clasificacion entre dos clases y crossentropy para un problema de clasificacion para multiples clases, mse para problemas de regression.

Keras soporta multiples arquitecturas de red, pudiendo construir cualquier modelo de deep learning.
Keras permite trabajar en backend con TensorFlow,CNTK y Theano.
A mas bajo nivel utilizaran librerias de tipo Cuda,Blas..

Para construir modelos se puede utilizar la clase Sequential para modelos lineales o una API funcional para construir arquitecturas arbitrarias.

Con la API funcional se manipulan los tensores de los datos directamente y aplicando capas a esos tensores como si fueran funciones.
