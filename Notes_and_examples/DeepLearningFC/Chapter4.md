Notes from Chapter 4 of the book

Aprendizeje supervisado -- mapear input a nuestros objetivos: clasificacion, regresion
Otros: 
generacion de secuencias(predecir una palabra/token de una secuencia)
prediccion del arbol de sintaxis de una frase
deteccion de objetos
segmentacion de imagenes

Aprendizaje no supervisado -- visualizacion de datos, compresion o quitar sonido a los datos. Las principales categorias son reduccion de las dimensiones y clustering

Aprendizaje semi-supervisado: generado las clasificacion usando un metodo heuristico. Auto-encoders, prediccion de los frames futuros de un video.

Aprendizaje reforzado -- Google DeepMind para jugar Go/ajedrez. Agente que recibe informacion del entorno y aprende a elegir una accion para maximizar los resultados.

Dividir los datos en un subconjunto de entreno, validacion y testeo se puede hacer mediante la validacion simple hold-out, validacion K-fold, validacion con mezcla K-fold .

Simple hold-out validation:
Dataset de entreno + set de validacion(para evaluar)

Eliminar redundancias en los datos y dividir correctamente sin que se interpongan el dataset de entreno y el de testing.

Procesamiento de datos:
-vectorizacion
Todos los inputs y los objetivos deben ser tensores de punto flotante
-normalizacion
diferentes escalas/caracteristicas transformarlos para que tengan sd de 1 y media de 0. Datos deben ser homogeneos.(mismo rango).
-manejar los valores tipo missing: Normalmente poner como 0 sin que 0 sea un valor significativo
-extraccion de caractersiticas:
Aplicar transformaciones a los datos antes de meterselo al modelo.Si entendemos el problema a alto nivel podemos elegir/hacer el modelo adecuado/ script adecuado.
Expresar un problema en terminos mas simples.

Las redes neuronales son capaces de extraer informacion por si solas.Resolver problemas con pocos recursos.

Overfitting/Underfitting

Overfitting suele pasar en cualquier problema de machine-learning.
Optimizacion -- ajustar el modelo para aprender 
Generalizacion -- como de bien actua el modelo frente a datos desconocidos
Al principio optimizacion/generalizacion estan correlocionados y el modelo esta underfit, es decir el modelo puede mejorar.Despues de unas iteraciones la generalizacion para de mejorarse y comienza el overfitting.

Para prevenir que un modelo aprenda datos irrelevantes la unica solucion es obtener mas datos. Si no es posible, anadir mas restricciones a los datos y hacer que una red pueda aprende solo determinados patrones.
El proceso de "combatir" el overfitting se llama regularizacion.

Para evitar overfitting: reducir la capacidad del modelo.
Sin embargo,si no tiene capacidad de memoerizacion se puede dar que el modelo no aprenda tan facilmente.
No hay formula para evaluar el numero correcto de capas/nodos para cada problema sino que se debe evaluar cada arquitectura.

Modelos mas simples son menos propensos al overfitting que los complejos.Es decir modelos cuyos parametros tiene menor entropia/menos parametros.
Tambien se puede regularizar el peso de los parametros dentro de cierto rango de valores.
L1 regularization - Coste anadido es proporcional al valor absoluto de los coeficientes de peso
L2 regularization - Coste anadido es proporcional al doble del valor de los coeficientes de peso.

```python
from keras import regularizers

model = models.Sequential()

model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

El regularizer.l2(0.001) significa que cada coeficiente de la matriz de pesos de las capas se le anadiran 0.001*valor_del_coeficiente a la perdida total de la red.

Como alternativa al L2 esta la opcion de regularizacion de keras:

```python
from keras import regularizers
 
regularizers.l1(0.001) ## regularizacion tipo l1
regularizers.l1_12(l1=0.001, l2=0.001) ## regularizacion simultanea l1 y l2
```

Dropout -- tecnica de regularizacion
Poner a 0 un numero de caracteristicas de output durante el entreno.A partir de un vector: [0.2, 0.5, 1.3, 0.8, 1.1] se transformara en [0, 0.5, 1.3, 0, 1.1].
Los valores de la capa se reduce por un factor igual a la velocidad de dropout.
```python
layer_output *= np.random.randint(0, high=2, size=layer_output.shape) ## Reduce a 0 la mitad de los inputs
```
Por lo tanto en la practica se implemntaria de la siguiente forma:

```python
layer_out *= np.random.randint(0, high=2, size=layer_out.shape)  ## en en entreno
layer_out /= 0.5 ## reduciendo a la mitad
```

Este metodo se basa en el hecho de que removiendo neuronas random prevee el hecho de que estas puedan "relacionarse"(memorizar patterns) y por lo tanto esto reduciria el overfitting.

En keras se anadiria mediante un layer especial:
```python
model = model.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
```

Workflow general de un problema de machine/deep learning:

-Definir dataset/input/que se intenta predecir
-Tipo de problema: binario/multiclase/regresion,etc.

Funcion de perdida/ metricas en las que basarse para valorar el modelo.
Para problemas de clasificacion balanceadas los mas comunes son el: accuracy y e el ROC AUC.
Para problemas de clasficacion no balanceadas se utiliza: precision y recall.
Para multilabel: mean average precision.

Como medir el progreso del modelo:
-Hold-out validation set(cuando tienes datos)
-K-fold cross-validation - si tienes menos datos y el metodo anterior no es optimo.
-Iterated K-fold validation: cuando tienes pocos datos.

Preparar los datos: 
-Tensores normalmente entre (0,1) o (-1,1)
-Normalizar los datos

Desarollar modelo:
-Ultima capa de activacion:
-Funcion de perdida
-Configuracion del optimizer: rata de aprendizaje

Elegir la mejor capa de activaion y funcion de perdida:

-Clasificacion binaria: sigmoid / binary_crossentropy
-Multiclase/una etiqueta: softmax / categorical_crossentropy
-Multiclas/multi-etiqueta: sigmoid / binary_crossentropy
-Regression valores arbitrarios: None / mse
-Regresion valores (0,1) - sigmoid / mse o binary_crossentropy


Para saber cuanto de grande tiene que ser el modelo necesitas primero un modelo que haga overfitting. Para ello, se anaden mas capas,capas mas grandes y mas iteraciones.Luego se monitoriza el training loss y el validation loss.
Luego empiezas a modificar los hiperparametros y regularizlo.
Para regularizar: anadir droput, diferentes arquitecturas, mas o menos capas, regularizacion l1,l2, hiperparametros,etc.

