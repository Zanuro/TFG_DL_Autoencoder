Notes from Chapter 1 of the book

ML-Analytical Engine, Datos+Respuestas espera unas reglas a obtener.Entrenar el modelo en vez de programarlo de forma explicita.
Tiene mucha semejanza con la estadistica aunque a diferencia de la estadistica trabaja con muchos datos.

Para ml necesitamos:
- Puntos de datos de input
- Outputs esperados
- Medir la diferencia entre el output obtenido y el output esperado(el ajuste se aprende y se transmite como feedback)

El problema principal por lo tanto es transformar los datos de una forma significativa,que nos ayuden a obtener un resultado cercano al deseado.(a la representacion deseada).

Si tenemos un conjunto de puntos blancos y negros y dado unas coordinadas queremos saber si es blanco o negro:
-Input: coordenadas del punto
-Output: colores de los puntos
-Porcentaje de acieto del color del punto.

Para ello necesitamos un metodo de separacion de los puntos.Si logramos esto podemos decir que los puntos blancos/negros estan cuando x>0 o x<0.

Si en vez de buscar otro sistema de representacion hariamos una busqueda directa seria ml dando como feedback el porcentaje de acierto.
Los modelos de ml no son tan creativos en buscar otro sistema de representacion sino buscan a traves de un set de operacion predefinidas.

El dl es un paso mas hacia la busqueda mas exhaustiva de la representacion de esos datos mediante multiples capas.

Estas capas componen la red neuronal.Estas capas transformas el dato del input en una representacion cada vez mas diferente a la original y mas informativa sobre el output final.

La especificacion de cada capa sobre lo que produce en el input es dado por el peso de la capa(parametrizado)
El proceso consta por lo tanto en buscar los "pesos" adecuados para las capas para obtener la respuesta correcta.
Pero una red neuronal puede tener millones de parametros.
Por lo tanto primero necesitamos observar que diferencia hay entre el resultado obtenido y el que se deberia obtener.(loss function).El resultado de la loss function se utilizara como feedback para ajustar los valores de los pesos.(obtener un loss score menor).Esto se hace mediante el optimizer utilizando una propagacion hacia atras.
Al principio el loss score es muy alto pero luego va disminuyendo.(training loop)

Algunas veces dl no es la mejor herramienta sobretodo en aplicaciones con pocos datos.

Modelo probabilistico es la aplicacion de los principios de la estadistica en el analisis de datos(algoritmo Naive Bayes)
Clasificador basado en el hecho de que todos los input son independientes uno del otro. Otro modelo relacionado a este es la regresion logistica.

Metodos kernel son un grupo de algoritmos de clasificacion- SVM,buscando los limites de decision entre dos grupos de categorias distintas.(Separando los datos en dos grupos).Sin embargo han resultado dificiles trabajando con datasets grandes.

Random forrest-basados en arboles de decision.

Desde 2012,las redes neuronales convolucionales son el mejor algoritmo para la clasificacion de las imagenes.

DL ha ganado a los otros algoritmos haciendo el proceso de transformacion de los datos mas asequible, no utilizando metodos del tipo feature engineering para obtener capas con mejores representaciones de los datos.DL automatiza este proceso.
En cuanto el modelo actualiza uno de sus parametros, el resto de los parametros se actualizan sin la necesidad del humano.

Las redes neuronales han empezado a utilizarse cuando se han empezado a hacer las siguientes mejorias:
Optimization schemes: RMSProp y Adam.
Funciones de activacion.
Redes neuronales con mas de 10 capas.


