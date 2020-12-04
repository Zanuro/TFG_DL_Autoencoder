Notes from Chapter 6 of the book

Redes neuronales recurrentes y convents 1d.
Vectorizar texto -- tensores(segmentacion de texto en palabras, texto en caracteres)
Tokenizacion -- romper texto en tokens (palabras,caracteres)
One-hot encoding/word embedding.

Tambien existe la tecnica de N-gram: decomponer una secuencia de texto en palabras solas o composiciones de como maximo N palabras.

One-hot encoding de palabras y caracteres
Es la tecnica mas comun.Asocia un indice unico con cada palabra.

Word embeddings -- son vectores densos, de dimensiones menores y se aprenden de los datos no como los one-hot vectors que se codifican directamente.
En caso de que haya vocabularios con 20.000 tokens los one-hot encodings ocuparian mucho espacio.

Word embedding con embedding layer
La relacion entre los vectores tiene que reflejar la relacion semantica entre las palabras. Es decir, esperas que la distancia geometrica entre las palabras este proporcional a la similitud semantica entre las palabras.

Por lo tanto los word-embeddings preveen muchos vectores potencialmente utiles.

En keras esto se realiza mediante las capas Embedding.

```python
from keras.layers import Embedding
embedding_layer = Embedding(1000,64) ### 1000 posibles token con vectores de dimension 64
```

La capa mapea indices enteros(palabras) a vectores densos.
Acepta batches de diferente forma pero todas las secuencias del mismo batch necesitan ser del mismo tamano.
Esta capa retorna un tensor 3d para que se pueda procesar por una rnn o un convnet 1d.
Al principio los pesos de los vectores token son random pero luego por la back-propagation los vectores se ajustan.

Pretrained word embeddings

A veces hay poco informacion para poder entrenar y hacer un word-embedding.
Para ello se puede pre-cargar un vector de un espacio embedding que contenga las propiedades que necesites.
Esto se puede hacer principalmente cuando no se disponen suficientes datos como para aprender caractersticas determinantes del dataset.
Estos word embeddings normalmente se realizan mediante estadisticas de ocurrencia de palabras usando varias tecnicas entre cuales las redes neuronales.

Hay varias bases de datos de word-embedding como: Word2vec, GloVe.Esta ultima se basa en la factorizacion de una matriz de ocurrencia de las palabras.

Vamos a hacer un ejemplo utilizando un word-embedding pre-entrenado.

Redes recurrentes

Las rnn a diferencia de las densas o convnets no tienen memoria.