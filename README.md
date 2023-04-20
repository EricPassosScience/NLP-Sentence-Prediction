# Estudio de caso: inteligencia artificial para predecir oraciones en mociones de aclaración
En este caso de estudio, vamos a trabajar con otro modelo de PNL utilizando PyTorch como nuestro framework. Recibiremos un Embargo de Declaración (texto) como entrada, prepararemos los datos, construiremos y entrenaremos un modelo de red neuronal artificial profunda (DeepLearning) y haremos predicciones de oraciones con el modelo entrenado.

## ¿Qué son los Embargos de Aclaración?
Los Emabrgos de Aclaración es un instrumento jurídico (brasileño) mediante el cual una de las partes puede solicitar al juez o tribunal aclaraciones sobre una decisión judicial. A través de ellos es posible resolver dudas provocadas por contradicciones u oscuridades. Asimismo, se pueden corregir omisiones o señalar errores materiales.

Encontramos más detalles aquí -> https://blog.sajadv.com.br/embargos-de-declaracao-novo-cpc/

¿Y cómo vamos a crear nuestra solución de IA? Utilizando un modelo CBoW. Comprendamos algunos conceptos y definamos qué es un modelo CBoW (es una red neuronal artificial).

## Word Embedding
Word Embedding es una de las representaciones más populares del vocabulario de un documento. Es capaz de captar el contexto de una palabra en un documento, similitud semántica y sintáctica, relación con otras palabras, etc.

En términos generales, las incrustaciones de palabras son representaciones vectoriales de una palabra específica. Word Embeddings es solo una forma elegante de decir "representación numérica de palabras". Una buena analogía sería cómo usamos la representación RBG para los colores.

## ¿Por qué necesitamos Word Embeddings?
Como ser humano, intuitivamente hablando, ¿no tiene mucho sentido querer representar palabras o cualquier otro objeto en el universo usando números porque los números se usan para cuantificar y por qué sería necesario cuantificar palabras?

Cuando en ciencia decimos que la velocidad de mi auto es de 50 km/h, nos damos cuenta de cuán rápido/lento estamos conduciendo. Si decimos que mi amigo conduce a 70 km/h, podemos comparar quién de nosotros va más rápido. Además, podemos calcular dónde estaremos en un momento dado, cuándo llegaremos a nuestro destino ya que sabemos la distancia de nuestro viaje, etc.

Asimismo, fuera de la ciencia, usamos números para cuantificar una cualidad; cuando cotizamos el precio de un objeto, tratamos de cuantificar su valor, el tamaño de una prenda de vestir, tratamos de cuantificar las proporciones corporales que se ajustan mejor.

Todas estas representaciones tienen sentido porque al usar números hacemos que sea mucho más fácil analizar y comparar en función de estas cualidades. ¿Cuánto vale un zapato o un bolso? Bueno, por diferentes que sean estos dos objetos, una forma de responder es comparar precios. Aparte de los aspectos de cuantificación, no hay nada más que ganar con esta representación.

Ahora que sabemos que la representación numérica de los objetos ayuda en el análisis al cuantificar una determinada cualidad, la pregunta es ¿qué cualidad de las palabras podemos cuantificar?

La respuesta a esto es que queremos cuantificar la semántica. Queremos representar las palabras de tal manera que capten su significado como lo hacen los humanos. No el significado exacto de la palabra, sino contextual.

## Continuous Bag of Words Model (CBOW) e Skip-gram
Ambas son arquitecturas para el aprendizaje de representaciones de palabras mediante redes neuronales, actualmente una de las principales técnicas de IA:

![imagem_2023-04-20_111337546](https://user-images.githubusercontent.com/97414922/233393465-6eb05cea-e187-40e9-87c8-456d6b501b39.png)

En el modelo CBOW, las representaciones distribuidas de contexto (o palabras circundantes) se combinan para predecir la palabra en el medio. Mientras que en el modelo Skip-gram, la representación distribuida de la palabra de entrada se usa para predecir el contexto.

Un requisito previo para cualquier red neuronal o cualquier técnica de entrenamiento supervisado es tener datos de entrenamiento etiquetados. ¿Cómo se entrena una red neuronal para predecir la incrustación de palabras cuando no tiene datos etiquetados como palabras y la incrustación de palabras correspondiente? Aquí es donde entra en juego el modelo Skip-gram.

Haremos esto creando una tarea "falsa" para que la red neuronal se entrene. No nos interesarán las entradas y salidas de esta red, pero el objetivo es solo aprender los pesos de las capas ocultas, que son las "matrices de palabras" que estamos tratando de aprender.

A la tarea "falsa" del modelo Skip-gram se le daría una palabra, intentaremos predecir las palabras vecinas. Definiremos una palabra vecina por tamaño de ventana: un hiperparámetro. Esta imagen a continuación demuestra cómo podríamos preprocesar los datos de texto para entrenar el modelo:

![imagem_2023-04-20_114648611](https://user-images.githubusercontent.com/97414922/233402800-ef0ab1ee-18f1-4b86-aab7-799bfc0dee49.png)

Las dimensiones de la matriz de entrada serán 1xV, donde V es el número de palabras en el vocabulario, es decir, una representación One-Hot Encoding de la palabra. La única capa oculta tendrá la dimensión VxE, donde E es el tamaño de la palabra incrustada y es un hiperparámetro. La salida de la capa oculta sería de dimensión 1xE, que introduciremos en una capa softmax. Las dimensiones de la capa de salida serán 1xV, donde cada valor de la matriz será la puntuación de probabilidad de la palabra objetivo en esa posición. El aprendizaje se realiza con el algoritmo Backpropagation.

Finalmente, destaco que todo el contenido aquí presentado es introductorio. Desafortunadamente, no pongo contenido avanzado aquí en mi repositorio de GitHub.

¡Abrazo!