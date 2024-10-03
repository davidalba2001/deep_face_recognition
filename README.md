

Como proyecto final de la asignatura de *Machine Learning*, se nos planteó el reto de desarrollar un sistema de reconocimiento facial. El objetivo principal es identificar a los compañeros de clase a partir de una imagen, es decir, reconocer quién aparece en una fotografía.

Durante el desarrollo del proyecto, exploramos tres enfoques diferentes para abordar este problema, los cuales detallamos a continuación.

### Introducción al reconocimiento facial

Para comenzar, decidimos implementar modelos sencillos y evaluarlos según su rendimiento, basándonos en dos requisitos principales: que el modelo pudiera entrenarse con la menor cantidad de imágenes posible y que fuese capaz de identificar personas que no necesariamente aparecieran en el conjunto de entrenamiento. Este enfoque inicial nos permitió familiarizarnos con el mundo del *face recognition* y evaluar si los resultados obtenidos cumplían nuestros estándares. Si no lo hacían, avanzaríamos hacia soluciones más complejas.

### Primer enfoque: Algoritmo Viola-Jones y LBPH

El problema inicial para identificar a una persona a partir de una imagen de su rostro radica en detectar primero el rostro dentro de la imagen. Para ello, empleamos el algoritmo propuesto por Paul Viola y Michael Jones, ampliamente utilizado para la detección de rostros. Este algoritmo sigue los siguientes pasos:

1. **Conversión a escala de grises**: La imagen se transforma a escala de grises para reducir la cantidad de datos a procesar.
2. **Exploración del fotograma**: Se utiliza una ventana deslizante que recorre toda la imagen.
3. **Detección mediante *haar-features***: Dentro de esta ventana, se identifican posibles rostros utilizando características de Haar.

Una vez detectado el rostro, aplicamos técnicas de *data augmentation* para generar variaciones de la imagen y mejorar la precisión en la fase de predicción. Posteriormente, utilizamos el algoritmo *LBPH* (Local Binary Patterns Histogram) para identificar a la persona en la imagen. LBPH crea una imagen intermedia que resalta las características faciales más relevantes, dividiendo el rostro en una cuadrícula. Para cada celda de la cuadrícula se genera un histograma con 256 valores que representan las intensidades de los píxeles. Estos histogramas se concatenan para formar uno solo, que define las características del rostro.

El flujo de trabajo de este primer enfoque es el siguiente:

1. Se abre la cámara del dispositivo.
2. Cada fotograma se procesa siguiendo estos pasos:
   - El algoritmo Viola-Jones detecta el rostro.
   - Se aplica LBPH para generar el histograma del rostro.
   - El histograma resultante se compara con los almacenados en la base de datos.
   - Se identifica el rostro con el histograma más similar.

Aunque el proceso fue eficiente en términos de velocidad, la precisión del reconocimiento fue baja, rondando entre el 50% y el 60%, incluso tras aplicar *data augmentation* y limpiar los datos. Esto nos llevó a explorar un enfoque más avanzado.

### Segundo enfoque: Redes Convolucionales Siamesas

El siguiente paso fue experimentar con redes convolucionales siamesas, que, a diferencia de las redes convolucionales tradicionales, no clasifican imágenes en categorías. En su lugar, las redes siamesas miden la distancia entre dos imágenes: si ambas pertenecen a la misma persona, la distancia será pequeña; si son de personas diferentes, la distancia será mayor.

Una ventaja clave de las redes siamesas es que no es necesario reentrenar el modelo cada vez que se añaden nuevas imágenes. Este enfoque fue especialmente prometedor, considerando que en nuestro grupo de más de 40 compañeros, necesitaríamos al menos 10 imágenes por persona para un entrenamiento adecuado, lo que sumaría un mínimo de 400 imágenes.

El desafío de incluir nuevos individuos en el sistema fue solucionado por las redes siamesas. A diferencia de los modelos de clasificación tradicionales, en los que cada vez que se añadía una nueva persona era necesario reentrenar el modelo (un proceso costoso y poco práctico), las redes siamesas no requieren aprender nuevas clases. Simplemente, al añadir una nueva imagen, el sistema calcula la distancia entre esta y las imágenes ya almacenadas, identificando si pertenece a la misma persona, sin necesidad de reentrenamiento.

Aunque lo ideal habría sido entrenar el modelo con imágenes de nuestros compañeros de clase, lo que permitiría que el modelo aprendiera las características específicas de sus rostros, nos enfrentamos a la falta de imágenes. Por esta razón, decidimos optar por entrenar el modelo con una gran cantidad de imágenes disponibles en internet. De este modo, el modelo podría aprender a reconocer similitudes entre rostros de manera generalizada y ser capaz de identificar con precisión rostros que no había visto previamente.

Además, decidimos evaluar diferentes funciones de similitud y seleccionar la más adecuada para nuestras necesidades, basándonos en los resultados obtenidos durante las pruebas.

