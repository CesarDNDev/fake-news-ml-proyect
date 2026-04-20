UT2. DESARROLLO DE APLICACIONES IA
PROYECTO ML: DE LOS DATOS AL DESPLIEGUE
OBJETIVO
Investiga para desarrollar una aplicación basada en machine learning que resuelva un problema real en uno de estos ámbitos:
* Medio ambiente / sostenibilidad 
* Detección de fake news 
* Detección de comentarios ofensivos o de odio en redes sociales 
El proyecto debe simular un caso real de empresa: desde la obtención de datos hasta una pequeña aplicación funcional.
TAREA A ENTREGAR: 
1. Cuaderno de Jupyter de nombre ProyectoMLXXX.ipynb ejecutado, correctamente comentado y con conclusiones adecuadas.
2. Este documento completando cada uno de los puntos que se piden que desarrolles
3. Exposición del proyecto al resto de la clase
FASES DEL PROYECTO 
1. Definición del problema (enfocado a empresa)
* Explica el caso práctico: 
   * Ej: “Una empresa quiere detectar comentarios ofensivos automáticamente” 
* Define: 
   * Tipo de problema: clasificación, regresión…
   * Qué predice el modelo 
* Ejemplo: 
   * Entrada: texto de un comentario 
   * Salida: ofensivo / no ofensivo 
2. Obtención de datos 
Selecciona un dataset real de los propuestos a continuación.
Debes indicar:
* Qué contiene el dataset 
* Número de registros 
* Qué columnas usarás 
DATASETS RECOMENDADOS (aunque puedes usar otros)
Detección de odio / lenguaje ofensivo
* Hate Speech and Offensive Language Dataset (Davidson) 
https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
   * Contiene tweets clasificados como: 
      * discurso de odio 
      * lenguaje ofensivo 
      * neutro 
* Jigsaw Toxic Comment Classification 
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
   * Comentarios etiquetados como: 
      * tóxico, insulto, amenaza, etc. 
Fake News
* Fake and Real News Dataset 
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
   * Noticias clasificadas como: 
      * fake / real 
   * Incluye título y texto 
* LIAR Dataset 
https://www.cs.ucsb.edu/~william/data/liar_dataset.zip 
   * Frases cortas verificadas 
   * Etiquetas: verdadero, falso, parcialmente falso 
Medio ambiente / sostenibilidad
* Air Quality Dataset (UCI) 
https://archive.ics.uci.edu/ml/datasets/Air+Quality 
   * Datos de contaminación del aire 
   * Permite predicción de niveles de polución 
* Global Temperature Time Series 
https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data
   * Datos históricos de temperatura 
   * Ideal para regresión 
* CO2 Emissions Dataset 
https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles 
   * Emisiones por país/año 
   * Permite análisis y predicción 
3. Preparación de datos
* Limpieza básica: 
   * eliminar nulos 
   * eliminar duplicados 
* Transformaciones: 
   * texto → tokenización (si aplica) 
   * normalización de datos numéricos 
* División: 
   * entrenamiento (80%) 
   * test (20%) 
________________


4. Entrenamiento del modelo
Usa herramientas accesibles:
* Python + scikit-learn 
* Opcional: TensorFlow o PyTorch 
Modelos recomendados:
* Clasificación: 
   * Regresión logística 
   * Random Forest 
* Regresión: 
   * Linear Regression 
5. Evaluación 
* Clasificación: 
   * accuracy 
   * matriz de confusión 
   * heatmap
* Regresión: 
   * error medio (MAE o MSE) 
Explica:
* ¿El modelo funciona bien? 
* ¿Dónde falla? 
6. Mejora básica
* Ajustar parámetros simples 
* Probar otros modelo 
* Equilibrar clases (si está desbalanceado) 
7. Despliegue 
Crea una pequeña app donde se pueda usar el modelo:
Opciones recomendadas:
* Streamlit 
* Flask
* Gradio 
Ejemplo:
* Escribes un comentario → el modelo dice si es ofensivo 
8. Entrega final
Debe incluir:
* Código organizado 
* Dataset o enlace 
* Explicación sencilla del proceso 
* Aplicación funcionando