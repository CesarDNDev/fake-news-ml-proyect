# UT2. DESARROLLO DE APLICACIONES IA
## PROYECTO ML: DE LOS DATOS AL DESPLIEGUE

---

## 1. Definición del problema (enfocado a empresa)

**Caso práctico.** Una redacción digital recibe cada día cientos de artículos, teletipos y publicaciones procedentes de redes sociales y agencias. El equipo editorial necesita una herramienta que, como **primer filtro automático**, marque qué noticias tienen alta probabilidad de ser *fake news* para revisarlas con prioridad antes de publicarlas o de citarlas como fuente.

No se busca sustituir al verificador humano, sino **reducir el volumen** que debe revisar manualmente para ahorrar tiempo.

- **Tipo de problema:** clasificación binaria supervisada.
- **Qué predice el modelo:**
  - **Entrada:** texto de una noticia (titular + cuerpo).
  - **Salida:** etiqueta `Fake` (0) o `Real` (1) con su probabilidad asociada.

---

## 2. Obtención de datos

Se utiliza el **Fake and Real News Dataset** (también conocido como ISOT Fake News Dataset), uno de los más usados en tareas de detección de noticias falsas en inglés.

- **Fuente original (Kaggle):** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **Mirror usado en el proyecto (HuggingFace, sin credenciales):** https://huggingface.co/datasets/Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022

### Qué contiene el dataset
- Dos ficheros CSV:
  - `Fake.csv` → noticias clasificadas como **falsas**.
  - `True.csv` → noticias clasificadas como **reales** (publicadas por medios verificados, principalmente Reuters).
- El dataset original incluye las columnas `title`, `text`, `subject`, `date`.

### Número de registros
- Aproximadamente **44.000 noticias** en total.
- Tras eliminar duplicados y textos muy cortos quedan **34.103** registros útiles.
- Las clases están equilibradas: ~17.047 fake / ~17.056 real.

### Columnas usadas
- `text` (obligatoria): contenido completo de la noticia.
- `title` (cuando está disponible): se concatena al texto para enriquecer la entrada.
- `label` (0/1): se genera automáticamente según el fichero de origen (`Fake.csv` = 0, `True.csv` = 1).

---

## 3. Preparación de datos

- **Limpieza básica**
  - Eliminación de registros con valores nulos en `text`.
  - Eliminación de duplicados.
  - Filtrado de textos con menos de 20 caracteres (ruido).
- **Transformaciones sobre el texto**
  - Conversión a minúsculas.
  - Eliminación de URLs (`http...` y `www...`).
  - Eliminación de símbolos y caracteres no alfanuméricos.
  - Normalización de espacios en blanco.
  - **Vectorización TF-IDF** con n-gramas `(1, 2)`, `max_df=0.7` y `min_df=2`.
- **División del dataset**
  - Entrenamiento: **80%** (27.282 registros).
  - Test: **20%** (6.821 registros).
  - División **estratificada** para mantener la proporción de clases.

---

## 4. Entrenamiento del modelo

- **Herramientas:** Python + scikit-learn.
- **Algoritmo:** Regresión Logística (`LogisticRegression`).
- **Pipeline:**
  1. `TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.7, min_df=2)`.
  2. `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`.

Se eligió la regresión logística porque es un modelo **simple, rápido e interpretable**, que funciona muy bien como baseline en clasificación de texto con TF-IDF.

---

## 5. Evaluación

### Métricas principales

| Métrica | Valor |
|---|---|
| Accuracy | **0.9719** |
| Precision (Fake) | 0.98 |
| Recall (Fake) | 0.96 |
| Precision (Real) | 0.96 |
| Recall (Real) | 0.98 |
| F1-score (macro) | 0.97 |

### Matriz de confusión

|                 | Pred. Fake | Pred. Real |
|-----------------|:---:|:---:|
| **Real Fake**   | 3276 | 134 |
| **Real Real**   | 58 | 3353 |

(Se incluye heatmap gráfico en `reports/confusion_matrix.png` y dentro del notebook.)

### ¿El modelo funciona bien?

Sí. Con un **97,19% de accuracy** y precision/recall equilibrados entre ambas clases, el modelo es claramente apto como primer filtro automático. No muestra sesgo hacia ninguna de las clases.

### ¿Dónde falla?

Se ha hecho análisis explícito de errores (ver notebook). De los 6.821 registros de test, el modelo falla en **192 (2,81%)**:

- **134 falsos negativos** (Fake predicha como Real): se corresponden con bulos bien redactados que imitan el estilo de un medio profesional.
- **58 falsos positivos** (Real predicha como Fake): noticias reales cortas, de opinión o con léxico coloquial.

Esto es coherente con las limitaciones de un modelo basado en frecuencias de palabras: se engaña si el texto adopta el registro "correcto".

---

## 6. Mejora básica

Se han aplicado varias mejoras sobre una configuración base:

- **`class_weight="balanced"`** en la regresión logística, para evitar sesgos ante posibles desbalances.
- **N-gramas `(1, 2)`** en el `TfidfVectorizer`, para capturar expresiones de dos palabras informativas (p. ej. *"breaking news"*, *"sources confirm"*).
- **`max_df=0.7` y `min_df=2`** para filtrar palabras demasiado comunes o demasiado raras.
- **Limpieza básica del texto** antes de vectorizar.
- **Eliminación de duplicados y textos muy cortos**, que solo introducen ruido.

Como mejoras futuras se proponen: probar **Random Forest, SVM lineal y Naive Bayes**, y eventualmente modelos de *deep learning* (LSTM, BERT).

---

## 7. Despliegue

Se ha creado una aplicación web con **Streamlit** (`app.py`) donde cualquier usuario puede:

1. Pegar el titular y el cuerpo de una noticia.
2. Pulsar **"Analizar noticia"**.
3. Obtener la predicción (`Fake` / `Real`) junto con la confianza del modelo.

La aplicación carga el modelo serializado (`models/fake_news_pipeline.joblib`) y llama a `predict_proba()` sobre el texto limpio.

### Opción 1 – Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Opción 2 – App desplegada en la nube
Está desplegada gratuitamente en **Streamlit Community Cloud**:

**https://fake-news-ml-proyect-cdn.streamlit.app/**

---

## 8. Entrega final

### Enlaces del proyecto

- **Repositorio GitHub:** https://github.com/CesarDNDev/fake-news-ml-proyect
- **App desplegada (Streamlit Cloud):** https://fake-news-ml-proyect-cdn.streamlit.app/
- **Dataset (Kaggle):** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Qué incluye la entrega

- **Código organizado** → repositorio público en GitHub con estructura clara (notebook, script de entrenamiento, app, requirements, modelo serializado, reports y carpeta `data/` con instrucciones).
- **Dataset / enlace** → el dataset no se incluye por tamaño (~85 MB); se facilitan los enlaces de Kaggle y HuggingFace en `data/README.md`.
- **Explicación del proceso** → este documento + el notebook `ProyectoMLFakeNews.ipynb` ejecutado con todas sus salidas y conclusiones.
- **Aplicación funcionando** → desplegada en Streamlit Cloud (link arriba) y ejecutable en local.

### Estructura del proyecto

```
fake-news-ml-proyect/
├── ProyectoMLFakeNews.ipynb         # Notebook ejecutado
├── PIAP_UT2_ProyectoML_RELLENO.md   # Este documento
├── PRESENTACION.md                   # Guion de exposición en clase
├── README.md                         # Instrucciones técnicas
├── app.py                            # App Streamlit
├── train_model.py                    # Script de entrenamiento
├── requirements.txt                  # Dependencias
├── .python-version                   # Python 3.12 (Streamlit Cloud)
├── models/
│   └── fake_news_pipeline.joblib    # Modelo entrenado (~27 MB)
├── reports/
│   ├── metrics.txt                   # Métricas finales
│   └── confusion_matrix.png          # Matriz de confusión (heatmap)
└── data/
    └── README.md                     # Enlaces al dataset
```
