# Proyecto ML - Detección de Fake News

Este proyecto implementa una solución sencilla de **clasificación de noticias falsas** usando Python, scikit-learn y Streamlit.

## Qué incluye

- `train_model.py`: carga y limpia el dataset, entrena el modelo y guarda resultados.
- `app.py`: pequeña aplicación web para probar noticias manualmente.
- `requirements.txt`: dependencias necesarias.
- `models/`: carpeta donde se guardará el modelo entrenado.
- `reports/`: métricas y matriz de confusión.
- `data/`: aquí debes colocar el dataset descargado.

## Dataset usado

Se espera el dataset **Fake and Real News Dataset** con estos archivos dentro de `data/`:

- `Fake.csv`
- `True.csv`

## Preparación del entorno

### 1) Crear entorno virtual

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Instalar dependencias

```bash
pip install -r requirements.txt
```

## Cómo entrenar el modelo

1. Descarga el dataset.
2. Copia `Fake.csv` y `True.csv` dentro de la carpeta `data/`.
3. Ejecuta:

```bash
python train_model.py
```

Si todo va bien, se generarán:

- `models/fake_news_pipeline.joblib`
- `reports/metrics.txt`
- `reports/confusion_matrix.png`

## Cómo ejecutar la app

Después de entrenar el modelo, lanza:

```bash
streamlit run app.py
```

Se abrirá una app local en el navegador para introducir una noticia y ver la predicción.

## Lógica del modelo

- **Problema**: clasificación binaria.
- **Entrada**: título + cuerpo de la noticia.
- **Salida**: `Fake` o `Real`.
- **Vectorización**: TF-IDF.
- **Modelo**: Regresión logística.

## Mejora básica aplicada

- Uso de `class_weight="balanced"`.
- Uso de n-gramas `(1,2)` en TF-IDF.
- Limpieza básica del texto.

## Limitaciones

- No verifica fuentes reales en internet.
- Puede fallar con textos muy cortos o ambiguos.
- Sirve como prototipo educativo, no como verificador periodístico profesional.


## Notebook Jupyter

También se incluye el archivo `ProyectoMLFakeNews.ipynb` para entregar el proyecto en formato Jupyter Notebook, ya ejecutado con todas las salidas.

## Despliegue en Streamlit Cloud

La app se puede desplegar gratis en <https://share.streamlit.io>:

1. Sube este proyecto a un repositorio público de GitHub.
2. Entra en <https://share.streamlit.io> con tu cuenta de GitHub.
3. Pulsa **New app**, selecciona el repo, rama `main` y archivo principal `app.py`.
4. Streamlit Cloud instalará `requirements.txt` y servirá la app en una URL pública.

El modelo `models/fake_news_pipeline.joblib` ya está en el repo, por lo que **no es necesario reentrenar** en la nube.

## Estructura de la entrega

```
fake_news_ml_project/
├── ProyectoMLFakeNews.ipynb         # Notebook ejecutado (requisito de entrega)
├── PIAP_UT2_ProyectoML.md           # Documento de entrega
├── README.md                         # Este archivo
├── app.py                            # App Streamlit
├── train_model.py                    # Script de entrenamiento
├── requirements.txt                  # Dependencias
├── runtime.txt                       # Versión de Python (para Streamlit Cloud)
├── models/
│   └── fake_news_pipeline.joblib    # Modelo entrenado
├── reports/
│   ├── metrics.txt                   # Métricas finales
│   └── confusion_matrix.png          # Matriz de confusión
└── data/
    └── README.md                     # Enlaces para descargar el dataset
```
