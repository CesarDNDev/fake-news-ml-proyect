# Proyecto ML – Detección de Fake News
## Presentación en clase · UT2 Desarrollo de Aplicaciones IA

### Enlaces
- **App desplegada:** https://fake-news-ml-proyect-cdn.streamlit.app/
- **Repositorio GitHub:** https://github.com/CesarDNDev/fake-news-ml-proyect

---

## 1. El problema

> *"Una redacción digital recibe cada día cientos de artículos, teletipos y publicaciones de redes sociales. El equipo editorial necesita un primer filtro automático que marque qué noticias tienen pinta de ser fake news para revisarlas con prioridad."*

- **Tipo de problema:** clasificación binaria supervisada.
- **Entrada:** texto de una noticia (titular + cuerpo).
- **Salida:** `Fake` o `Real` + probabilidad.
- **Objetivo de negocio:** reducir el volumen de noticias que el verificador humano debe revisar manualmente.

---

## 2. Los datos

**Fake and Real News Dataset** (ISOT, Universidad de Victoria).

| Atributo | Valor |
|---|---|
| Registros totales | ~44.000 |
| Tras limpieza | **34.103** |
| Clases | Fake (0) / Real (1) |
| Balance | ~50% / 50% |
| Idioma | Inglés |
| Fuente principal | Reuters (reales) + webs de desinformación (falsas) |

Fuente: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## 3. Preparación de los datos

- Unión de `Fake.csv` (label 0) + `True.csv` (label 1).
- Eliminación de **nulos** y **duplicados**.
- Filtrado de textos con menos de 20 caracteres.
- **Limpieza de texto:** minúsculas, eliminación de URLs, símbolos y espacios redundantes.
- **Vectorización:** TF-IDF con n-gramas (1, 2), `max_df=0.7`, `min_df=2`.
- **División:** 80% train / 20% test, estratificado.

---

## 4. El modelo

### Tipo de algoritmo
- **Familia:** Aprendizaje supervisado.
- **Tarea:** Clasificación binaria (2 clases: `Fake` / `Real`).
- **Algoritmo elegido:** **Regresión Logística** (`sklearn.linear_model.LogisticRegression`).
- **Representación del texto:** **TF-IDF** (Term Frequency – Inverse Document Frequency) con n-gramas (1, 2).

### Pipeline completo
```
Pipeline:
  TfidfVectorizer(ngram_range=(1,2), max_df=0.7, min_df=2)
     │   (convierte texto → vector numérico)
     ▼
  LogisticRegression(class_weight="balanced", max_iter=2000)
     │   (clasifica como Fake / Real)
     ▼
  predict_proba() → probabilidad de cada clase
```

### Por qué Regresión Logística
- Es un clasificador **lineal**, **simple** e **interpretable** (podemos ver qué palabras "pesan" en cada clase).
- Muy rápido de entrenar (~segundos sobre 27.000 documentos).
- Baseline sólido y ampliamente recomendado para clasificación de texto con TF-IDF.
- Encaja con lo que pide el enunciado (clasificación + Regresión Logística / Random Forest).

---

## 5. Resultados

### Métricas globales

| Métrica | Valor |
|---|---|
| **Accuracy** | **0.9719** |
| Precision (Fake) | 0.98 |
| Recall (Fake) | 0.96 |
| Precision (Real) | 0.96 |
| Recall (Real) | 0.98 |

### Matriz de confusión

|             | Pred. Fake | Pred. Real |
|-------------|:---:|:---:|
| **Real Fake** | **3276** | 134 |
| **Real Real** | 58 | **3353** |

### Análisis de errores (2.81% del test)

- **134 falsos negativos** (Fake → Real): bulos bien redactados imitando el estilo de un medio serio.
- **58 falsos positivos** (Real → Fake): noticias reales cortas, de opinión o con léxico coloquial.

---

## 6. Mejoras aplicadas

- `class_weight="balanced"` → evita sesgo en caso de ligero desbalance.
- N-gramas `(1, 2)` → captura expresiones como *"breaking news"* o *"sources confirm"*.
- Filtros `max_df` y `min_df` → descarta ruido léxico.
- Limpieza de URLs y símbolos.

---

## 7. Despliegue

### Stack
- **Streamlit** (interfaz web)
- **scikit-learn** (modelo)
- **joblib** (serialización)
- **Streamlit Community Cloud** (hosting gratuito)

### Demo en vivo
**https://fake-news-ml-proyect-cdn.streamlit.app/**

Flujo:
1. El usuario pega el titular y el cuerpo de una noticia.
2. La app ejecuta `model.predict_proba()` sobre el texto limpio.
3. Muestra la etiqueta predicha y la confianza.

---

## 8. Limitaciones

- Solo noticias en **inglés**.
- Entrenado con noticias de un período concreto (2015-2017 aprox.) → puede degradarse con temas recientes.
- No verifica fuentes en internet: clasifica **por estilo**, no por veracidad objetiva.
- Un modelo basado en frecuencias de palabras puede ser engañado por textos redactados para parecer "serios".

---

## 9. Próximos pasos

- Probar **Random Forest**, **SVM** y **Naive Bayes** para comparar.
- Probar modelos de **deep learning** (LSTM, BERT).
- Extender a noticias en **español**.
- Integrar verificación cruzada con APIs de fact-checking.

---

## 10. Demo

1. **Notebook** `ProyectoMLFakeNews.ipynb` → proceso completo con salidas.
2. **App Streamlit** desplegada → probar predicciones en vivo.
3. **Repositorio GitHub** → código organizado y reproducible.

---

## Preguntas
