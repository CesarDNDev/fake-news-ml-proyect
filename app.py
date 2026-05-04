from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "fake_news_pipeline.joblib"

st.set_page_config(page_title="Detector de Fake News")

st.title("Detector de Fake News")
st.write(
    "Introduce un titular y, si quieres, el cuerpo de la noticia. "
    "La app estimará si parece falsa o real según el modelo entrbkbjkbenado."
)



if not MODEL_FILE.exists():
    st.error(
        "No se encontró el modelo entrenado. Ejecuta primero: `python train_model.py`"
    )
    st.stop()

model = joblib.load(MODEL_FILE)

headline = st.text_input("Titular", placeholder="Ejemplo: Scientists confirm water is wet")
body = st.text_area("Texto de la noticia", height=220)

if st.button("Analizar noticia"):
    content = f"{headline} {body}".strip()

    if len(content) < 20:
        st.warning("Escribe un poco más de texto para poder analizarlo.")
    else:
        prediction = model.predict([content])[0]
        probabilities = model.predict_proba([content])[0]

        labels = {0: "Fake", 1: "Real"}
        result = labels[prediction]
        confidence = max(probabilities) * 100

        if prediction == 0:
            st.error(f"Predicción: {result}")
        else:
            st.success(f"Predicción: {result}")

        st.write(f"Confianza aproximada: **{confidence:.2f}%**")
        st.progress(min(int(confidence), 100))

        st.caption(
            "Esto es una ayuda educativa. Un modelo puede equivocarse, "
            "especialmente con ironía, contexto incompleto o temas nuevos."
        )
