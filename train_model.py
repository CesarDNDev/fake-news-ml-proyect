from __future__ import annotations

import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

FAKE_FILE = DATA_DIR / "Fake.csv"
TRUE_FILE = DATA_DIR / "True.csv"
MODEL_FILE = MODELS_DIR / "fake_news_pipeline.joblib"
METRICS_FILE = REPORTS_DIR / "metrics.txt"
CONFUSION_FILE = REPORTS_DIR / "confusion_matrix.png"



def clean_text(text: str) -> str:
    """Limpieza básica para texto periodístico."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def load_data() -> pd.DataFrame:
    if not FAKE_FILE.exists() or not TRUE_FILE.exists():
        raise FileNotFoundError(
            "No se encontraron los archivos Fake.csv y True.csv en la carpeta data/.\n"
            "Descarga el dataset 'Fake and Real News Dataset' y copia ambos archivos ahí."
        )

    fake_df = pd.read_csv(FAKE_FILE)
    true_df = pd.read_csv(TRUE_FILE)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Soporta dos esquemas: (title, text, subject, date) y (text)
    if "title" in df.columns:
        df["content"] = (df["title"].fillna("") + " " + df["text"].fillna(""))
    else:
        df["content"] = df["text"].fillna("")

    df = df.dropna(subset=["text"])
    df = df.drop_duplicates(subset=["content"])
    df["content"] = df["content"].map(clean_text)
    df = df[df["content"].str.len() > 20].copy()

    return df



def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_df=0.7,
                    min_df=2,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )



def save_confusion_matrix(cm) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()
    plt.xticks([0, 1], ["Fake", "Real"])
    plt.yticks([0, 1], ["Fake", "Real"])
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CONFUSION_FILE, dpi=150)
    plt.close()



def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    df = load_data()
    print(f"Registros listos para entrenar: {len(df):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    print("Entrenando modelo...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Fake", "Real"])

    metrics_text = (
        "RESULTADOS DEL MODELO\n"
        "=====================\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        "Matriz de confusión:\n"
        f"{cm}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )

    joblib.dump(pipeline, MODEL_FILE)
    METRICS_FILE.write_text(metrics_text, encoding="utf-8")
    save_confusion_matrix(cm)

    print("\nEntrenamiento completado.")
    print(f"Modelo guardado en: {MODEL_FILE}")
    print(f"Métricas guardadas en: {METRICS_FILE}")
    print(f"Matriz de confusión guardada en: {CONFUSION_FILE}")
    print("\nResumen:")
    print(metrics_text)


if __name__ == "__main__":
    main()
