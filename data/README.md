# Dataset

Los archivos CSV **no** se incluyen en el repositorio por tamaño (~85 MB).

## Opción 1 - Kaggle (fuente original)

Dataset: **Fake and Real News Dataset**
<https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>

Descarga `Fake.csv` y `True.csv` y colócalos en esta carpeta `data/`.

## Opción 2 - Mirror HuggingFace (sin credenciales)

```powershell
curl.exe -L -o "data\Fake.csv" "https://huggingface.co/datasets/Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022/resolve/main/raw/Fake%20ok.csv"
curl.exe -L -o "data\True.csv" "https://huggingface.co/datasets/Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022/resolve/main/raw/True%20ok%20v2.csv"
```

El código (`train_model.py` y el notebook) soporta ambos esquemas (con o sin las columnas `title`, `subject`, `date`).
