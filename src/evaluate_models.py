import os
import time
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import pipeline

MODELOS = [
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/bert-base-uncased-SST-2",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
]

def normalizar_etiqueta(label):
    label = str(label).lower()
    if "positive" in label or label in ["pos", "label_1"]:
        return "positive"
    if "negative" in label or label in ["neg", "label_0"]:
        return "negative"
    if "neutral" in label:
        return "neutral"
    return label

def evaluar(data_path="data/customer_sentiment_dataset.csv", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    textos = df["text"].tolist()
    y_real = df["label"].tolist()
    device = 0 if torch.cuda.is_available() else -1
    resultados = []
    for model_id in MODELOS:
        clf = pipeline("text-classification", model=model_id, device=device, truncation=True, batch_size=8)
        _ = clf([textos[0]])
        inicio = time.perf_counter()
        salidas = clf(textos)
        fin = time.perf_counter()
        y_pred = [normalizar_etiqueta(s["label"]) for s in salidas]
        acc = accuracy_score(y_real, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_real, y_pred, labels=["negative", "positive"], average="macro", zero_division=0)
        resultados.append({
            "modelo": model_id,
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "latencia_ms_por_texto": ((fin-inicio)/len(textos))*1000
        })
    pd.DataFrame(resultados).to_csv(os.path.join(output_dir, "resultados_modelos.csv"), index=False)
    return resultados

if __name__ == "__main__":
    print(evaluar())
