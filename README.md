# Actividad 4 - Comparación de modelos preentrenados en Hugging Face

## Caso de uso
Clasificación de sentimiento en comentarios de clientes de retail omnicanal usando modelos preentrenados de Hugging Face.

## Modelos evaluados
1. `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
2. `textattack/bert-base-uncased-SST-2`
3. `cardiffnlp/twitter-roberta-base-sentiment-latest`

## Estructura del repositorio
```text
actividad4_hf_sentiment/
├── README.md
├── requirements.txt
├── data/
│   └── customer_sentiment_dataset.csv
├── notebooks/
│   └── Actividad_4_HF_sentiment.ipynb
├── src/
│   └── evaluate_models.py
├── results/
│   └── resultados_modelos_referencia.csv
└── docs/
    └── reporte_tecnico.md
```

## Ejecución en Google Colab
1. Activar GPU: `Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU`.
2. Abrir `notebooks/Actividad_4_HF_sentiment.ipynb`.
3. Ejecutar las celdas en orden.
4. Revisar los archivos generados en `/content/results`.

## Conclusión esperada
El modelo recomendado para MVP suele ser DistilBERT si se prioriza eficiencia. BERT o RoBERTa pueden ser mejores si la prioridad es precisión, pero requieren mayor costo computacional.
