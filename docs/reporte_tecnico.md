# Reporte técnico - Actividad 4

## Caso de uso
Se seleccionó NLP para clasificar sentimiento en comentarios de clientes de retail omnicanal.

## Criterios de selección
- Compatibilidad con Hugging Face Transformers.
- Disponibilidad como modelo preentrenado.
- Costo computacional y tamaño aproximado.
- Licencia o restricciones de uso.
- Precisión y latencia en inferencia.

## Modelos
- DistilBERT SST-2.
- BERT SST-2.
- Twitter RoBERTa Sentiment.

## Evaluación
El notebook calcula accuracy, precision, recall, F1-score y latencia promedio por texto. Los resultados de referencia están en `results/resultados_modelos_referencia.csv` y deben recalcularse al ejecutar el notebook en Colab.

## Recomendación
Para un MVP se recomienda DistilBERT porque balancea precisión, velocidad y facilidad de despliegue. Si se requiere mayor robustez y existe presupuesto de cómputo, se puede evaluar BERT o RoBERTa como segunda fase.
