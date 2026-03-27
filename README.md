# Clasificación de Reseñas de Películas - IMDB

## Descripción

Proyecto de NLP para **Film Junky Union**, una comunidad de aficionados al cine clásico. Se desarrolla un sistema de clasificación automática de reseñas de películas como **positivas** o **negativas** utilizando el dataset de IMDB.

**Objetivo:** Alcanzar un F1 Score >= 0.85

## Dataset

- **Fuente:** IMDB Movie Reviews
- **Archivo:** `imdb_reviews.tsv` (64 MB)
- **Registros:** ~50,000 reseñas etiquetadas
- **Balance:** Distribución equilibrada entre clases positiva y negativa
- **Columnas:** `tconst`, `start_year`, `rating`, `votes`, `review`, `pos`, `ds_part`

## Resultados

| # | Modelo | F1 Test | ROC AUC | Tiempo Train |
|---|--------|---------|---------|--------------|
| 0 | DummyClassifier (Baseline) | 0.0000 | 0.5000 | 0.00s |
| 1 | TF-IDF word + LR | 0.8784 | 0.9499 | 0.41s |
| 2 | TF-IDF char + LR | 0.8844 | 0.9546 | 6.77s |
| 3 | TF-IDF word + RF | 0.8564 | 0.9342 | 12.15s |
| 4 | TF-IDF word + CNB | 0.8450 | 0.9304 | 0.03s |
| 5 | spaCy + LGBM | 0.8717 | 0.9456 | 68.69s |
| **6** | **BERT + LR** | **0.8880** | **0.9567** | **0.99s** |

**Mejor modelo:** BERT + Logistic Regression (F1 = 0.8880)

## Pipeline

1. **EDA** - Análisis de distribución temporal, balance de clases, ratings
2. **Normalización** - Dos estrategias:
   - Modelos clásicos: limpieza HTML, lowercase, eliminación de dígitos y puntuación
   - BERT: limpieza mínima (solo HTML y espacios), conserva puntuación y estructura
3. **Vectorización** - TF-IDF (word/char n-grams), spaCy lematización, BERT embeddings (mean pooling)
4. **Entrenamiento** - 7 modelos con medición de tiempos
5. **Evaluación** - F1, Accuracy, ROC AUC, curvas ROC/PRC
6. **Tabla comparativa** - Métricas y tiempos de entrenamiento/predicción
7. **Validación** - Prueba con reseñas propias

## Stack Tecnológico

| Categoría | Herramientas |
|-----------|-------------|
| ML | scikit-learn, LightGBM |
| NLP | spaCy (en_core_web_sm), TF-IDF |
| Deep Learning | PyTorch (CUDA), HuggingFace Transformers (BERT) |
| Datos | pandas, NumPy |
| Visualización | matplotlib, seaborn |

## Hardware Utilizado

- **CPU:** 32 cores (aprovechados con `n_jobs=-1`)
- **RAM:** 64 GB
- **GPU:** NVIDIA RTX 4080 SUPER (BERT con mixed precision fp16)

## Estructura

```
Sprint 17/
├── README.md
├── crispy-sentiment-cls.ipynb    # Notebook principal
├── imdb_reviews.tsv               # Dataset
└── bert_embeddings.npz            # Embeddings BERT (generado al ejecutar)
```

## Cómo Ejecutar

1. Crear entorno virtual con Python 3.12:
   ```bash
   python3.12 -m venv .venv
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   python -m spacy download en_core_web_sm
   ```

3. Ejecutar el notebook:
   ```bash
   jupyter notebook notebook_(16_project).ipynb
   ```

## Conclusiones

- **Objetivo cumplido:** 5 de los 7 modelos superaron el umbral de F1 >= 0.85.
- **Mejor modelo:** BERT + LR (F1 = 0.8880), gracias a mean pooling y preprocesamiento ligero que conserva la puntuación.
- **Mejor modelo clásico:** TF-IDF char + LR (F1 = 0.8844), muy cerca de BERT pero mucho más rápido.
- Para producción con baja latencia, **TF-IDF word + LR** es la mejor opción: F1 alto, predicción casi instantánea y fácil de mantener.
- Si se busca la mayor precisión, **BERT + LR** ofrece la mejor comprensión semántica del texto.
