# IMDB Movie Review Classification

## Description

NLP project for **Film Junky Union**, a community of classic cinema enthusiasts. An automatic classification system is built to categorize movie reviews as **positive** or **negative** using the IMDB dataset.

**Goal:** Achieve an F1 Score >= 0.85

## Dataset

- **Source:** IMDB Movie Reviews
- **File:** `imdb_reviews.tsv` (64 MB)
- **Records:** ~50,000 labeled reviews
- **Balance:** Evenly distributed between positive and negative classes
- **Columns:** `tconst`, `start_year`, `rating`, `votes`, `review`, `pos`, `ds_part`

## Results

| # | Model | F1 Test | ROC AUC | Train Time |
|---|-------|---------|---------|------------|
| 0 | DummyClassifier (Baseline) | 0.0000 | 0.5000 | 0.00s |
| 1 | TF-IDF word + LR | 0.8784 | 0.9499 | 0.41s |
| 2 | TF-IDF char + LR | 0.8844 | 0.9546 | 6.77s |
| 3 | TF-IDF word + RF | 0.8564 | 0.9342 | 12.15s |
| 4 | TF-IDF word + CNB | 0.8450 | 0.9304 | 0.03s |
| 5 | spaCy + LGBM | 0.8717 | 0.9456 | 68.69s |
| **6** | **BERT + LR** | **0.8880** | **0.9567** | **0.99s** |

**Best model:** BERT + Logistic Regression (F1 = 0.8880)

## Pipeline

1. **EDA** - Temporal distribution analysis, class balance, ratings
2. **Normalization** - Two strategies:
   - Classical models: HTML cleaning, lowercase, removal of digits and punctuation
   - BERT: minimal cleaning (HTML and whitespace only), preserves punctuation and structure
3. **Vectorization** - TF-IDF (word/char n-grams), spaCy lemmatization, BERT embeddings (mean pooling)
4. **Training** - 7 models with time tracking
5. **Evaluation** - F1, Accuracy, ROC AUC, ROC/PRC curves
6. **Comparison table** - Metrics and training/prediction times
7. **Validation** - Testing with custom reviews

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | scikit-learn, LightGBM |
| NLP | spaCy (en_core_web_sm), TF-IDF |
| Deep Learning | PyTorch (CUDA), HuggingFace Transformers (BERT) |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |

## Hardware Used

- **CPU:** 32 cores (leveraged with `n_jobs=-1`)
- **RAM:** 64 GB
- **GPU:** NVIDIA RTX 4080 SUPER (BERT with mixed precision fp16)

## Structure

```
Sprint 17/
├── README.md
├── crispy-sentiment-cls.ipynb    # Main notebook
├── imdb_reviews.tsv               # Dataset
└── bert_embeddings.npz            # BERT embeddings (generated at runtime)
```

## How to Run

1. Create a virtual environment with Python 3.12:
   ```bash
   python3.12 -m venv .venv
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   python -m spacy download en_core_web_sm
   ```

3. Run the notebook:
   ```bash
   jupyter notebook crispy-sentiment-cls.ipynb 
   ```

## Conclusions

- **Goal achieved:** 5 out of 7 models surpassed the F1 >= 0.85 threshold.
- **Best model:** BERT + LR (F1 = 0.8880), thanks to mean pooling and light preprocessing that preserves punctuation.
- **Best classical model:** TF-IDF char + LR (F1 = 0.8844), very close to BERT but much faster.
- For low-latency production, **TF-IDF word + LR** is the best choice: high F1, near-instant prediction, and easy to maintain.
- For maximum accuracy, **BERT + LR** offers the best semantic understanding of the text.
