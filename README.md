# Hybrid Personalized Recommendation System

End-to-end recommendation system built on MovieLens-1M covering collaborative filtering, content-based models, FAISS vector search, offline evaluation, and A/B testing.

## Architecture

```
User Ratings  ──▶  CF Models (SVD, MF)  ──┐
                                           ├──▶  Hybrid Ranker  ──▶  Top-K Recs
Movie Metadata ──▶  Content Model (SBERT) ──┘
                         │
                         └──▶  FAISS Index  ──▶  Candidate Retrieval (Stage 1)
```

## Project Structure


```
recsys/
├── data/               # raw + processed datasets (not committed)
├── notebooks/          # EDA and experimentation
├── src/
│   ├── data.py         # loaders and train/val/test splits
│   ├── cf.py           # user-user, item-item CF + SVD + PyTorch MF
│   ├── content.py      # TF-IDF and Sentence-Transformer embeddings
│   ├── hybrid.py       # weighted hybrid ranker
│   ├── retrieval.py    # FAISS HNSW index
│   ├── evaluate.py     # NDCG, MAP, Recall@K
│   ├── ab_test.py      # simulated A/B testing framework
│   └── coldstart.py    # cold-start strategies
├── mlflow_runs/        # experiment artifacts
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download MovieLens-1M and place it at `data/ml-1m/`:
```
http://files.grouplens.org/datasets/movielens/ml-1m.zip
```

## Results

Evaluated on 200 validation users (Precision@10). Full NDCG/MAP/Recall metrics coming in Phase 7.

| Model                    | P@10   | Notes |
|--------------------------|--------|-------|
| User-User CF             | 0.0015 | Noisy on sparse users |
| Item-Item CF             | 0.0855 | Best single model |
| Matrix Factorization     | 0.0070 | PyTorch, 64 factors, best val RMSE=0.8822 |
| TF-IDF Content           | 0.0160 | Genres + title, 500 features |
| Sentence-Transformer     | 0.0060 | all-MiniLM-L6-v2, d=384 |
| **Hybrid (α=1.0)**       | **0.0855** | CF dominates on dense MovieLens-1M |

> Content signal adds noise on this dense dataset. Alpha tuning showed monotonic improvement toward pure CF — content would provide lift on sparser datasets or cold-start users (see Phase 10).

### FAISS Retrieval (Phase 5)
| Metric | Value |
|---|---|
| Index build time | 0.19s (3,883 vectors, d=384) |
| Retrieval latency (k=50) | 0.238 ms |
| Queries per second | 4,209 |

## A/B Test

Compares CF-only (control) vs. Hybrid (treatment) with statistical significance (t-test, p < 0.05).

## Experiment Tracking

```bash
mlflow ui
# open http://localhost:5000
```
