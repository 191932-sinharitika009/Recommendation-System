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

Evaluated on 200 validation users (temporal per-user split).

| Model | NDCG@10 | MAP | Recall@10 | P@10 |
|---|---|---|---|---|
| User-User CF | 0.0065 | 0.0004 | 0.0013 | 0.0015 |
| Item-Item CF | **0.2495** | **0.0371** | **0.0871** | **0.0855** |
| Matrix Factorization | 0.0877 | 0.0082 | 0.0253 | 0.0265 |
| TF-IDF Content | 0.0597 | 0.0045 | 0.0126 | 0.0160 |
| Sentence-Transformer | 0.0271 | 0.0021 | 0.0066 | 0.0060 |
| **Hybrid (α=1.0)** | **0.2495** | **0.0371** | **0.0871** | **0.0855** |

> Content signal adds noise on dense MovieLens-1M — alpha tuning converges to pure CF. Matrix Factorization ranks 2nd under NDCG when early-stopped at epoch 2 (best val RMSE=0.8822).

### FAISS Retrieval (Phase 5)
| Metric | Value |
|---|---|
| Index build time | 0.19s (3,883 vectors, d=384) |
| Retrieval latency (k=50) | 0.238 ms |
| Queries per second | 4,209 |

## A/B Test

| Experiment | Lift | p-value | Significant |
|---|---|---|---|
| CF vs Hybrid α=0.6 | +0.87% | 0.88 | No |
| CF vs Hybrid α=0.8 | +3.68% | 0.39 | No |
| MF vs Item-Item CF | +132.5% | ~0.0 | **Yes** |

## Cold-Start

Three-tier fallback: cold (<20 ratings) → popularity, warm (20–49) → TF-IDF content, active (50+) → CF.

> On MovieLens-1M even "cold" users have 20+ training ratings — CF outperforms the fallbacks. Cold-start strategies provide lift only for truly new users (0–5 ratings), which this dataset cannot simulate.

## Experiment Tracking

```bash
mlflow ui
# open http://localhost:5000
```
