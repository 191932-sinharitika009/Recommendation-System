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

| Model          | NDCG@10 | MAP  | Recall@5 |
|----------------|---------|------|----------|
| User-User CF   | —       | —    | —        |
| Item-Item CF   | —       | —    | —        |
| SVD            | —       | —    | —        |
| MF (PyTorch)   | —       | —    | —        |
| Content-Based  | —       | —    | —        |
| **Hybrid**     | —       | —    | —        |

_Results populated after training (Phase 7)._

## A/B Test

Compares CF-only (control) vs. Hybrid (treatment) with statistical significance (t-test, p < 0.05).

## Experiment Tracking

```bash
mlflow ui
# open http://localhost:5000
```
