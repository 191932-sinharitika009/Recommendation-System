# Project Understanding — Phase-by-Phase Notes

Running log of what was built, why each decision was made, and key concepts per phase.

---

## Phase 0: Setup ✅

### What we did
- Created full project folder structure: `data/`, `notebooks/`, `src/`, `mlflow_runs/`
- Wrote `requirements.txt` with all dependencies
- Implemented `src/data.py` (the only file with real code this phase)
- Created placeholder files for all future modules (`cf.py`, `content.py`, etc.)
- Added `.gitignore` — excludes `data/` and `mlflow_runs/` from git (large files, not source code)
- Wrote `README.md` with architecture diagram and empty results table

### Key decisions
- **`data/` is gitignored** — MovieLens-1M is ~25MB raw; datasets are never committed, only code
- **`data.py` implemented now** — every phase depends on data loading and splits, so it makes sense to build it first
- **Placeholder files created upfront** — gives a clear map of what's coming and makes imports work from day one

### What `src/data.py` does
- `load_ratings()` / `load_movies()` / `load_users()` — reads the `.dat` files (`::`-separated format)
- `split_per_user()` — **temporal split per user**: sorts each user's ratings by timestamp, holds out the last 10% as test, next 10% as val, rest as train. This is the correct way to split RecSys data — avoids leaking future interactions into training
- `build_interaction_matrix()` — builds a sparse CSR matrix (users × items) from the ratings dataframe
- `compute_sparsity()` — measures how empty the matrix is; expected ~95%+ for MovieLens

### Concepts to remember
- **Sparsity**: most users rate very few items out of thousands. A 95% sparse matrix means 95% of cells are 0. This is why CF is hard — you're predicting from very little signal
- **Temporal split over random split**: random split can leak future ratings into training (a user's future behavior influences their past profile). Temporal split respects the real-world order
- **CSR (Compressed Sparse Row)**: efficient format for row-wise operations (e.g., fetch all items a user rated). Used because user-user similarity needs fast row access

---

## Phase 1: Data & EDA ✅

### What we did
- Created `notebooks/01_eda.ipynb` with 10 sections covering the full dataset
- Verified data loads correctly via `src/data.py`
- Ran train/val/test split and saved CSVs to `data/`

### Dataset facts (MovieLens-1M)
| Stat | Value |
|------|-------|
| Users | 6,040 |
| Movies | ~3,706 unique rated |
| Ratings | 1,000,209 |
| Sparsity | ~95%+ |
| Avg ratings/user | ~165 |
| Avg ratings/movie | ~270 |

### Key observations from EDA
- **Long-tail**: a small % of popular movies get the majority of ratings; most movies have very few. This is the core challenge in RecSys — being useful beyond the top-10 blockbusters
- **Rating skew**: users tend to rate movies they liked (selection bias). The average rating is above 3.5, not 3.0
- **Cold-start users**: users with < 20 ratings need special handling — covered in Phase 10
- **Genre**: Drama and Comedy dominate the catalog

### Split strategy: per-user temporal split
- Each user's ratings sorted by timestamp
- Last 10% → test, next 10% → val, rest → train
- **Why temporal and not random?** Random split can leak future ratings into training — e.g., a user's 2001 rating influences their 2000 profile. Temporal split mirrors real deployment where you always predict future behavior from past history
- **Why per-user?** Guarantees every user has training data. A global random split could leave some users entirely in test with no train signal

### Sparsity intuition
- 95%+ sparsity means the user-item matrix is almost entirely zeros
- Out of 6040 × 3953 = ~23.9M possible ratings, only ~1M exist
- This is why collaborative filtering is hard — you must predict from very sparse signal
- CSR (Compressed Sparse Row) format stores only non-zero values efficiently

---

## Phase 2: Memory-Based Collaborative Filtering ✅

### What we did
- Implemented `UserUserCF` and `ItemItemCF` classes in `src/cf.py`
- Created `notebooks/02_cf.ipynb` — fits both models, shows recommendations with movie titles, evaluates Precision@10 on the val set

### How each model works

**User-User CF**
1. Build the dense user-item matrix (6040 × 3952, float32, ~90MB)
2. Row-normalize each user vector to unit length
3. Compute user-user cosine similarity: `sim = normalized @ normalized.T` → (6040 × 6040)
4. For a target user: find top-K most similar users (neighbors), then for each unrated item compute a weighted average of neighbor ratings:
   - `score(i) = (sim_neighbors · ratings_neighbors_i) / (|sim_neighbors| · rated_mask_i)`
5. Return top-N items by score, excluding already-seen items

**Item-Item CF**
1. Same dense matrix, but transpose it → (3952 × 6040, items as rows)
2. Column-normalize (item vectors) → compute item-item cosine similarity: (3952 × 3952)
3. For a target user: for each item they rated, accumulate weighted scores from similar items
   - `scores += sim[item] * user_rating[item]` for every rated item
4. Divide by number of rated items to normalize, zero out seen items, return top-N

### Key design: fully vectorized
Both models avoid Python loops over items/users. The critical operations are matrix multiplications (`@`), enabling fast fit (< 2s) on a 6040-user dataset.

### Results
| Model | Precision@10 (200 users) |
|---|---|
| User-User CF | 0.0015 |
| Item-Item CF | 0.0855 |

### Why Item-Item CF wins by so much
- **User-User CF** surfaces niche, high-prestige films that user's neighbors universally rated 5 stars — but these often aren't what the target user watches *next* (temporal val split). Score = 5.0 for all top recs is a sign the model is over-confident on rarely-rated items
- **Item-Item CF** recommends items similar to the user's own history. Because the val set contains the user's actual next interactions, items similar to their past choices are a much better predictor
- This is a well-known result: Item-Item CF is generally more stable and accurate than User-User CF, which is why Amazon pioneered it for production use

### Concepts to remember
- **Cosine similarity**: measures angle between vectors, ignoring magnitude. Two users who both love/hate the same movies will have similarity ≈ 1 even if one rates on a 1–3 scale and the other on a 3–5 scale
- **Top-K neighbors**: using all users adds noise (dissimilar users degrade predictions). K=50 is a common default
- **Precision@K**: fraction of top-K recommendations that appear in the user's held-out val set. Low values (< 0.1) are normal for sparse RecSys datasets
- **Item-Item sim matrix is stable**: user tastes change over time, but item-item relationships (Sci-Fi fans like both Star Wars and Alien) are relatively static — another reason Item-Item CF is preferred in practice

---

## Phase 3: Model-Based Collaborative Filtering 🔲

_To be filled after Phase 3 is complete._

---

## Phase 4: Content-Based Recommender 🔲

_To be filled after Phase 4 is complete._

---

## Phase 5: FAISS ANN Index 🔲

_To be filled after Phase 5 is complete._

---

## Phase 6: Hybrid Model 🔲

_To be filled after Phase 6 is complete._

---

## Phase 7: Offline Evaluation 🔲

_To be filled after Phase 7 is complete._

---

## Phase 8: A/B Testing Framework 🔲

_To be filled after Phase 8 is complete._

---

## Phase 9: MLflow Experiment Tracking 🔲

_To be filled after Phase 9 is complete._

---

## Phase 10: Cold-Start Strategies 🔲

_To be filled after Phase 10 is complete._
