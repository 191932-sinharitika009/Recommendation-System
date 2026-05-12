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

## Phase 3: Model-Based Collaborative Filtering ✅

### What we did
- Added `MatrixFactorization` class (and helpers `_MFModel`, `_RatingsDataset`) to `src/cf.py`
- Created `notebooks/03_mf.ipynb` — trains 20 epochs, plots learning curve, evaluates Precision@10

### How it works
**Architecture:** Each user and item gets a latent vector of size `n_factors=64`. Predicted rating = `global_mean + user_emb · item_emb + user_bias + item_bias`

**Training:**
- Loss = MSE + explicit L2 reg on embeddings only (not biases)
- Optimizer: Adam with `lr=0.005`, `weight_decay=0` (reg applied manually in loss)
- Global mean baked into model so predictions start near ~3.58 instead of 0
- Mini-batch SGD with `batch_size=2048`

**Bug we fixed:** Original code used `weight_decay=0.02` inside Adam directly. Adam's weight_decay is too aggressive compared to SGD — it collapsed embeddings toward zero, keeping RMSE stuck at 3.3 for all 20 epochs. Fix: set `weight_decay=0` in Adam, add `reg * (user_emb.pow(2).mean() + item_emb.pow(2).mean())` to the loss manually.

### Results
| Model | Precision@10 | Val RMSE |
|---|---|---|
| User-User CF | 0.0015 | — |
| Item-Item CF | 0.0855 | — |
| Matrix Factorization | 0.0070 | 0.8822 (best at epoch 2) |

Training time: ~410s on CPU for 20 epochs over 805K ratings.

### Why MF still loses to Item-Item CF on P@10
The model overfits after epoch 2 (train RMSE → 0.51, val RMSE → 1.12). At epoch 20, it scores obscure items very highly (because rare user-item pairs dominate gradient updates), producing recommendations that look off. Early stopping at epoch 2 would give ~0.88 val RMSE and better ranking quality.

### Concepts to remember
- **Matrix Factorization**: decomposes the sparse user-item matrix into two dense matrices (U × F and I × F). The dot product of a user and item vector gives the predicted rating
- **Global mean + biases**: without these, predictions start at 0 while actual ratings are 1–5. The model wastes early epochs just learning the mean shift
- **weight_decay in Adam vs SGD**: in SGD, weight_decay=0.02 is mild. In Adam, it interacts with the adaptive learning rates and can be 10× more aggressive — always use manual L2 reg in the loss when using Adam for RecSys
- **Overfitting in MF**: the model has `n_users * n_factors + n_items * n_factors` parameters (~640K for this dataset). With 800K training samples, it overfits quickly. Solutions: early stopping, dropout on embeddings, or lower n_factors

---

## Phase 4: Content-Based Recommender ✅

### What we did
- Implemented `TFIDFRecommender` and `SentenceTransformerRecommender` in `src/content.py`
- Created `notebooks/04_content.ipynb` — fits both, shows recommendations with titles, evaluates Precision@10

### How each model works

**TF-IDF Recommender**
1. Build corpus: `genres (pipe → space) + title` for each movie
2. Fit `TfidfVectorizer(max_features=500)` → sparse item matrix (3883 × 500)
3. User profile = weighted mean of TF-IDF vectors for rated items (weights = `rating − 2.5`, centred at 0)
4. Score candidates via cosine similarity between profile and all item vectors
5. Return top-N, excluding seen items

**Sentence-Transformer Recommender**
1. Same corpus, encoded with `all-MiniLM-L6-v2` → dense (3883 × 384) float32 matrix
2. Embeddings L2-normalised so scoring = dot product (= cosine similarity)
3. User profile built the same way (weighted mean, then normalise)
4. Score = `item_embs @ profile` — fast matrix-vector product

### Results
| Model | Precision@10 |
|---|---|
| User-User CF | 0.0015 |
| Matrix Factorization | 0.0070 |
| Sentence-Transformer | 0.0060 |
| TF-IDF Content | 0.0160 |
| Item-Item CF | 0.0855 |

### Why TF-IDF beats Sentence-Transformer here
Genre + title text is short and keyword-heavy ("Action Comedy Drama"). TF-IDF excels at exact keyword matching. Sentence-Transformers are trained on rich natural language — they add little value when the "document" is just a list of genre tags. ST would likely win with full movie plot descriptions or review text.

### Why both content models lose to Item-Item CF
Content models only see what a movie *is* (genres, title). CF models see what users *do* (rating patterns). A user's interaction history is a far stronger signal than genre tags — two users who both love "Toy Story" are similar regardless of whether they also rate "other animated films" highly.

### Concepts to remember
- **User profile as weighted mean**: subtract 2.5 from ratings before weighting — items rated 5 pull the profile toward them, items rated 1 push the profile away. This is more expressive than a simple mean
- **Content-based cold start advantage**: content models work for new users (just ask their preferences) and new items (just read the metadata). CF requires prior interactions — this is why hybrid models combine both
- **L2 normalisation + dot product = cosine similarity**: pre-normalising the item matrix means scoring at inference is a single matrix-vector multiply — very fast

---

## Phase 5: FAISS ANN Index ✅

### What we did
- Implemented `FAISSRetriever` and `two_stage_recommend` in `src/retrieval.py`
- Created `notebooks/05_faiss.ipynb` — builds index, benchmarks latency, runs two-stage pipeline

### How it works

**Index: HNSW (Hierarchical Navigable Small World)**
- Graph-based ANN structure where each node connects to M=32 neighbours
- At build time, `ef_construction=200` controls beam width (more = better graph, slower build)
- At query time, `ef_search=50` controls recall vs speed trade-off
- Metric: inner product on L2-normalised vectors = cosine similarity
- Build: 0.19s for 3,883 vectors (d=384)

**Two-stage pipeline**
1. **Stage 1 — Retrieve**: FAISS returns top-50 candidate movie IDs in ~0.24ms
2. **Stage 2 — Re-rank**: a slower, exact scorer (e.g. hybrid model) re-scores only those 50 candidates and returns top-N

### Results
| Metric | Value |
|---|---|
| Index build time | 0.19s |
| Retrieval latency (k=50) | 0.238 ms |
| Queries per second | 4,209 (CPU only) |

### Concepts to remember
- **Why ANN over exact search**: exhaustive cosine similarity over 1M items takes ~100ms per query. HNSW does it in < 1ms with >95% recall — mandatory at production scale
- **HNSW vs IVF**: HNSW is graph-based (no training needed, good recall at low k). IVF (Inverted File Index) clusters vectors first (requires training, faster at very large scale). For < 1M items, HNSW is the default choice
- **efSearch trade-off**: higher efSearch → higher recall → higher latency. Tune based on SLA (e.g. p99 < 10ms)
- **L2-normalise before indexing**: converts inner product to cosine similarity. Do this once at build time so queries don't need to know the normalisation happened
- **Two-stage is the production pattern**: retrieval (cheap, approximate, runs on all items) feeds a re-ranker (expensive, exact, runs on ~50–200 candidates). This decouples scale from quality

---

## Phase 6: Hybrid Model ✅

### What we did
- Implemented `HybridRecommender` and score helpers in `src/hybrid.py`
- Created `notebooks/06_hybrid.ipynb` — sweeps alpha 0.0→1.0 on 200 val users, plots P@10 vs alpha, reports best alpha

### How it works
1. For each user, compute a full score vector over all unseen items from **both** models:
   - `_cf_scores_full()` — runs the Item-Item CF scoring loop over all items (same as `recommend()` but without the top-k cut), returns `{movieId: score}`
   - `_content_scores_full()` — computes `item_embs @ user_profile` for all movies, returns `{movieId: cosine_sim}`
2. Take the **intersection** of movie IDs both models can score
3. **Min-max normalize** each score vector independently to [0, 1]
4. Combine: `hybrid = alpha * cf_norm + (1 − alpha) * content_norm`
5. Return top-N by hybrid score

### Alpha tuning
Instead of calling `recommend()` 11 times per user, we precompute both score dicts once per user and reuse them across all alpha values — 11× faster sweep.

### Results
| alpha | P@10 |
|---|---|
| 0.0 (pure content) | 0.0060 |
| 0.1 | 0.0240 |
| 0.2 | 0.0430 |
| 0.3 | 0.0600 |
| 0.4 | 0.0720 |
| 0.5 | 0.0795 |
| 0.6 | 0.0810 |
| 0.7 | 0.0835 |
| 0.8 | 0.0845 |
| 0.9 | 0.0845 |
| **1.0 (pure CF)** | **0.0855** |

Best alpha = 1.0 — the hybrid does not improve over pure Item-Item CF.

### Why CF dominates completely
- MovieLens-1M is a **dense** dataset (~165 ratings/user, ~270 ratings/movie). In this regime, collaborative signal is extremely strong
- The content features (genre tags + title) are short and keyword-heavy — they capture *what category* a movie is, not *how good* it is
- Content would provide lift on **sparse** datasets, **new items** (no ratings yet), or **cold-start users** (< 5 ratings) — exactly what Phase 10 targets

### Concepts to remember
- **Hybrid systems don't always win**: on dense data, the weaker model degrades the stronger one. Hybridization pays off when models have complementary strengths (e.g. CF strong on popular items, content strong on long-tail)
- **Min-max normalization before blending**: raw CF scores (weighted avg ratings, ~0–5) and content scores (cosine similarity, ~0–1) live on different scales. Normalize first so neither dominates by magnitude
- **Intersection vs union**: we use the intersection of movie IDs both models cover. Union would require a fallback score for missing movies (e.g. 0 or the min), which can introduce bias
- **Alpha as A/B test variable**: alpha is exactly the knob you'd expose in an A/B test — control (alpha=1, pure CF) vs treatment (alpha=0.6, hybrid). Phase 8 formalizes this

---

## Phase 7: Offline Evaluation ✅

### What we did
- Implemented `dcg_at_k`, `ndcg_at_k`, `recall_at_k`, `precision_at_k`, `average_precision`, and `evaluate_model` in `src/evaluate.py` from scratch
- Created `notebooks/07_evaluate.ipynb` — fits all 6 models, runs full metric comparison, saves bar chart

### Metrics explained (implemented from scratch)

**NDCG@K (Normalized Discounted Cumulative Gain)**
- Rewards relevant items ranked *higher* in the list via a log₂ discount
- `DCG@K = Σ relevance_i / log₂(i+1)` for positions i=1..K
- Normalized by IDCG (ideal DCG — what you'd get if all relevant items were at the top)
- Range [0,1]; 1.0 = perfect ranking. Most informative single metric for ranking quality

**MAP (Mean Average Precision)**
- For each relevant item found in the list, compute precision at that position, then average
- `AP = Σ precision@i / |relevant|` across hit positions
- Averaged over all users = MAP
- Strictest metric — penalizes any gap between relevant items in the ranked list

**Recall@K**
- Fraction of the user's ground-truth items found in top-K
- `Recall@K = |recommended ∩ relevant| / |relevant|`
- Shows how much of the "correct answer" your list recovers

**Precision@K**
- Fraction of your top-K that are actually relevant
- Trade-off with Recall: higher K → higher Recall, lower Precision

### Results (200 val users)
| Model | NDCG@5 | NDCG@10 | MAP | Recall@5 | Recall@10 | P@10 |
|---|---|---|---|---|---|---|
| User-User CF | 0.0032 | 0.0065 | 0.0004 | 0.0007 | 0.0013 | 0.0015 |
| Item-Item CF | **0.2099** | **0.2495** | **0.0371** | **0.0521** | **0.0871** | **0.0855** |
| Matrix Factorization | 0.0725 | 0.0877 | 0.0082 | 0.0141 | 0.0253 | 0.0265 |
| TF-IDF Content | 0.0441 | 0.0597 | 0.0045 | 0.0072 | 0.0126 | 0.0160 |
| Sentence-Transformer | 0.0196 | 0.0271 | 0.0021 | 0.0047 | 0.0066 | 0.0060 |
| Hybrid (α=1.0) | 0.2099 | 0.2495 | 0.0371 | 0.0521 | 0.0871 | 0.0855 |

### Key insights

**MF jumps to 2nd place under NDCG**
Phase 3 reported P@10=0.0070 for MF (20 epochs, overfit). Phase 7 uses n_epochs=2 (early stopping at best val RMSE=0.8822). Early stopping prevents the model from over-scoring obscure items, giving dramatically better ranking quality: P@10=0.0265, NDCG@10=0.0877. **Lesson: always evaluate ranking quality, not just RMSE.**

**NDCG vs Precision tell different stories**
- P@10 only checks if relevant items are in the top-10 (binary)
- NDCG@10 also checks *where* in the top-10 they land — an item at rank 1 is worth log₂(3)≈1.58× more than one at rank 2
- This makes NDCG a much richer signal for ranking model quality

**Item-Item CF dominates across every metric**
NDCG@10=0.2495 means the model achieves ~25% of perfect ranking quality — strong for a simple memory-based approach with no training. The gap to MF (0.0877) shows how powerful nearest-neighbor methods are on dense datasets.

### Concepts to remember
- **DCG discount = log₂(position+1)**: position 1 → log₂(2)=1 (no discount), position 10 → log₂(11)≈3.46 (strong discount). The log makes early positions exponentially more valuable
- **Why MAP is stricter than NDCG**: MAP requires relevant items to appear *compactly* near the top. NDCG tolerates spreading them out as long as they're individually ranked high
- **Early stopping is critical for MF**: train RMSE and val RMSE diverge fast (epoch 2 best). A model that minimizes rating error (RMSE) doesn't necessarily maximize ranking quality (NDCG) — these are different objectives
- **Recall@K grows with K**: always report the K you chose and justify it (K=10 is standard for "top recommendations"; K=100 for candidate generation stages)

---

## Phase 8: A/B Testing Framework ✅

### What we did
- Implemented `ABTestFramework` in `src/ab_test.py` — accepts any two `recommend_fn` callables, computes per-user NDCG@10, runs a paired t-test, returns lift %, p-value, significance flag, and 95% CI
- Created `notebooks/08_ab_test.ipynb` — ran 3 experiments comparing models

### How it works
1. For each user in the sample, compute the metric (NDCG@10 by default) for both control and treatment
2. This gives two arrays of per-user scores — one per model
3. Run a **paired t-test** (`scipy.stats.ttest_rel`): each user's control score is compared to their treatment score
4. Report: control mean, treatment mean, lift %, p-value, significance (p < 0.05), 95% CI on the mean difference

**Why paired t-test?** Each user sees both models (offline simulation), so we can subtract out individual user difficulty. A user who is generally hard to recommend for will have low scores on both models — the paired test removes this noise and is more statistically powerful than an independent t-test.

### Results

| Experiment | Control | Treatment | Control NDCG@10 | Treatment NDCG@10 | Lift | p-value | Significant |
|---|---|---|---|---|---|---|---|
| 1 | Item-Item CF | Hybrid α=0.6 | 0.2495 | 0.2517 | +0.87% | 0.8804 | No |
| 2 | Item-Item CF | Hybrid α=0.8 | 0.2495 | 0.2587 | +3.68% | 0.3879 | No |
| 3 | MF (epoch 2) | Item-Item CF | 0.1073 | 0.2495 | +132.55% | ~0.0 | **Yes** |

### Key insights

**Experiments 1 & 2 — hybrid is not significantly different from pure CF**
Both hybrids show small positive lifts (+0.87% and +3.68%), but neither is statistically significant (p=0.88, p=0.39). The 95% CI straddles zero in both cases, meaning we cannot conclude the treatment is better or worse. This is a **"no-ship" decision**: there is no evidence the hybrid helps, so deploying it adds complexity with no confirmed gain.

**This contradicts Phase 6's P@10 sweep** — which showed content signal hurts. NDCG@10 tells a slightly different story: content might be nudging relevant items higher *within* the top-10 even when it doesn't add new relevant items. P@10 is blind to ordering within top-K; NDCG is not. With more users (n=2000+), the α=0.8 blend might reach significance.

**Experiment 3 — ItemItemCF significantly outperforms MF**
+132% lift with p≈0 and a 95% CI entirely above zero (0.0965, 0.1879). This is a **definitive result**: ItemItemCF is unambiguously better than MF at 2 epochs on this dataset. The effect size is so large that 200 users is more than enough to detect it.

**MF variance across runs**
MF NDCG@10 was 0.0877 in Phase 7 but 0.1073 here. Embeddings are randomly initialized and mini-batch SGD is stochastic — with only 2 epochs the model has high run-to-run variance. Lesson: report mean ± std over multiple seeds for fair model comparison.

### Concepts to remember
- **Paired vs independent t-test**: paired removes user-level variance (same users see both arms). Independent is used when different users are assigned to control and treatment (real online A/B test). Paired is more powerful but requires the same user pool
- **Statistical significance ≠ practical significance**: a +0.87% lift might be real but not worth the engineering cost of maintaining a hybrid system. Always report effect size alongside p-value
- **p-value interpretation**: p=0.39 means "if the models were equal, we'd see a difference this large 39% of the time by chance" — not evidence for the hybrid being equal, just insufficient evidence it's better
- **Sample size matters**: with n=200, only large effects (like MF vs CF, +132%) are detectable. Small effects (< 5% lift) require n=1000+ users to achieve 80% statistical power
- **95% CI**: the interval (-0.0117, 0.0301) for Exp 2 shows the true mean difference is probably between -1.2% and +3% — ambiguous, not actionable

---

## Phase 9: MLflow Experiment Tracking 🔲

_To be filled after Phase 9 is complete._

---

## Phase 10: Cold-Start Strategies 🔲

_To be filled after Phase 10 is complete._
