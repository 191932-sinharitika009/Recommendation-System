# FAISS ANN Index (Phase 5)

import time
import numpy as np
import faiss


class FAISSRetriever:
    """
    Two-stage retrieve-and-rank using a FAISS HNSW index.

    Stage 1: FAISS retrieves top-K candidates (fast, approximate)
    Stage 2: Caller re-ranks candidates with a slower exact scorer

    Embeddings must be L2-normalised before indexing so that inner-product
    search equals cosine similarity.
    """

    def __init__(self, M=32, ef_construction=200, ef_search=50):
        self.M               = M                # HNSW neighbours per node
        self.ef_construction = ef_construction  # build-time beam width
        self.ef_search       = ef_search        # query-time beam width
        self.index           = None
        self.movie_ids       = None             # maps row → movieId (1-indexed)

    def build(self, item_embs: np.ndarray, movie_ids: np.ndarray):
        """
        Build the HNSW index from L2-normalised item embeddings.

        item_embs  : (n_items, d) float32, already L2-normalised
        movie_ids  : (n_items,) int array mapping row index → movieId
        """
        embs = item_embs.astype(np.float32)
        # Normalise defensively (idempotent if already unit vectors)
        faiss.normalize_L2(embs)

        d = embs.shape[1]
        index = faiss.IndexHNSWFlat(d, self.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch       = self.ef_search
        index.add(embs)

        self.index     = index
        self.movie_ids = movie_ids.copy()
        return self

    def retrieve(self, query_vec: np.ndarray, k: int = 50):
        """
        Return (movie_ids, scores) for the top-k nearest items.

        query_vec : (d,) float32 user profile vector (will be normalised)
        """
        q = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        scores  = scores[0]
        indices = indices[0]

        # Filter out FAISS sentinel (-1 means not enough results)
        valid = indices >= 0
        return self.movie_ids[indices[valid]], scores[valid]

    def benchmark(self, query_vecs: np.ndarray, k: int = 50, n_runs: int = 100):
        """Measure mean per-query latency in milliseconds."""
        n = len(query_vecs)
        t0 = time.perf_counter()
        for i in range(n_runs):
            self.retrieve(query_vecs[i % n], k=k)
        elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
        return elapsed_ms


def two_stage_recommend(retriever, scorer_fn, user_profile, seen_ids,
                        k_retrieve=50, n_final=10):
    """
    Retrieve k_retrieve candidates with FAISS, re-rank with scorer_fn.

    scorer_fn(movie_ids) -> np.ndarray of scores, same order as movie_ids
    seen_ids             -> set of movieIds already rated by the user
    """
    candidate_ids, _ = retriever.retrieve(user_profile, k=k_retrieve)

    # Exclude seen items
    candidate_ids = np.array([m for m in candidate_ids if m not in seen_ids])
    if len(candidate_ids) == 0:
        return []

    scores    = scorer_fn(candidate_ids)
    order     = np.argsort(scores)[::-1][:n_final]
    return [(int(candidate_ids[i]), float(scores[i])) for i in order]
