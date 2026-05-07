# Collaborative Filtering: memory-based (Phase 2) and model-based (Phase 3)

import numpy as np
import pandas as pd

from .data import build_interaction_matrix


class UserUserCF:
    """Memory-based User-User Collaborative Filtering via cosine similarity."""

    def __init__(self, K=50):
        self.K = K
        self.matrix = None   # dense (n_users, n_items)
        self.sim = None      # (n_users, n_users) cosine similarity
        self.n_users = None
        self.n_items = None
        self.user_rated = {}  # user_idx -> set of item indices seen in train

    def fit(self, train_df):
        self.n_users = train_df["userId"].max()
        self.n_items = train_df["movieId"].max()

        sparse = build_interaction_matrix(train_df, self.n_users, self.n_items)
        self.matrix = sparse.toarray().astype(np.float32)  # (n_users, n_items)

        # Row-normalize for cosine similarity
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_mat = self.matrix / norms
        self.sim = norm_mat @ norm_mat.T  # (n_users, n_users)

        for uid, grp in train_df.groupby("userId"):
            self.user_rated[uid - 1] = set(grp["movieId"].values - 1)

        return self

    def recommend(self, user_id, n=10):
        u = user_id - 1
        sims = self.sim[u].copy()
        sims[u] = 0.0  # exclude self

        # Top-K neighbor indices
        neighbors = np.argsort(sims)[::-1][: self.K]
        neighbor_sims = sims[neighbors]                     # (K,)
        neighbor_ratings = self.matrix[neighbors, :]        # (K, n_items)

        # Weighted average: numerator = sim . ratings, denom = sim . (rated > 0)
        numerator = neighbor_sims @ neighbor_ratings        # (n_items,)
        rated_mask = (neighbor_ratings > 0).astype(np.float32)
        denominator = np.abs(neighbor_sims) @ rated_mask    # (n_items,)
        scores = numerator / (denominator + 1e-8)

        # Zero out already-seen items
        seen = self.user_rated.get(u, set())
        for i in seen:
            scores[i] = 0.0

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i + 1), float(scores[i])) for i in top_idx if scores[i] > 0]


class ItemItemCF:
    """Memory-based Item-Item Collaborative Filtering via cosine similarity."""

    def __init__(self, K=50):
        self.K = K
        self.matrix = None   # dense (n_users, n_items)
        self.sim = None      # (n_items, n_items) cosine similarity
        self.n_users = None
        self.n_items = None
        self.user_rated = {}

    def fit(self, train_df):
        self.n_users = train_df["userId"].max()
        self.n_items = train_df["movieId"].max()

        sparse = build_interaction_matrix(train_df, self.n_users, self.n_items)
        self.matrix = sparse.toarray().astype(np.float32)  # (n_users, n_items)

        # Column-normalize for item cosine similarity
        item_mat = self.matrix.T                            # (n_items, n_users)
        norms = np.linalg.norm(item_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_mat = item_mat / norms
        self.sim = norm_mat @ norm_mat.T                    # (n_items, n_items)

        for uid, grp in train_df.groupby("userId"):
            self.user_rated[uid - 1] = set(grp["movieId"].values - 1)

        return self

    def recommend(self, user_id, n=10):
        u = user_id - 1
        seen = self.user_rated.get(u, set())
        if not seen:
            return []

        user_vec = self.matrix[u]                           # (n_items,)
        scores = np.zeros(self.n_items, dtype=np.float32)

        for item_idx in seen:
            sims = self.sim[item_idx].copy()
            sims[item_idx] = 0.0
            rating = user_vec[item_idx]
            scores += sims * rating

        # Normalize by number of rated items that contributed
        scores /= (len(seen) + 1e-8)

        # Zero out already-seen items
        for i in seen:
            scores[i] = 0.0

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i + 1), float(scores[i])) for i in top_idx if scores[i] > 0]
