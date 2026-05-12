# Hybrid Ranker (Phase 6)

import numpy as np


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _cf_scores_full(cf_model, user_id: int) -> dict:
    """Raw ItemItemCF score for every unseen item. Returns {movieId: score}."""
    u_idx = user_id - 1
    seen = cf_model.user_rated.get(u_idx, set())
    user_vec = cf_model.matrix[u_idx]                    # (n_items,)

    scores = np.zeros(cf_model.n_items, dtype=np.float32)
    for item_idx in seen:
        scores += cf_model.sim[item_idx] * float(user_vec[item_idx])
    scores /= (len(seen) + 1e-8)

    return {i + 1: float(scores[i]) for i in range(cf_model.n_items) if i not in seen}


def _content_scores_full(content_model, user_id: int, train_df) -> dict:
    """Cosine similarity between user profile and every item. Returns {movieId: score}."""
    profile = content_model._user_profile(user_id, train_df)
    if profile is None:
        return {}
    sims = content_model.item_embs @ profile              # (n_movies,)
    seen = content_model.user_rated.get(user_id, set())
    return {
        int(content_model.movie_ids[i]): float(sims[i])
        for i in range(len(content_model.movie_ids))
        if int(content_model.movie_ids[i]) not in seen
    }


class HybridRecommender:
    """Weighted hybrid of CF and content-based scores.

    Both score vectors are min-max normalized to [0, 1] before combining:
        hybrid = alpha * cf_norm + (1 - alpha) * content_norm

    alpha=1.0 → pure CF, alpha=0.0 → pure content. Tune alpha on val set.
    """

    def __init__(self, cf_model, content_model, alpha: float = 0.6):
        self.cf_model      = cf_model
        self.content_model = content_model
        self.alpha         = alpha

    def recommend(self, user_id: int, train_df, n: int = 10):
        """Return [(movieId, score), ...] sorted descending."""
        cf_dict = _cf_scores_full(self.cf_model, user_id)
        ct_dict = _content_scores_full(self.content_model, user_id, train_df)

        # Only score items both models cover
        common_ids = list(set(cf_dict) & set(ct_dict))
        if not common_ids:
            return []

        cf_arr = np.array([cf_dict[m] for m in common_ids])
        ct_arr = np.array([ct_dict[m] for m in common_ids])

        hybrid = self.alpha * _normalize(cf_arr) + (1.0 - self.alpha) * _normalize(ct_arr)
        order  = np.argsort(hybrid)[::-1][:n]
        return [(common_ids[idx], float(hybrid[idx])) for idx in order]
