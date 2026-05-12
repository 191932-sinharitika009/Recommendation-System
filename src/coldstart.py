# Cold-Start Strategies (Phase 10)

import numpy as np
import pandas as pd


def build_popularity(train_df, movies_df) -> pd.DataFrame:
    """Return movies sorted by number of ratings in train (most popular first)."""
    counts = (train_df.groupby('movieId')['rating']
              .count()
              .reset_index()
              .rename(columns={'rating': 'n_ratings'}))
    return (movies_df[['movieId', 'title', 'genres']]
            .merge(counts, on='movieId')
            .sort_values('n_ratings', ascending=False)
            .reset_index(drop=True))


class ColdStartRecommender:
    """Three-tier cold-start strategy based on number of training ratings.

    Tier 1 — Cold   (< cold_thresh ratings) : globally popular items
    Tier 2 — Warm   (< warm_thresh ratings) : content-based (TF-IDF)
    Tier 3 — Active (>= warm_thresh ratings): collaborative filtering
    """

    def __init__(self, cf_model, content_model, popularity_df,
                 cold_thresh: int = 20, warm_thresh: int = 50):
        self.cf_model      = cf_model
        self.content_model = content_model
        self.popularity    = popularity_df   # output of build_popularity()
        self.cold_thresh   = cold_thresh
        self.warm_thresh   = warm_thresh

    def recommend(self, user_id: int, train_df, n: int = 10):
        """Return [(movieId, score), ...] using the appropriate tier."""
        n_ratings = int((train_df['userId'] == user_id).sum())

        if n_ratings < self.cold_thresh:
            # Tier 1: no CF signal — return globally popular unseen items
            seen  = set(train_df.loc[train_df['userId'] == user_id, 'movieId'])
            recs  = [mid for mid in self.popularity['movieId'] if mid not in seen]
            return [(int(mid), float(n - i)) for i, mid in enumerate(recs[:n])]

        if n_ratings < self.warm_thresh:
            # Tier 2: sparse CF signal — use content model
            return self.content_model.recommend(user_id, train_df, n=n)

        # Tier 3: enough CF history
        return self.cf_model.recommend(user_id, n=n)

    def tier(self, user_id: int, train_df) -> str:
        n = int((train_df['userId'] == user_id).sum())
        if n < self.cold_thresh:
            return 'cold'
        if n < self.warm_thresh:
            return 'warm'
        return 'active'
