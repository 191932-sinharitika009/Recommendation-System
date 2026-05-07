# Content-Based Recommender (Phase 4)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def _build_corpus(movies_df):
    """Combine title + genres into one text document per movie."""
    genres_clean = movies_df["genres"].str.replace("|", " ", regex=False)
    return (genres_clean + " " + movies_df["title"]).tolist()


class TFIDFRecommender:
    """Content-based recommender using TF-IDF on title + genres."""

    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vectorizer   = None
        self.item_vectors = None   # (n_items, max_features) sparse
        self.movie_ids    = None   # array of movieId values (1-indexed)
        self.user_rated   = {}

    def fit(self, movies_df, train_df):
        self.movie_ids = movies_df["movieId"].values
        corpus = _build_corpus(movies_df)

        self.vectorizer   = TfidfVectorizer(max_features=self.max_features)
        self.item_vectors = self.vectorizer.fit_transform(corpus)  # sparse

        for uid, grp in train_df.groupby("userId"):
            self.user_rated[uid] = set(grp["movieId"].values)

        return self

    def _user_profile(self, user_id, train_df):
        """Weighted mean of TF-IDF vectors for items the user rated."""
        user_rows = train_df[train_df["userId"] == user_id]
        if user_rows.empty:
            return None

        # Map movieId to row index in item_vectors
        mid_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        idxs    = [mid_to_idx[m] for m in user_rows["movieId"] if m in mid_to_idx]
        ratings = user_rows.set_index("movieId").loc[
            [m for m in user_rows["movieId"] if m in mid_to_idx], "rating"
        ].values.astype(np.float32)

        if len(idxs) == 0:
            return None

        weights = ratings - 2.5   # centre around 0
        vecs    = self.item_vectors[idxs].toarray()
        profile = np.average(vecs, axis=0, weights=weights + 1e-8)
        norm    = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def recommend(self, user_id, train_df, n=10):
        profile = self._user_profile(user_id, train_df)
        if profile is None:
            return []

        scores = cosine_similarity(profile.reshape(1, -1),
                                   self.item_vectors).flatten()

        seen = self.user_rated.get(user_id, set())
        mid_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        for mid in seen:
            if mid in mid_to_idx:
                scores[mid_to_idx[mid]] = -1.0

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(self.movie_ids[i]), float(scores[i])) for i in top_idx]


class SentenceTransformerRecommender:
    """Content-based recommender using dense sentence embeddings."""

    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=64):
        self.model_name  = model_name
        self.batch_size  = batch_size
        self.model       = None
        self.item_embs   = None   # (n_items, 384) float32 numpy array
        self.movie_ids   = None
        self.user_rated  = {}

    def fit(self, movies_df, train_df):
        self.movie_ids = movies_df["movieId"].values
        corpus = _build_corpus(movies_df)

        print(f"Encoding {len(corpus)} movies with {self.model_name}...")
        self.model     = SentenceTransformer(self.model_name)
        embs           = self.model.encode(corpus, batch_size=self.batch_size,
                                           show_progress_bar=True,
                                           convert_to_numpy=True)
        # L2-normalize for cosine similarity via dot product
        norms          = np.linalg.norm(embs, axis=1, keepdims=True)
        self.item_embs = (embs / np.where(norms == 0, 1, norms)).astype(np.float32)

        for uid, grp in train_df.groupby("userId"):
            self.user_rated[uid] = set(grp["movieId"].values)

        return self

    def _user_profile(self, user_id, train_df):
        """Weighted mean of item embeddings, normalized."""
        user_rows = train_df[train_df["userId"] == user_id]
        if user_rows.empty:
            return None

        mid_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        idxs    = [mid_to_idx[m] for m in user_rows["movieId"] if m in mid_to_idx]
        ratings = user_rows.set_index("movieId").loc[
            [m for m in user_rows["movieId"] if m in mid_to_idx], "rating"
        ].values.astype(np.float32)

        if len(idxs) == 0:
            return None

        weights = ratings - 2.5
        vecs    = self.item_embs[idxs]
        profile = np.average(vecs, axis=0, weights=weights + 1e-8)
        norm    = np.linalg.norm(profile)
        return (profile / norm).astype(np.float32) if norm > 0 else profile

    def recommend(self, user_id, train_df, n=10):
        profile = self._user_profile(user_id, train_df)
        if profile is None:
            return []

        # Dot product = cosine similarity (both sides normalized)
        scores = self.item_embs @ profile

        seen = self.user_rated.get(user_id, set())
        mid_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        for mid in seen:
            if mid in mid_to_idx:
                scores[mid_to_idx[mid]] = -1.0

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(self.movie_ids[i]), float(scores[i])) for i in top_idx]
