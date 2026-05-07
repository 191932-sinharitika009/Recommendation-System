# Collaborative Filtering: memory-based (Phase 2) and model-based (Phase 3)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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


# ---------------------------------------------------------------------------
# Phase 3: Model-Based CF — Matrix Factorization (PyTorch)
# ---------------------------------------------------------------------------

class _RatingsDataset(Dataset):
    def __init__(self, df):
        self.users   = torch.tensor(df["userId"].values - 1,  dtype=torch.long)
        self.items   = torch.tensor(df["movieId"].values - 1, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values,      dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class _MFModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors, global_mean=0.0):
        super().__init__()
        self.global_mean = global_mean
        self.user_emb  = nn.Embedding(n_users, n_factors)
        self.item_emb  = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        # Small init so dot products start near zero; biases learn the mean offset
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        dot  = (self.user_emb(user) * self.item_emb(item)).sum(dim=1)
        bias = self.user_bias(user).squeeze(1) + self.item_bias(item).squeeze(1)
        return self.global_mean + dot + bias


class MatrixFactorization:
    """Model-based CF: SGD matrix factorization with biases via PyTorch."""

    def __init__(self, n_factors=64, n_epochs=20, lr=0.005, reg=0.1,
                 batch_size=2048, device=None):
        self.n_factors  = n_factors
        self.n_epochs   = n_epochs
        self.lr         = lr
        self.reg        = reg
        self.batch_size = batch_size
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = None
        self.n_users    = None
        self.n_items    = None
        self.user_rated = {}
        self.history    = {"train_loss": [], "val_loss": []}

    def fit(self, train_df, val_df=None):
        self.n_users = train_df["userId"].max()
        self.n_items = train_df["movieId"].max()
        global_mean  = float(train_df["rating"].mean())

        self.model = _MFModel(self.n_users, self.n_items, self.n_factors,
                              global_mean).to(self.device)
        # weight_decay=0: L2 reg applied manually in loss so it only hits embeddings
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                      weight_decay=0.0)
        criterion  = nn.MSELoss()

        train_loader = DataLoader(_RatingsDataset(train_df),
                                  batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(_RatingsDataset(val_df),
                                  batch_size=self.batch_size) if val_df is not None else None

        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            total_loss, n = 0.0, 0
            for users, items, ratings in train_loader:
                users, items, ratings = (users.to(self.device), items.to(self.device),
                                         ratings.to(self.device))
                optimizer.zero_grad()
                preds = self.model(users, items)
                mse   = criterion(preds, ratings)
                # Explicit L2 reg on embeddings only (not biases)
                l2 = (self.model.user_emb(users).pow(2).mean() +
                      self.model.item_emb(items).pow(2).mean())
                loss = mse + self.reg * l2
                loss.backward()
                optimizer.step()
                total_loss += mse.item() * len(ratings)
                n += len(ratings)
            train_rmse = (total_loss / n) ** 0.5
            self.history["train_loss"].append(train_rmse)

            if val_loader is not None:
                val_rmse = self._eval_rmse(val_loader)
                self.history["val_loss"].append(val_rmse)
                print(f"Epoch {epoch:02d}/{self.n_epochs}  "
                      f"train RMSE={train_rmse:.4f}  val RMSE={val_rmse:.4f}")
            else:
                print(f"Epoch {epoch:02d}/{self.n_epochs}  train RMSE={train_rmse:.4f}")

        for uid, grp in train_df.groupby("userId"):
            self.user_rated[uid - 1] = set(grp["movieId"].values - 1)

        return self

    def _eval_rmse(self, loader):
        self.model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for users, items, ratings in loader:
                users, items, ratings = (users.to(self.device), items.to(self.device),
                                         ratings.to(self.device))
                preds = self.model(users, items)
                total_loss += nn.MSELoss()(preds, ratings).item() * len(ratings)
                n += len(ratings)
        return (total_loss / n) ** 0.5

    def recommend(self, user_id, n=10):
        u = user_id - 1
        seen = self.user_rated.get(u, set())
        all_items   = torch.arange(self.n_items, device=self.device)
        user_tensor = torch.full((self.n_items,), u, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(user_tensor, all_items).cpu().numpy()

        for i in seen:
            scores[i] = -np.inf

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i + 1), float(scores[i])) for i in top_idx]
