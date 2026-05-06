
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_ratings(path=None):
    p = path or DATA_DIR / "ml-1m" / "ratings.dat"
    return pd.read_csv(p, sep="::", engine="python",
                       names=["userId", "movieId", "rating", "timestamp"])


def load_movies(path=None):
    p = path or DATA_DIR / "ml-1m" / "movies.dat"
    return pd.read_csv(p, sep="::", engine="python", encoding="latin-1",
                       names=["movieId", "title", "genres"])


def load_users(path=None):
    p = path or DATA_DIR / "ml-1m" / "users.dat"
    return pd.read_csv(p, sep="::", engine="python",
                       names=["userId", "gender", "age", "occupation", "zip"])


def split_per_user(df, test_ratio=0.1, val_ratio=0.1):
    """Per-user temporal split: hold out last interactions by timestamp."""
    df = df.sort_values(["userId", "timestamp"])
    train, val, test = [], [], []
    for uid, group in df.groupby("userId"):
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test.append(group.iloc[-n_test:])
        val.append(group.iloc[-(n_test + n_val):-n_test])
        train.append(group.iloc[:-(n_test + n_val)])
    return pd.concat(train), pd.concat(val), pd.concat(test)


def build_interaction_matrix(df, n_users=None, n_items=None):
    """Build sparse user-item interaction matrix from ratings dataframe."""
    users = df["userId"].values - 1
    items = df["movieId"].values - 1
    ratings = df["rating"].values.astype(np.float32)
    n_users = n_users or df["userId"].max()
    n_items = n_items or df["movieId"].max()
    return sp.csr_matrix((ratings, (users, items)), shape=(n_users, n_items))


def compute_sparsity(matrix):
    nnz = matrix.nnz
    total = matrix.shape[0] * matrix.shape[1]
    return 1 - nnz / total
