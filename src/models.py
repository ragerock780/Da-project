"""Recommendation algorithms for MovieLens 100K:
1. PopularityBaseline (Bayesian weighted rating, IMDB-style)
2. ItemKNN (Item-based collaborative filtering with top-30 neighbors)
3. SVDRecommender (Truncated SVD matrix factorization on mean-centered ratings)
"""

from typing import Self
import numpy as np
import scipy.sparse.linalg


class PopularityBaseline:
    """Bayesian weighted rating (IMDB-style) popularity baseline."""

    def __init__(self) -> None:
        self.wr_scores: np.ndarray | None = None
        self.score_matrix: np.ndarray | None = None

    def fit(self, R_train: np.ndarray) -> Self:
        """Compute Bayesian weighted ratings for all items and broadcast across all users.

        v = number of train ratings per movie
        R = mean train rating per movie
        C = global mean of train ratings
        m = 75th percentile of v
        WR = v/(v+m)*R + m/(v+m)*C
        """
        num_users, num_items = R_train.shape
        v = (R_train > 0).sum(axis=0)  # shape: (num_items,)

        rated_mask = v > 0
        R = np.zeros(num_items, dtype=float)
        R[rated_mask] = R_train.sum(axis=0)[rated_mask] / v[rated_mask]

        train_ratings = R_train[R_train > 0]
        C = float(train_ratings.mean())
        m = float(np.percentile(v, 75))

        denom = v + m
        # Movies with no ratings get C
        wr = np.full(num_items, C, dtype=float)
        valid = denom > 0
        wr[valid] = (v[valid] / denom[valid]) * R[valid] + (m / denom[valid]) * C

        self.wr_scores = wr
        self.score_matrix = np.tile(wr, (num_users, 1))
        return self

    def predict_matrix(self) -> np.ndarray:
        """Return (num_users x num_items) predicted score matrix on 1-5 scale."""
        if self.score_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.score_matrix.copy()


class ItemKNN:
    """Item-based Collaborative Filtering with cosine similarity and top-30 neighbors."""

    def __init__(self, k: int = 30) -> None:
        self.k = k
        self.score_matrix: np.ndarray | None = None

    def fit(self, R_train: np.ndarray) -> Self:
        """Fit ItemKNN on dense train matrix R_train.

        - Cosine similarity between item columns on R_train (0 = missing).
        - Set diagonal to 0.
        - Keep top-30 most similar neighbors per item (zero out the rest).
        - Vectorized prediction: num = R @ S.T ; den = (R>0).astype(float) @ |S|.T
        - Fallback: where den == 0, use user's mean rating (rated entries only).
        - Clip predictions to [1, 5].
        """
        num_users, num_items = R_train.shape

        # Compute cosine similarity between item columns
        item_norms = np.linalg.norm(R_train, axis=0)  # (num_items,)
        norm_outer = np.outer(item_norms, item_norms)
        safe_norm = np.where(norm_outer > 0, norm_outer, 1.0)
        sim = (R_train.T @ R_train) / safe_norm
        sim[norm_outer == 0] = 0.0

        # Set diagonal to 0
        np.fill_diagonal(sim, 0.0)

        # Keep only top-k most similar neighbors per item
        # S[i, j] is similarity between item i and neighbor item j
        S = np.zeros_like(sim)
        k = min(self.k, num_items - 1)
        if k > 0:
            top_k_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
            row_idx = np.arange(num_items)[:, None]
            S[row_idx, top_k_idx] = sim[row_idx, top_k_idx]

        # Fully vectorized prediction
        num = R_train @ S.T
        den = (R_train > 0).astype(float) @ np.abs(S).T

        # User mean rating over rated entries only
        user_sums = R_train.sum(axis=1)
        user_counts = (R_train > 0).sum(axis=1)
        user_means = np.zeros((num_users, 1), dtype=float)
        has_ratings = user_counts > 0
        user_means[has_ratings, 0] = user_sums[has_ratings] / user_counts[has_ratings]

        # Fallback to user mean where den == 0, then clip to [1, 5]
        user_means_2d = np.broadcast_to(user_means, num.shape).copy()
        pred = np.divide(num, den, out=user_means_2d, where=(den > 0))
        pred = np.clip(pred, 1.0, 5.0)

        self.score_matrix = pred
        return self

    def predict_matrix(self) -> np.ndarray:
        """Return (num_users x num_items) predicted score matrix on 1-5 scale."""
        if self.score_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.score_matrix.copy()


class SVDRecommender:
    """Matrix factorization using scipy.sparse.linalg.svds on mean-centered ratings."""

    def __init__(self, k: int = 20, random_state: int = 42) -> None:
        self.k = k
        self.random_state = random_state
        self.score_matrix: np.ndarray | None = None

    def fit(self, R_train: np.ndarray) -> Self:
        """Fit SVD recommender.

        - Compute each user's mean over rated entries only.
        - Build centered matrix: rated entries minus user mean, missing entries = 0.
        - Run svds(centered, k=20).
        - Reconstruct U @ diag(sigma) @ Vt + user_mean.
        - Clip to [1, 5].
        """
        num_users, num_items = R_train.shape

        user_sums = R_train.sum(axis=1)
        user_counts = (R_train > 0).sum(axis=1)
        user_means = np.zeros(num_users, dtype=float)
        has_ratings = user_counts > 0
        user_means[has_ratings] = user_sums[has_ratings] / user_counts[has_ratings]

        # Centered matrix: rated entries minus user mean, missing entries = 0
        centered = np.where(R_train > 0, R_train - user_means[:, None], 0.0)

        # Truncated SVD
        n_factors = min(self.k, min(num_users, num_items) - 1)
        u, s, vt = scipy.sparse.linalg.svds(
            centered, k=n_factors, random_state=self.random_state
        )

        # Reconstruct and add back user mean
        recon = (u * s) @ vt + user_means[:, None]

        # Clip to [1, 5]
        self.score_matrix = np.clip(recon, 1.0, 5.0)
        return self

    def predict_matrix(self) -> np.ndarray:
        """Return (num_users x num_items) predicted score matrix on 1-5 scale."""
        if self.score_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.score_matrix.copy()
