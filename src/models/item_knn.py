"""STAGE 4B: ITEM-BASED COLLABORATIVE FILTERING (ITEM-KNN)
======================================================
Pipeline Stage: Stage 4 / 7 (Models)
File: src/models/item_knn.py

What this file does:
    Implements item-based collaborative filtering using cosine similarity between
    item rating vectors. For each movie, it retains only the top-30 most similar
    neighbors (zeroing out the rest) and computes a weighted average of the user's
    ratings on those neighbor movies.

Inputs:
    - R_train: Dense NumPy training matrix of shape (943, 1682) where 0 = missing

Outputs:
    - predict_matrix(): 2D NumPy array of shape (943, 1682) containing predicted scores

Why this stage exists:
    Item-based CF personalizes recommendations based on the principle that if a user
    liked movie A, and movie B has a very similar rating pattern to movie A across
    all users, the user will likely enjoy movie B as well.
"""

from typing import Self
import numpy as np


class ItemKNN:
    """# [ALGORITHM] Item-based Collaborative Filtering with cosine similarity and top-k neighbors."""

    def __init__(self, k: int = 30) -> None:
        """Initialize ItemKNN recommender.

        Args:
            k: Number of most similar neighbor items to keep per movie (default 30).
        """
        self.k = k
        self.score_matrix: np.ndarray | None = None

    def fit(self, R_train: np.ndarray) -> Self:
        """Fit the ItemKNN model on R_train using vectorized cosine similarity.

        Formula:
            # Prediction for user u on item i:
            # pred(u, i) = sum_j (sim[i, j] * R[u, j]) / sum_j (|sim[i, j]| * 1[R[u, j] > 0])
            # Vectorized implementation:
            # num = R @ S.T
            # den = (R > 0).astype(float) @ |S|.T

        Args:
            R_train: Dense training matrix (num_users x num_items).

        Returns:
            ItemKNN: The fitted model instance.
        """
        num_users, num_items = R_train.shape

        # Step 1: Compute pairwise cosine similarity between item column vectors
        # Each column of R_train represents one movie's rating vector across all 943 users
        item_norms = np.linalg.norm(R_train, axis=0)  # shape: (1682,)
        norm_outer = np.outer(item_norms, item_norms)
        safe_norm = np.where(norm_outer > 0, norm_outer, 1.0)
        sim = (R_train.T @ R_train) / safe_norm
        # Zero out similarity for items that received zero training ratings
        sim[norm_outer == 0] = 0.0

        # Step 2: Set diagonal to 0 so an item never considers itself as its own neighbor
        np.fill_diagonal(sim, 0.0)

        # Step 3: Keep only the top-k (30) most similar neighbors per item (zero out the rest)
        # S[i, j] is the similarity between target movie i and neighbor movie j
        S = np.zeros_like(sim)
        k = min(self.k, num_items - 1)
        if k > 0:
            # np.argpartition finds the indices of the largest k elements along each row
            top_k_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
            row_idx = np.arange(num_items)[:, None]
            S[row_idx, top_k_idx] = sim[row_idx, top_k_idx]

        # Step 4: Fully vectorized prediction numerator and denominator
        # num = R @ S.T accumulates neighbor ratings weighted by similarity
        # den = (R > 0).astype(float) @ |S|.T sums the similarity weights of rated neighbors
        num = R_train @ S.T
        den = (R_train > 0).astype(float) @ np.abs(S).T

        # Step 5: Compute each user's mean rating over their RATED entries only
        # [MISSING VALUES] Unrated entries (zeros) are strictly excluded from the user mean
        user_sums = R_train.sum(axis=1)
        user_counts = (R_train > 0).sum(axis=1)
        user_means = np.zeros((num_users, 1), dtype=float)
        has_ratings = user_counts > 0
        user_means[has_ratings, 0] = user_sums[has_ratings] / user_counts[has_ratings]

        # Step 6: Fall back to user mean where den == 0 (no neighbor ratings available)
        # [MISSING VALUES] If a user hasn't rated any of the movie's 30 neighbors,
        # we predict the user's personal baseline mean rather than inventing arbitrary ratings
        user_means_2d = np.broadcast_to(user_means, num.shape).copy()
        pred = np.divide(num, den, out=user_means_2d, where=(den > 0))

        # Step 7: Clip all predicted scores strictly to the valid 1-5 rating scale
        pred = np.clip(pred, 1.0, 5.0)

        self.score_matrix = pred
        return self

    def predict_matrix(self) -> np.ndarray:
        """Return the predicted score matrix on the 1-5 scale.

        Returns:
            np.ndarray: Predicted scores of shape (num_users, num_items).
        """
        if self.score_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.score_matrix.copy()
