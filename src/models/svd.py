"""STAGE 4C: MATRIX FACTORIZATION (TRUNCATED SVD)
==============================================
Pipeline Stage: Stage 4 / 7 (Models)
File: src/models/svd.py

What this file does:
    Implements matrix factorization using Truncated Singular Value Decomposition (SVD).
    It centers each user's ratings by subtracting their personal mean rating, decomposes
    the centered sparse matrix into k=20 latent factors using scipy.sparse.linalg.svds,
    reconstructs the full rating matrix, adds user means back, and clips predictions to [1, 5].

Inputs:
    - R_train: Dense NumPy training matrix of shape (943, 1682) where 0 = missing

Outputs:
    - predict_matrix(): 2D NumPy array of shape (943, 1682) containing predicted scores

Why this stage exists:
    Users have different rating biases (some give harsh 2-star reviews while others
    give generous 5-star reviews). Centering removes individual bias, and low-rank SVD
    uncovers hidden latent taste themes (e.g., sci-fi action vs. romantic drama)
    that connect users and movies beyond raw surface-level title matching.
"""

from typing import Self
import numpy as np
import scipy.sparse.linalg


class SVDRecommender:
    """# [ALGORITHM] Matrix factorization via Truncated SVD on user-mean-centered ratings."""

    def __init__(self, k: int = 20, random_state: int = 42) -> None:
        """Initialize SVD recommender.

        Args:
            k: Number of latent concepts / factors to extract (default 20).
            random_state: Random seed for deterministic SVD decomposition.
        """
        self.k = k
        self.random_state = random_state
        self.score_matrix: np.ndarray | None = None

    def fit(self, R_train: np.ndarray) -> Self:
        """Fit the SVD recommender by decomposing the mean-centered training matrix.

        Mathematical Steps:
            # 1. User mean: mu_u = (sum of ratings by user u) / (number of ratings by user u)
            # 2. Centered matrix: A[u, i] = R[u, i] - mu_u if rated, else 0.0
            # 3. Truncated SVD: A ~ U @ diag(sigma) @ Vt  (k=20 factors)
            # 4. Reconstruction: R_pred = U @ diag(sigma) @ Vt + mu_u
            # 5. Clip predictions to [1.0, 5.0]

        Args:
            R_train: Dense training matrix (num_users x num_items).

        Returns:
            SVDRecommender: The fitted model instance.
        """
        num_users, num_items = R_train.shape

        # Step 1: Compute each user's average rating over their RATED entries only
        # [MISSING VALUES] Missing entries (0.0) are completely excluded from the mean
        user_sums = R_train.sum(axis=1)
        user_counts = (R_train > 0).sum(axis=1)
        user_means = np.zeros(num_users, dtype=float)
        has_ratings = user_counts > 0
        user_means[has_ratings] = user_sums[has_ratings] / user_counts[has_ratings]

        # Step 2: Build mean-centered rating matrix
        # [MISSING VALUES] Where a user rated a movie: (rating - user_mean)
        # Where a cell was missing (0.0): it remains 0.0 (neutral unobserved interaction)
        centered = np.where(R_train > 0, R_train - user_means[:, None], 0.0)

        # Step 3: Compute Truncated SVD with k=20 latent factors
        # scipy.sparse.linalg.svds extracts the top-k largest singular values/vectors
        n_factors = min(self.k, min(num_users, num_items) - 1)
        u, s, vt = scipy.sparse.linalg.svds(
            centered, k=n_factors, random_state=self.random_state
        )

        # Step 4: Reconstruct the full predicted matrix and restore each user's personal mean
        # (u * s) @ vt is equivalent to u @ np.diag(s) @ vt
        recon = (u * s) @ vt + user_means[:, None]

        # Step 5: Clip all predictions to the valid MovieLens [1.0, 5.0] rating scale
        self.score_matrix = np.clip(recon, 1.0, 5.0)

        return self

    def predict_matrix(self) -> np.ndarray:
        """Return the predicted score matrix on the 1-5 scale.

        Returns:
            np.ndarray: Predicted scores of shape (num_users, num_items).
        """
        if self.score_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.score_matrix.copy()
