"""STAGE 4A: POPULARITY BASELINE RECOMMENDER
===========================================
Pipeline Stage: Stage 4 / 7 (Models)
File: src/models/popularity.py

What this file does:
    Implements a non-personalized popularity baseline using Bayesian weighted
    ratings (the IMDB formula). It balances a movie's average rating with the
    number of ratings it received, pulling movies with few ratings toward the
    global catalog mean.

Inputs:
    - R_train: Dense NumPy training matrix of shape (943, 1682) where 0 = missing

Outputs:
    - predict_matrix(): 2D NumPy array of shape (943, 1682) containing predicted scores

Why this stage exists:
    A popularity baseline serves as the essential benchmark in recommendation systems.
    Any personalized algorithm (collaborative filtering or matrix factorization)
    must prove it delivers real value beyond simply recommending the most popular hits.
"""

from typing import Self
import numpy as np


class PopularityBaseline:
    """# [ALGORITHM] Bayesian weighted rating (IMDB-style) popularity baseline."""

    def __init__(self) -> None:
        """Initialize model attributes."""
        self.wr_scores: np.ndarray | None = None
        self.score_matrix: np.ndarray | None = None

    def fit(self, R_train: np.ndarray) -> Self:
        """Fit the popularity model on R_train.

        Mathematical Formula:
            # Weighted rating = v/(v+m)*R + m/(v+m)*C
            # v = votes for the movie (count of non-zero ratings in R_train)
            # m = 75th percentile of votes across all movies (confidence threshold)
            # R = movie's average rating in the training set
            # C = global mean rating across all training ratings

        Args:
            R_train: Dense training matrix (num_users x num_items).

        Returns:
            PopularityBaseline: The fitted model instance.
        """
        num_users, num_items = R_train.shape

        # Step 1: Count non-zero ratings (votes) per movie
        # Missing values: zeros in R_train are unrated and excluded from vote count
        # [MISSING VALUES]
        v = (R_train > 0).sum(axis=0)  # shape: (1682,)

        # Step 2: Compute average rating per movie (for movies with at least 1 vote)
        rated_mask = v > 0
        R = np.zeros(num_items, dtype=float)
        R[rated_mask] = R_train.sum(axis=0)[rated_mask] / v[rated_mask]

        # Step 3: Compute global mean rating C across all explicit ratings
        train_ratings = R_train[R_train > 0]
        C = float(train_ratings.mean())

        # Step 4: Compute minimum votes threshold m (75th percentile of v)
        m = float(np.percentile(v, 75))

        # Step 5: Compute Bayesian Weighted Rating for each movie
        # Movies with no ratings get C (global mean)
        denom = v + m
        wr = np.full(num_items, C, dtype=float)
        valid = denom > 0
        wr[valid] = (v[valid] / denom[valid]) * R[valid] + (m / denom[valid]) * C

        self.wr_scores = wr

        # Step 6: Construct score matrix by repeating the exact same WR vector for all users
        # This captures pure popularity bias: every user initially receives the identical ranking
        self.score_matrix = np.tile(wr, (num_users, 1))

        return self

    def predict_matrix(self) -> np.ndarray:
        """Return the predicted score matrix on the 1-5 scale.

        Returns:
            np.ndarray: Predicted scores of shape (num_users, num_items).
        """
        if self.score_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.score_matrix.copy()
