"""STAGE 5A: TOP-10 RANKING & TRAIN-ITEM MASKING
=============================================
Pipeline Stage: Stage 5 / 7 (Evaluation)
File: src/evaluation/ranking.py

What this file does:
    Generates personalized Top-10 recommendation lists for each user from model
    score matrices. It masks all movies the user already rated in the training set
    with negative infinity (-inf) so they are never recommended, selects the top-10
    highest scoring unseen movies via np.argpartition, sorts them descending, and
    asserts zero training leakage.

Inputs:
    - pred_matrix: 2D NumPy array (num_users x num_items) of predicted scores
    - R_train: 2D NumPy array (num_users x num_items) of training ratings

Outputs:
    - top10_dict: dict[int, list[int]] mapping user_id -> list of 10 recommended item_ids

Why this stage exists:
    Recommending movies a user has already watched is useless in a real streaming service.
    Masking training items guarantees that recommendation evaluation reflects true discovery
    of new, previously unseen content.
"""

import numpy as np


def compute_top10_recommendations(
    pred_matrix: np.ndarray,
    R_train: np.ndarray,
) -> dict[int, list[int]]:
    """# [EVALUATION] Compute Top-10 recommended movie IDs for each user with train items masked.

    Algorithm:
        1. Copy the predicted score matrix.
        2. Set scores of all items in that user's TRAIN set to -inf (masking).
        3. Use np.argpartition to efficiently identify top 10 candidates in O(N) time.
        4. Sort the 10 candidates in descending order of predicted score.
        5. Verify that no training item ever leaks into the Top-10 list.

    Args:
        pred_matrix: Predicted rating/score matrix of shape (num_users, num_items).
        R_train: Dense training matrix of shape (num_users, num_items).

    Returns:
        dict[int, list[int]]: Mapping from user_id to an ordered list of 10 item_ids.
    """
    num_users, num_items = pred_matrix.shape
    scores = pred_matrix.copy()

    # Mask movies the user already rated in training so we only recommend NEW movies
    # [EVALUATION] Setting to -inf ensures these items can never compete for Top-10 slots
    scores[R_train > 0] = -np.inf

    # np.argpartition finds the top-10 highest scores along each row in linear time O(N)
    cand_idx = np.argpartition(scores, -10, axis=1)[:, -10:]

    top10_dict: dict[int, list[int]] = {}
    for u in range(num_users):
        u_cand = cand_idx[u]
        u_scores = scores[u, u_cand]

        # Sort the top 10 candidates in descending order of predicted score
        sorted_cand = u_cand[np.argsort(-u_scores)].tolist()
        top10_dict[u] = sorted_cand

        # Sanity assertion: no train item may ever appear in Top-10
        # Protects against data leakage and invalid recommendations of already-watched movies
        train_items = np.where(R_train[u] > 0)[0]
        overlap = set(sorted_cand).intersection(set(train_items))
        assert len(overlap) == 0, f"Critical Leakage: User {u} has train items in Top-10: {overlap}"

        # Sanity assertion: each Top-10 list must contain exactly 10 distinct items
        assert len(set(sorted_cand)) == 10, f"User {u} does not have 10 distinct recommendations"

    return top10_dict
