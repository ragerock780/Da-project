"""Evaluation module for recommendation algorithms:
- RMSE on test ratings
- Top-10 ranking metrics: Precision@10, Recall@10 (on relevant test items rating >= 4.0)
- Catalog Coverage (%)
"""

from typing import Any
import numpy as np
import pandas as pd


def compute_top10_recommendations(
    pred_matrix: np.ndarray,
    R_train: np.ndarray,
) -> dict[int, list[int]]:
    """Compute top-10 recommended item indices for each user.

    Items in that user's TRAIN set have their score set to -inf so they are never recommended.
    Uses np.argpartition followed by sorting descending by score.
    """
    num_users, num_items = pred_matrix.shape
    scores = pred_matrix.copy()

    # Mask train items with -inf
    scores[R_train > 0] = -np.inf

    # Candidate top 10 per user using argpartition
    cand_idx = np.argpartition(scores, -10, axis=1)[:, -10:]

    top10_dict: dict[int, list[int]] = {}
    for u in range(num_users):
        u_cand = cand_idx[u]
        u_scores = scores[u, u_cand]
        # Sort descending by score
        sorted_cand = u_cand[np.argsort(-u_scores)].tolist()
        top10_dict[u] = sorted_cand

        # Sanity assertion: no train item may ever appear in top-10
        train_items = np.where(R_train[u] > 0)[0]
        overlap = set(sorted_cand).intersection(set(train_items))
        assert len(overlap) == 0, f"User {u} has train items in Top-10: {overlap}"

    return top10_dict


def evaluate_models(
    models: dict[str, Any],
    R_train: np.ndarray,
    test_df: pd.DataFrame,
    num_items: int = 1682,
) -> tuple[pd.DataFrame, dict[str, dict[int, list[int]]]]:
    """Evaluate models on RMSE, Precision@10, Recall@10, and Catalog Coverage.

    Relevant item: test rating >= 4.0.
    Precision@10 = hits / 10
    Recall@10 = hits / (number of relevant items in the user's test set)
    Averaged strictly over users that have at least 1 relevant test item.
    Catalog Coverage = (# unique movies appearing in any user's Top-10) / num_items * 100.
    """
    num_users = R_train.shape[0]

    # Pre-extract test pairs
    test_u = test_df["user_id"].to_numpy()
    test_i = test_df["item_id"].to_numpy()
    test_r = test_df["rating"].to_numpy()

    # Relevant test items (rating >= 4.0) per user
    relevant_df = test_df[test_df["rating"] >= 4.0]
    relevant_by_user: dict[int, set[int]] = (
        relevant_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    # Eligible users for ranking evaluation: at least 1 relevant test item
    eval_users = [u for u in range(num_users) if len(relevant_by_user.get(u, set())) > 0]
    print(f"Users with at least 1 relevant test item (rating >= 4.0): {len(eval_users)} / {num_users}")

    all_top10: dict[str, dict[int, list[int]]] = {}
    rows: list[dict[str, Any]] = []

    for name, model in models.items():
        preds = model.predict_matrix()
        assert preds.shape == (num_users, num_items), f"Unexpected shape {preds.shape} for {name}"

        # 1. RMSE over all test (user, item) pairs
        pred_test_ratings = preds[test_u, test_i]
        rmse = float(np.sqrt(np.mean((test_r - pred_test_ratings) ** 2)))

        # 2. Top-10 recommendations (train items masked)
        top10 = compute_top10_recommendations(preds, R_train)
        all_top10[name] = top10

        # 3. Precision@10 and Recall@10 over eligible users
        p_list: list[float] = []
        r_list: list[float] = []
        for u in eval_users:
            rec_items = set(top10[u])
            rel_items = relevant_by_user[u]
            hits = len(rec_items.intersection(rel_items))
            p_list.append(hits / 10.0)
            r_list.append(hits / len(rel_items))

        precision_10 = float(np.mean(p_list))
        recall_10 = float(np.mean(r_list))

        # 4. Catalog Coverage (%)
        unique_recs = set()
        for u in range(num_users):
            unique_recs.update(top10[u])
        coverage = float(len(unique_recs) / num_items * 100.0)
        assert 0.0 <= coverage <= 100.0, f"Coverage out of range: {coverage}"

        rows.append(
            {
                "Model": name,
                "RMSE": rmse,
                "Precision@10": precision_10,
                "Recall@10": recall_10,
                "Coverage (%)": coverage,
            }
        )

    results_df = pd.DataFrame(rows).set_index("Model")
    # Verify exact shape: 3 rows, 4 metric columns
    assert results_df.shape == (len(models), 4)

    return results_df, all_top10
