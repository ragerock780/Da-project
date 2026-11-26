"""STAGE 5B: EVALUATION METRICS MODULE
===================================
Pipeline Stage: Stage 5 / 7 (Evaluation)
File: src/evaluation/metrics.py

What this file does:
    Evaluates all trained recommendation models across four standard industry metrics:
    1. RMSE (Root Mean Squared Error) on test ratings.
    2. Precision@10 on relevant test items (rating >= 4.0).
    3. Recall@10 on relevant test items (rating >= 4.0).
    4. Catalog Coverage (%) across all users' Top-10 recommendations.

Inputs:
    - models: dict[str, Any] of fitted recommendation models
    - R_train: Dense NumPy training matrix (num_users x num_items)
    - test_df: pd.DataFrame of held-out test ratings
    - num_items: Total items in catalog (1682)

Outputs:
    - results_df: pd.DataFrame with 3 rows (Popularity, ItemKNN, SVD) and 4 metric columns
    - all_top10: dict[str, dict[int, list[int]]] of Top-10 recommendations per model

Why this stage exists:
    A single metric never tells the full story: RMSE measures rating prediction accuracy,
    Precision@10 and Recall@10 measure top-of-funnel ranking quality, and Catalog Coverage
    measures whether an algorithm personalizes across the long tail or gets trapped in popularity bias.
"""

from typing import Any
import numpy as np
import pandas as pd

from src.evaluation.ranking import compute_top10_recommendations


def evaluate_models(
    models: dict[str, Any],
    R_train: np.ndarray,
    test_df: pd.DataFrame,
    num_items: int = 1682,
) -> tuple[pd.DataFrame, dict[str, dict[int, list[int]]]]:
    """# [EVALUATION] Benchmark models across RMSE, Precision@10, Recall@10, and Catalog Coverage.

    Formulas:
        # 1. RMSE = sqrt(mean((actual_rating - predicted_rating)^2)) over test pairs
        # 2. Relevant Item: test rating >= 4.0 stars
        # 3. Precision@10 = hits / 10
        # 4. Recall@10 = hits / (total relevant items in user's test set)
        # 5. Catalog Coverage (%) = (count of unique items recommended across all users) / 1682 * 100

    Args:
        models: Dictionary mapping model names to model instances with predict_matrix().
        R_train: Dense training matrix (num_users x num_items).
        test_df: Held-out test ratings DataFrame.
        num_items: Total catalog items (1682).

    Returns:
        tuple[pd.DataFrame, dict]: (results_df, all_top10_dict).
    """
    num_users = R_train.shape[0]

    # Pre-extract test arrays for fast vectorized NumPy indexing
    test_u = test_df["user_id"].to_numpy()
    test_i = test_df["item_id"].to_numpy()
    test_r = test_df["rating"].to_numpy()

    # Define relevant items: in MovieLens, ratings of 4.0 and 5.0 denote clear positive user preference
    relevant_df = test_df[test_df["rating"] >= 4.0]
    relevant_by_user: dict[int, set[int]] = (
        relevant_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    # Eligible users for ranking metrics: users who have AT LEAST 1 relevant item in their test set
    # [EVALUATION] If a user has 0 relevant test items, Recall is mathematically undefined (0/0).
    # We restrict ranking averages strictly to the same cohort of 907 users for both metrics.
    eval_users = [u for u in range(num_users) if len(relevant_by_user.get(u, set())) > 0]
    print(f"Users with at least 1 relevant test item (rating >= 4.0): {len(eval_users)} / {num_users}")

    all_top10: dict[str, dict[int, list[int]]] = {}
    rows: list[dict[str, Any]] = []

    for name, model in models.items():
        # Retrieve the complete predicted rating matrix (num_users x num_items)
        preds = model.predict_matrix()
        assert preds.shape == (num_users, num_items), f"Unexpected shape {preds.shape} for {name}"

        # -----------------------------------------------------------
        # Metric 1: RMSE (Lower is Better)
        # -----------------------------------------------------------
        pred_test_ratings = preds[test_u, test_i]
        rmse = float(np.sqrt(np.mean((test_r - pred_test_ratings) ** 2)))

        # -----------------------------------------------------------
        # Top-10 Recommendations Generation (Train items masked)
        # -----------------------------------------------------------
        top10 = compute_top10_recommendations(preds, R_train)
        all_top10[name] = top10

        # -----------------------------------------------------------
        # Metric 2 & 3: Precision@10 and Recall@10 (Higher is Better)
        # -----------------------------------------------------------
        p_list: list[float] = []
        r_list: list[float] = []
        for u in eval_users:
            rec_items = set(top10[u])
            rel_items = relevant_by_user[u]
            hits = len(rec_items.intersection(rel_items))

            # Precision@10: fraction of the 10 recommended items that the user liked
            p_list.append(hits / 10.0)

            # Recall@10: fraction of the user's liked test items captured in the Top-10
            r_list.append(hits / len(rel_items))

        precision_10 = float(np.mean(p_list))
        recall_10 = float(np.mean(r_list))

        # -----------------------------------------------------------
        # Metric 4: Catalog Coverage % (Higher is Better)
        # -----------------------------------------------------------
        # Measures what percentage of the catalog gets recommended to at least one user
        unique_recs = set()
        for u in range(num_users):
            unique_recs.update(top10[u])
        coverage = float(len(unique_recs) / num_items * 100.0)

        # Sanity check: coverage must be a valid percentage between 0% and 100%
        assert 0.0 <= coverage <= 100.0, f"Coverage percentage out of bounds: {coverage}"

        rows.append(
            {
                "Model": name,
                "RMSE": rmse,
                "Precision@10": precision_10,
                "Recall@10": recall_10,
                "Coverage (%)": coverage,
            }
        )

    # Format into a clean benchmark table DataFrame
    results_df = pd.DataFrame(rows).set_index("Model")

    # Sanity check: must have exactly 3 rows (Popularity, ItemKNN, SVD) and 4 metric columns
    assert results_df.shape == (len(models), 4), (
        f"Expected shape ({len(models)}, 4), got {results_df.shape}"
    )

    return results_df, all_top10
