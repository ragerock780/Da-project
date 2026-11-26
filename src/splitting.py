"""STAGE 3: TEMPORAL TRAIN/TEST SPLIT & MATRIX GENERATION
=========================================================
Pipeline Stage: Stage 3 / 7
File: src/splitting.py

What this file does:
    Applies a realistic chronological (temporal) train/test split individually for
    each user (first 80% of ratings in time -> train, remaining 20% -> test).
    Then, it constructs the dense user-item rating matrix R_train (943 x 1682),
    where missing interactions are encoded as 0.0, and verifies matrix sparsity.

Inputs:
    - df: Cleaned ratings DataFrame ['user_id', 'item_id', 'rating', 'timestamp']

Outputs:
    - train_df: pd.DataFrame (first floor(0.8 * n) ratings per user)
    - test_df: pd.DataFrame (remaining ratings per user)
    - R_train: np.ndarray of shape (943, 1682) containing training ratings (0.0 = unrated)

Why this stage exists:
    Random splits cause 'data leakage from the future', where models learn from movies
    a user watched tomorrow to predict what they watched yesterday. A temporal per-user
    split accurately mimics real-world production systems by evaluating future predictions
    based strictly on past interaction history.
"""

import numpy as np
import pandas as pd

NUM_USERS = 943
NUM_ITEMS = 1682
NUM_RATINGS = 100000


# ---------------------------------------------------------------
# SECTION: Per-User Chronological Train/Test Split
# ---------------------------------------------------------------
def temporal_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """# [PREPROCESSING] Split each user's history chronologically into 80% train and 20% test.

    Logic:
        1. Sort user interactions chronologically using a stable sort (mergesort).
        2. For a user with n ratings, allocate the first floor(0.8 * n) to train_df.
        3. Allocate the remaining ratings to test_df.

    Args:
        df: Input ratings DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df).
    """
    # Use stable sort (mergesort) to preserve identical-timestamp ordering deterministically
    df_sorted = df.sort_values(by=["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Compute rating sequence position (0, 1, ..., n-1) for each user
    df_sorted["user_rating_rank"] = df_sorted.groupby("user_id").cumcount()

    # Total ratings given by each user
    user_counts = df_sorted.groupby("user_id")["rating"].transform("count")

    # Cutoff threshold: first floor(0.8 * n) ratings belong to training
    df_sorted["train_threshold"] = np.floor(0.8 * user_counts).astype(int)

    # Separate train and test subsets
    is_train = df_sorted["user_rating_rank"] < df_sorted["train_threshold"]

    cols_to_drop = ["user_rating_rank", "train_threshold"]
    train_df = df_sorted[is_train].drop(columns=cols_to_drop).reset_index(drop=True)
    test_df = df_sorted[~is_train].drop(columns=cols_to_drop).reset_index(drop=True)

    # Sanity check: verify that no ratings were lost or duplicated
    # Protects against data leakage or dropped interaction records
    assert len(train_df) + len(test_df) == len(df), (
        f"Split count mismatch: {len(train_df)} + {len(test_df)} != {len(df)}"
    )

    # Verify temporal guarantee: for each user, max train timestamp <= min test timestamp
    # Protects against future data leakage into the training set
    u_train_max = train_df.groupby("user_id")["timestamp"].max()
    u_test_min = test_df.groupby("user_id")["timestamp"].min()
    common_users = u_train_max.index.intersection(u_test_min.index)
    assert (u_train_max.loc[common_users] <= u_test_min.loc[common_users]).all(), (
        "Temporal leak detected: a training timestamp is later than a test timestamp!"
    )

    return train_df, test_df


# ---------------------------------------------------------------
# SECTION: Dense Train Matrix Construction & Sparsity
# ---------------------------------------------------------------
def build_train_matrix(
    train_df: pd.DataFrame,
    num_users: int = NUM_USERS,
    num_items: int = NUM_ITEMS,
) -> np.ndarray:
    """# [MISSING VALUES] Build dense (num_users x num_items) rating matrix R_train.

    Missing values:
        Unrated user-item cells stay 0.0. In recommendation systems, 0.0 indicates
        an unobserved interaction (missing data), NOT a zero-star rating.
        We never impute or invent fake ratings.

    Args:
        train_df: Training ratings subset.
        num_users: Total user count (default 943).
        num_items: Total item count (default 1682).

    Returns:
        np.ndarray: 2D array of shape (num_users, num_items) with training ratings.
    """
    # Initialize empty matrix with zeros for missing entries
    R_train = np.zeros((num_users, num_items), dtype=float)

    # Place explicit ratings at their corresponding [user, movie] indices
    user_indices = train_df["user_id"].values
    item_indices = train_df["item_id"].values
    ratings = train_df["rating"].values

    R_train[user_indices, item_indices] = ratings

    # Sanity check: shape must be exactly (943, 1682)
    assert R_train.shape == (num_users, num_items), (
        f"Incorrect matrix shape {R_train.shape}, expected ({num_users}, {num_items})"
    )

    return R_train


def compute_sparsity(R_train: np.ndarray) -> float:
    """Calculate the percentage of unrated cells in the training matrix.

    Formula:
        sparsity (%) = (1 - (num_rated_cells / (num_users * num_items))) * 100

    Args:
        R_train: 2D NumPy array of ratings.

    Returns:
        float: Sparsity percentage (typically ~94.98% for MovieLens 100K train).
    """
    total_cells = R_train.shape[0] * R_train.shape[1]
    rated_cells = int(np.count_nonzero(R_train))
    sparsity_pct = (1.0 - (rated_cells / total_cells)) * 100.0
    return float(sparsity_pct)


def split_and_build_matrix(
    df: pd.DataFrame,
    num_users: int = NUM_USERS,
    num_items: int = NUM_ITEMS,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Convenience pipeline function executing both split and matrix construction.

    Args:
        df: Input ratings DataFrame.
        num_users: Total user count.
        num_items: Total item count.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: (train_df, test_df, R_train).
    """
    train_df, test_df = temporal_train_test_split(df)
    R_train = build_train_matrix(train_df, num_users, num_items)
    sparsity = compute_sparsity(R_train)

    # Print sanity check summary
    print("--- Temporal Train/Test Split Sanity Check ---")
    print(f"Total ratings: {len(df):,}")
    print(f"Train ratings: {len(train_df):,} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Test ratings:  {len(test_df):,} ({len(test_df)/len(df)*100:.2f}%)")
    print(f"Users: {num_users}, Items: {num_items}")
    print(f"Train matrix shape: {R_train.shape}")
    print(f"Train matrix sparsity: {sparsity:.2f}%")
    print("----------------------------------------------")

    return train_df, test_df, R_train
