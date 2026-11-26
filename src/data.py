"""BACKWARD COMPATIBILITY ADAPTER FOR DATA MODULE
=================================================
File: src/data.py

Re-exports data loading, validation, and splitting functions from:
- src.data_loading
- src.preprocessing
- src.splitting
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loading import ensure_movielens_data, load_movies, load_ratings, load_raw_data
from src.preprocessing import clean, validate
from src.splitting import build_train_matrix, compute_sparsity, split_and_build_matrix, temporal_train_test_split


def load_and_split_data(
    data_dir: Path | str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Compatibility wrapper that loads raw data, validates it, and performs temporal split."""
    raw_df = load_ratings(data_dir)
    validate(raw_df)
    clean_df = clean(raw_df)
    train_df, test_df, R_train = split_and_build_matrix(clean_df)
    return train_df, test_df, R_train


__all__ = [
    "ensure_movielens_data",
    "load_movies",
    "load_ratings",
    "load_raw_data",
    "load_and_split_data",
    "temporal_train_test_split",
    "build_train_matrix",
    "compute_sparsity",
    "split_and_build_matrix",
]
