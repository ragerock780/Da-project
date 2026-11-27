"""STAGE 2: PREPROCESSING, VALIDATION & OUTLIER DETECTION
======================================================
Pipeline Stage: Stage 2 / 7
File: src/preprocessing.py

What this file does:
    Performs rigorous data quality checks on the loaded MovieLens 100K ratings.
    It validates data integrity (no nulls, valid rating ranges), detects user
    and item activity outliers (reporting only without deletion to prevent data loss),
    applies safe data cleaning, calculates a cryptographic data fingerprint (SHA-256)
    for change tracking, and saves a preprocessing audit report.

Inputs:
    - df: pd.DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']

Outputs:
    - Cleaned and verified pd.DataFrame (100,000 ratings intact)
    - Outlier report dictionary & formatted summary
    - SHA-256 fingerprint string for integrity verification

Why this stage exists:
    Real-world recommendation systems can fail silently if corrupted data enters
    the training pipeline; validating schemas and documenting outliers upfront ensures
    experiments are reproducible and reliable for examiners.
"""

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# SECTION: Schema & Integrity Validation
# ---------------------------------------------------------------
def validate(df: pd.DataFrame) -> bool:
    """# [PREPROCESSING] Validate structure, types, nulls, and value ranges of ratings dataset.

    Args:
        df: Input DataFrame containing ratings.

    Returns:
        bool: True if validation passes; raises AssertionError otherwise.
    """
    # Check 1: Required columns must all be present
    required_cols = {"user_id", "item_id", "rating", "timestamp"}
    missing_cols = required_cols - set(df.columns)
    assert not missing_cols, f"Missing required columns: {missing_cols}"

    # Check 2: No missing or null values allowed in raw ratings
    # [MISSING VALUES] MovieLens 100K has no NaN entries in ratings; all ratings are explicit
    null_counts = df[list(required_cols)].isnull().sum()
    assert null_counts.sum() == 0, f"Null values detected in dataset: {null_counts.to_dict()}"

    # Check 3: Ratings must be bounded between 1.0 and 5.0 stars
    min_rating = df["rating"].min()
    max_rating = df["rating"].max()
    assert min_rating >= 1.0 and max_rating <= 5.0, (
        f"Ratings out of bounds [1, 5]: min={min_rating}, max={max_rating}"
    )

    # Check 4: User and Item IDs must be non-negative integers
    assert (df["user_id"] >= 0).all(), "user_id contains negative values"
    assert (df["item_id"] >= 0).all(), "item_id contains negative values"

    # Check 5: Dynamic integrity check on dataset size
    assert len(df) > 0, "Ratings DataFrame is empty!"
    if len(df) != 100000:
        print(f"  [DATA MODIFICATION NOTICE] Row count is {len(df):,} (standard benchmark baseline is 100,000).")
    assert df["item_id"].max() < 1682, f"item_id exceeds max catalog size: {df['item_id'].max()}"

    return True


# ---------------------------------------------------------------
# SECTION: Outlier detection (report only, nothing is deleted)
# ---------------------------------------------------------------
def detect_outliers(df: pd.DataFrame) -> dict[str, Any]:
    """# [OUTLIERS] Detect activity outliers in users and items using the Interquartile Range (IQR).

    In recommender systems, extreme power users (rating 500+ movies) and blockbuster
    movies (rated 500+ times) are natural characteristics of human consumption, NOT data errors.
    We report them for transparency, but do NOT delete them to preserve benchmark validity.

    Args:
        df: Input ratings DataFrame.

    Returns:
        dict[str, Any]: Detailed statistics on user and item activity outliers.
    """
    # 1. User activity distribution (number of ratings per user)
    user_counts = df.groupby("user_id")["rating"].count()
    u_q1 = float(user_counts.quantile(0.25))
    u_q3 = float(user_counts.quantile(0.75))
    u_iqr = u_q3 - u_q1
    # Threshold for high-activity user outliers: Q3 + 1.5 * IQR
    u_upper_threshold = u_q3 + 1.5 * u_iqr
    user_outliers = user_counts[user_counts > u_upper_threshold]

    # 2. Movie popularity distribution (number of ratings per movie)
    movie_counts = df.groupby("item_id")["rating"].count()
    m_q1 = float(movie_counts.quantile(0.25))
    m_q3 = float(movie_counts.quantile(0.75))
    m_iqr = m_q3 - m_q1
    # Threshold for blockbuster movie outliers: Q3 + 1.5 * IQR
    m_upper_threshold = m_q3 + 1.5 * m_iqr
    movie_outliers = movie_counts[movie_counts > m_upper_threshold]

    # 3. Rating score distribution
    rating_counts = df["rating"].value_counts().to_dict()

    return {
        "user_activity": {
            "min": int(user_counts.min()),
            "median": float(user_counts.median()),
            "mean": float(user_counts.mean()),
            "max": int(user_counts.max()),
            "q1": u_q1,
            "q3": u_q3,
            "iqr": u_iqr,
            "upper_bound": u_upper_threshold,
            "outlier_count": len(user_outliers),
            "outlier_percentage": float(len(user_outliers) / len(user_counts) * 100.0),
        },
        "item_popularity": {
            "min": int(movie_counts.min()),
            "median": float(movie_counts.median()),
            "mean": float(movie_counts.mean()),
            "max": int(movie_counts.max()),
            "q1": m_q1,
            "q3": m_q3,
            "iqr": m_iqr,
            "upper_bound": m_upper_threshold,
            "outlier_count": len(movie_outliers),
            "outlier_percentage": float(len(movie_outliers) / len(movie_counts) * 100.0),
        },
        "ratings_breakdown": rating_counts,
    }


def outlier_summary(outlier_info: dict[str, Any]) -> str:
    """Format the outlier inspection results into a readable summary string for console logging.

    Args:
        outlier_info: Dictionary generated by detect_outliers().

    Returns:
        str: Human-readable multi-line summary.
    """
    u_info = outlier_info["user_activity"]
    m_info = outlier_info["item_popularity"]

    summary = (
        f"  - User Activity: Median = {u_info['median']:.0f} ratings, Max = {u_info['max']} ratings.\n"
        f"    Detected {u_info['outlier_count']} power users (> {u_info['upper_bound']:.1f} ratings, {u_info['outlier_percentage']:.1f}%).\n"
        f"  - Movie Popularity: Median = {m_info['median']:.0f} ratings, Max = {m_info['max']} ratings.\n"
        f"    Detected {m_info['outlier_count']} blockbuster movies (> {m_info['upper_bound']:.1f} ratings, {m_info['outlier_percentage']:.1f}%).\n"
        f"  - Action: Retained all rows (natural long-tail distribution preserved, 0 rows deleted)."
    )
    return summary


# ---------------------------------------------------------------
# SECTION: Data Cleaning (Safety Preserving)
# ---------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """# [PREPROCESSING] Clean and deduplicate ratings while guaranteeing zero data loss.

    By default, clean() removes any exact duplicates if they exist, casts IDs to integer,
    and asserts that all 100,000 ratings remain perfectly intact.

    Args:
        df: Input ratings DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    initial_rows = len(df)

    # Remove any duplicate user-item interaction pairs if present
    # MovieLens 100K has unique user-movie interactions
    cleaned_df = df.drop_duplicates(subset=["user_id", "item_id"]).copy()

    # Ensure integer and float datatypes are correctly cast
    cleaned_df["user_id"] = cleaned_df["user_id"].astype(int)
    cleaned_df["item_id"] = cleaned_df["item_id"].astype(int)
    cleaned_df["rating"] = cleaned_df["rating"].astype(float)
    cleaned_df["timestamp"] = cleaned_df["timestamp"].astype(int)

    # Protect against accidental row deletion: row count must remain 100,000
    assert len(cleaned_df) == initial_rows, (
        f"Data loss detected during cleaning: {initial_rows} -> {len(cleaned_df)}"
    )

    return cleaned_df


# ---------------------------------------------------------------
# SECTION: Cryptographic Data Change Tracking
# ---------------------------------------------------------------
def data_fingerprint(df: pd.DataFrame) -> str:
    """# [DATA CHANGE TRACKING] Compute deterministic SHA-256 hash of the ratings dataset.

    This fingerprint acts as a tamper-evident seal: if any rating, user ID, or
    timestamp is altered, this hash will change immediately.

    Args:
        df: Ratings DataFrame.

    Returns:
        str: Hexadecimal SHA-256 hash string.
    """
    # Sort deterministically by user, item, timestamp to ensure order-independent hashing
    sorted_df = df.sort_values(by=["user_id", "item_id", "timestamp"]).reset_index(drop=True)

    # Convert records to byte stream
    byte_stream = sorted_df.to_csv(index=False).encode("utf-8")
    sha256 = hashlib.sha256(byte_stream).hexdigest()
    return sha256


# ---------------------------------------------------------------
# SECTION: Audit Report Persistence
# ---------------------------------------------------------------
def save_reports(
    outlier_info: dict[str, Any],
    fingerprint: str,
    output_dir: Path | str = "results",
    total_ratings: int = 100000,
) -> Path:
    """Save preprocessing and data integrity audit reports to JSON for verification.

    Args:
        outlier_info: Outlier metrics dictionary.
        fingerprint: SHA-256 dataset hash.
        output_dir: Results directory.
        total_ratings: Current total rating count in the dataset.

    Returns:
        Path: Path to the generated report file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "data_preprocessing_report.json"

    report_payload = {
        "status": "VALIDATED_AND_CLEANED",
        "total_ratings": total_ratings,
        "is_standard_benchmark": (total_ratings == 100000),
        "unique_movies": 1682,
        "sha256_fingerprint": fingerprint,
        "outlier_analysis": outlier_info,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    return report_file
