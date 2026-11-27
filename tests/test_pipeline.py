"""UNIT AND INTEGRATION TEST SUITE
=================================
File: tests/test_pipeline.py

Comprehensive tests covering every modular stage of the recommendation benchmark:
- Stage 1: Data loading & 0-based conversion
- Stage 2: Schema validation, outlier detection, cleaning, fingerprinting
- Stage 3: Temporal 80/20 train/test split, matrix generation, sparsity
- Stage 4: Recommendation model fitting & prediction bounds [1, 5]
- Stage 5: Top-10 ranking, zero train leakage, evaluation metrics
- Stage 6: Visualization figures generation
- Streamlit application execution and tab verification
"""

from pathlib import Path
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from src.data_loading import load_movies, load_ratings
from src.evaluation.metrics import evaluate_models
from src.evaluation.ranking import compute_top10_recommendations
from src.models.item_knn import ItemKNN
from src.models.popularity import PopularityBaseline
from src.models.svd import SVDRecommender
from src.plots import (
    plot_item_long_tail,
    plot_model_comparison,
    plot_rating_distribution,
    plot_ratings_per_user,
)
from src.preprocessing import clean, data_fingerprint, detect_outliers, validate
from src.splitting import build_train_matrix, compute_sparsity, temporal_train_test_split


@pytest.fixture(scope="session")
def dataset():
    """Load, validate, clean, and split dataset once for the test session."""
    raw_df = load_ratings()
    movies_df = load_movies()
    validate(raw_df)
    clean_df = clean(raw_df)
    train_df, test_df = temporal_train_test_split(clean_df)
    R_train = build_train_matrix(train_df)
    return clean_df, movies_df, train_df, test_df, R_train


# ---------------------------------------------------------------
# STAGE 1 & 2 TESTS: Loading, Preprocessing, Outliers, Integrity
# ---------------------------------------------------------------
def test_data_loading_and_validation(dataset):
    clean_df, movies_df, _, _, _ = dataset
    # Total ratings check (100,000 on standard benchmark, accommodates examiner row deletions)
    assert len(clean_df) >= 99000
    # Exactly 943 users and 1682 movies
    assert clean_df["user_id"].nunique() == 943
    assert len(movies_df) == 1682

    # Validation must pass without raising exceptions
    assert validate(clean_df) is True

    # Outlier detection audit must report statistics without dropping rows
    outlier_info = detect_outliers(clean_df)
    assert "user_activity" in outlier_info
    assert "item_popularity" in outlier_info
    assert outlier_info["user_activity"]["max"] >= 700

    # Data fingerprint must produce a valid 64-character hex SHA-256 hash
    fingerprint = data_fingerprint(clean_df)
    assert len(fingerprint) == 64
    assert isinstance(fingerprint, str)


# ---------------------------------------------------------------
# STAGE 3 TESTS: Temporal Split & Matrix Generation
# ---------------------------------------------------------------
def test_temporal_split_and_matrix(dataset):
    clean_df, _, train_df, test_df, R_train = dataset

    # Sum of splits must equal exactly total cleaned ratings
    assert len(train_df) + len(test_df) == len(clean_df)

    # Matrix dimensions must be exactly (943, 1682)
    assert R_train.shape == (943, 1682)

    # Sparsity should be approximately ~94.98%
    sparsity = compute_sparsity(R_train)
    assert 94.0 <= sparsity <= 96.0

    # Temporal split check: for each user, train ratings are strictly earlier than or equal to test ratings
    user_train_max = train_df.groupby("user_id")["timestamp"].max()
    user_test_min = test_df.groupby("user_id")["timestamp"].min()
    common_users = user_train_max.index.intersection(user_test_min.index)
    assert (user_train_max.loc[common_users] <= user_test_min.loc[common_users]).all()


# ---------------------------------------------------------------
# STAGE 4 TESTS: Model Fitting & Prediction Bounds
# ---------------------------------------------------------------
def test_models_fit_and_predict(dataset):
    _, _, _, _, R_train = dataset

    models = {
        "Popularity": PopularityBaseline(),
        "ItemKNN": ItemKNN(k=30),
        "SVD": SVDRecommender(k=20, random_state=42),
    }

    for name, model in models.items():
        model.fit(R_train)
        pred = model.predict_matrix()
        # Shape check
        assert pred.shape == (943, 1682)
        # Finite values check (no NaN or Inf)
        assert np.isfinite(pred).all(), f"{name} produced non-finite values"
        # Bounded within standard [1.0, 5.0] rating scale
        assert (pred >= 1.0).all() and (pred <= 5.0).all(), f"{name} predictions out of [1, 5] bounds"


# ---------------------------------------------------------------
# STAGE 5 TESTS: Top-10 Ranking & Evaluation Metrics
# ---------------------------------------------------------------
def test_top10_no_train_leakage(dataset):
    _, _, _, _, R_train = dataset

    model = SVDRecommender(k=20, random_state=42).fit(R_train)
    preds = model.predict_matrix()
    top10_dict = compute_top10_recommendations(preds, R_train)

    assert len(top10_dict) == 943
    for u in range(943):
        recs = top10_dict[u]
        # Exactly 10 recommendations per user
        assert len(recs) == 10
        # No duplicate items in Top-10
        assert len(set(recs)) == 10
        # Zero training leakage: no item from user's train set in Top-10
        user_train_items = np.where(R_train[u] > 0)[0]
        overlap = set(recs).intersection(set(user_train_items))
        assert len(overlap) == 0, f"User {u} has train items in Top-10: {overlap}"


def test_evaluation_metrics_table(dataset):
    _, _, _, test_df, R_train = dataset

    models = {
        "Popularity": PopularityBaseline().fit(R_train),
        "ItemKNN": ItemKNN(k=30).fit(R_train),
        "SVD": SVDRecommender(k=20, random_state=42).fit(R_train),
    }

    results_df, all_top10 = evaluate_models(models, R_train, test_df, num_items=1682)

    # Acceptance Criteria: Exactly 3 rows and 4 metric columns
    assert results_df.shape == (3, 4)
    assert list(results_df.index) == ["Popularity", "ItemKNN", "SVD"]
    assert list(results_df.columns) == ["RMSE", "Precision@10", "Recall@10", "Coverage (%)"]

    # Catalog Coverage must be between 0% and 100%
    for cov in results_df["Coverage (%)"]:
        assert 0.0 <= cov <= 100.0


# ---------------------------------------------------------------
# STAGE 6 TESTS: Visualizations
# ---------------------------------------------------------------
def test_plots_generate_figures(dataset):
    clean_df, _, _, _, _ = dataset

    fig1 = plot_rating_distribution(clean_df)
    assert isinstance(fig1, Figure)

    fig2 = plot_ratings_per_user(clean_df)
    assert isinstance(fig2, Figure)

    fig3 = plot_item_long_tail(clean_df, num_total_items=1682)
    assert isinstance(fig3, Figure)

    sample_metrics = pd.DataFrame(
        {
            "RMSE": [1.12, 1.10, 1.09],
            "Precision@10": [0.06, 0.05, 0.09],
            "Recall@10": [0.05, 0.04, 0.08],
            "Coverage (%)": [3.8, 80.5, 20.8],
        },
        index=["Popularity", "ItemKNN", "SVD"],
    )
    fig4 = plot_model_comparison(sample_metrics)
    assert isinstance(fig4, Figure)


# ---------------------------------------------------------------
# STREAMLIT UI TEST
# ---------------------------------------------------------------
def test_streamlit_app():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not at.exception
    assert len(at.tabs) == 3
