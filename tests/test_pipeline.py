"""Unit and integration tests for recommendation system comparison benchmark."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.data import load_and_split_data, load_movies
from src.evaluate import compute_top10_recommendations, evaluate_models
from src.models import ItemKNN, PopularityBaseline, SVDRecommender
from src.plots import (
    plot_item_long_tail,
    plot_model_comparison,
    plot_rating_distribution,
    plot_ratings_per_user,
)


@pytest.fixture(scope="session")
def dataset():
    """Load and cache dataset for tests."""
    train_df, test_df, R_train = load_and_split_data()
    movies_df = load_movies()
    return train_df, test_df, R_train, movies_df


def test_data_shapes_and_splits(dataset):
    train_df, test_df, R_train, movies_df = dataset
    assert len(train_df) + len(test_df) == 100000
    assert R_train.shape == (943, 1682)
    assert len(movies_df) == 1682

    # Check temporal split logic: for each user, train ratings are earlier than test ratings
    # Merge train and test max/min timestamps per user
    user_train_max = train_df.groupby("user_id")["timestamp"].max()
    user_test_min = test_df.groupby("user_id")["timestamp"].min()
    common_users = user_train_max.index.intersection(user_test_min.index)
    # With stable sort, train timestamp max <= test timestamp min
    assert (user_train_max.loc[common_users] <= user_test_min.loc[common_users]).all()


def test_models_fit_and_predict(dataset):
    train_df, test_df, R_train, _ = dataset

    models = {
        "Popularity": PopularityBaseline(),
        "ItemKNN": ItemKNN(k=30),
        "SVD": SVDRecommender(k=20, random_state=42),
    }

    for name, model in models.items():
        model.fit(R_train)
        pred = model.predict_matrix()
        assert pred.shape == (943, 1682)
        assert np.isfinite(pred).all(), f"{name} produced non-finite values"
        assert (pred >= 1.0).all() and (pred <= 5.0).all(), f"{name} predictions out of [1, 5] bounds"


def test_top10_no_train_leakage(dataset):
    train_df, test_df, R_train, _ = dataset

    model = SVDRecommender(k=20, random_state=42).fit(R_train)
    preds = model.predict_matrix()
    top10_dict = compute_top10_recommendations(preds, R_train)

    assert len(top10_dict) == 943
    for u in range(943):
        recs = top10_dict[u]
        assert len(recs) == 10
        # No duplicates in Top-10
        assert len(set(recs)) == 10
        # No item from user's train set in Top-10
        user_train_items = np.where(R_train[u] > 0)[0]
        overlap = set(recs).intersection(set(user_train_items))
        assert len(overlap) == 0, f"User {u} has train items in Top-10: {overlap}"


def test_evaluation_metrics_table(dataset):
    train_df, test_df, R_train, _ = dataset

    models = {
        "Popularity": PopularityBaseline().fit(R_train),
        "ItemKNN": ItemKNN(k=30).fit(R_train),
        "SVD": SVDRecommender(k=20, random_state=42).fit(R_train),
    }

    results_df, all_top10 = evaluate_models(models, R_train, test_df, num_items=1682)

    # Acceptance Criterion: Exactly 3 rows and 4 metric columns
    assert results_df.shape == (3, 4)
    assert list(results_df.index) == ["Popularity", "ItemKNN", "SVD"]
    assert list(results_df.columns) == ["RMSE", "Precision@10", "Recall@10", "Coverage (%)"]

    # Coverage between 0 and 100%
    for cov in results_df["Coverage (%)"]:
        assert 0.0 <= cov <= 100.0


def test_plots_generate_figures(dataset):
    train_df, test_df, R_train, _ = dataset
    total_df = pd.concat([train_df, test_df], ignore_index=True)

    fig1 = plot_rating_distribution(total_df)
    assert isinstance(fig1, Figure)

    fig2 = plot_ratings_per_user(total_df)
    assert isinstance(fig2, Figure)

    fig3 = plot_item_long_tail(total_df, num_total_items=1682)
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


def test_streamlit_app():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not at.exception
    assert len(at.tabs) == 3

