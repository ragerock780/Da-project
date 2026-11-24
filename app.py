"""Streamlit UI for Comparing Recommendation Algorithms on MovieLens 100K.
Loads pre-computed results from results/ without retraining.
"""

from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

from src.plots import (
    plot_item_long_tail,
    plot_model_comparison,
    plot_rating_distribution,
    plot_ratings_per_user,
)

RESULTS_DIR = Path("results")

st.set_page_config(
    page_title="Comparing Recommendation Algorithms",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data
def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load pre-computed benchmark artifacts from results/."""
    metrics_path = RESULTS_DIR / "metrics.csv"
    train_path = RESULTS_DIR / "train.csv"
    test_path = RESULTS_DIR / "test.csv"
    movies_path = RESULTS_DIR / "movies.csv"
    top10_path = RESULTS_DIR / "top10_recommendations.pkl"

    if not (metrics_path.exists() and train_path.exists() and test_path.exists() and movies_path.exists() and top10_path.exists()):
        raise FileNotFoundError(
            "Precomputed results not found. Please run 'python run_experiments.py' first."
        )

    metrics_df = pd.read_csv(metrics_path, index_col=0)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    movies_df = pd.read_csv(movies_path)
    with open(top10_path, "rb") as f:
        top10_recommendations = pickle.load(f)

    return metrics_df, train_df, test_df, movies_df, top10_recommendations


def main() -> None:
    st.title("Comparing Recommendation Algorithms")
    st.caption("Benchmark of PopularityBaseline, ItemKNN, and SVD on MovieLens 100K (Per-User Temporal Split)")

    try:
        metrics_df, train_df, test_df, movies_df, top10_recommendations = load_cached_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Execute `python run_experiments.py` in your terminal to generate results.")
        return

    movie_title_map = dict(zip(movies_df["item_id"], movies_df["title"]))
    total_df = pd.concat([train_df, test_df], ignore_index=True)

    tab1, tab2, tab3 = st.tabs(["Data Exploration", "Model Comparison", "User Inspector"])

    # ---------------- TAB 1: DATA EXPLORATION ----------------
    with tab1:
        st.subheader("MovieLens 100K Dataset Overview")
        num_users = total_df["user_id"].nunique()
        num_movies = len(movies_df)
        total_ratings = len(total_df)
        sparsity = (1.0 - (total_ratings / (num_users * num_movies))) * 100.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Users", f"{num_users:,}")
        col2.metric("Movies", f"{num_movies:,}")
        col3.metric("Ratings", f"{total_ratings:,}")
        col4.metric("Matrix Sparsity", f"{sparsity:.2f}%")

        st.markdown("---")
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            st.pyplot(plot_rating_distribution(total_df))
        with eda_col2:
            st.pyplot(plot_ratings_per_user(total_df))

        st.pyplot(plot_item_long_tail(total_df, num_total_items=num_movies))

    # ---------------- TAB 2: MODEL COMPARISON ----------------
    with tab2:
        st.subheader("Performance Metrics")
        display_metrics = metrics_df.copy()
        display_metrics["RMSE"] = display_metrics["RMSE"].map(lambda x: f"{x:.4f}")
        display_metrics["Precision@10"] = display_metrics["Precision@10"].map(lambda x: f"{x:.4f}")
        display_metrics["Recall@10"] = display_metrics["Recall@10"].map(lambda x: f"{x:.4f}")
        display_metrics["Coverage (%)"] = display_metrics["Coverage (%)"].map(lambda x: f"{x:.2f}%")
        st.table(display_metrics)

        st.subheader("Comparison Charts")
        st.pyplot(plot_model_comparison(metrics_df))

        st.markdown("### Key Takeaways")
        st.write(
            "The Popularity baseline achieves competitive Precision@10 by repeatedly recommending universally acclaimed blockbusters, "
            "but suffers from near-zero catalog coverage (~1-2%) and provides no personalization. "
            "In contrast, ItemKNN and SVD deliver substantial personalization by exploring the catalog, achieving far higher catalog coverage "
            "while maintaining strong ranking metrics and significantly lower prediction error (RMSE)."
        )

    # ---------------- TAB 3: USER INSPECTOR ----------------
    with tab3:
        st.subheader("User Recommendation Inspector")
        all_users = sorted(train_df["user_id"].unique())
        selected_user = st.selectbox(
            "Select User ID",
            options=all_users,
            format_func=lambda u: f"User {u}",
        )

        st.markdown(f"#### Top 5 Highest-Rated Train Movies for User {selected_user}")
        user_train = (
            train_df[train_df["user_id"] == selected_user]
            .sort_values(by=["rating", "timestamp"], ascending=[False, False])
            .head(5)
        )
        user_train_display = []
        for _, row in user_train.iterrows():
            item_id = int(row["item_id"])
            title = movie_title_map.get(item_id, f"Movie {item_id}")
            user_train_display.append({"Title": title, "Rating": int(row["rating"])})
        st.dataframe(pd.DataFrame(user_train_display), hide_index=True)

        st.markdown("#### Model Top-10 Recommendations")
        st.caption("Titles marked with ✅ indicate hits in the user's test set with rating ≥ 4.0.")

        # Test set relevant items for selected user
        user_test_relevant = test_df[
            (test_df["user_id"] == selected_user) & (test_df["rating"] >= 4.0)
        ]
        relevant_test_item_ratings = dict(
            zip(user_test_relevant["item_id"].astype(int), user_test_relevant["rating"])
        )

        m_cols = st.columns(3)
        model_names = ["Popularity", "ItemKNN", "SVD"]

        for idx, m_name in enumerate(model_names):
            with m_cols[idx]:
                st.markdown(f"**{m_name}**")
                recs = top10_recommendations.get(m_name, {}).get(selected_user, [])
                for rank, item_id in enumerate(recs, start=1):
                    title = movie_title_map.get(item_id, f"Movie {item_id}")
                    if item_id in relevant_test_item_ratings:
                        rating_val = relevant_test_item_ratings[item_id]
                        st.markdown(f"**{rank}.** {title} ✅ *(test: {rating_val:.0f}★)*")
                    else:
                        st.markdown(f"**{rank}.** {title}")


if __name__ == "__main__":
    main()
