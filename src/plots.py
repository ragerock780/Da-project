"""Visualization module for MovieLens 100K data and recommender evaluation.
All functions return matplotlib Figure objects with seaborn styling.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for web apps and multi-threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set global seaborn aesthetic
sns.set_theme(style="whitegrid", font="sans-serif")

MODEL_COLORS = {
    "Popularity": "#4C72B0",  # Muted Blue
    "ItemKNN": "#DD8452",     # Muted Coral
    "SVD": "#55A868",         # Muted Green
}


def plot_rating_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plot count and percentage of 1-5 star ratings."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    rating_counts = df["rating"].value_counts().sort_index()
    total_ratings = len(df)
    ratings = rating_counts.index.astype(int)
    counts = rating_counts.values

    bars = ax.bar(ratings, counts, color="#3470A3", edgecolor="#1D4367", width=0.6, alpha=0.9)

    for bar in bars:
        h = bar.get_height()
        pct = (h / total_ratings) * 100
        ax.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
        )

    ax.set_title("Rating Distribution (1–5 Stars)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Rating (Stars)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Number of Ratings", fontsize=11, fontweight="medium")
    ax.set_xticks(ratings)
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    return fig


def plot_ratings_per_user(df: pd.DataFrame) -> plt.Figure:
    """Plot histogram of rating count per user with a log-scale x-axis."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    user_counts = df.groupby("user_id")["rating"].count().values
    min_count = user_counts.min()
    max_count = user_counts.max()
    median_val = np.median(user_counts)
    mean_val = np.mean(user_counts)

    # Logarithmically spaced bins for log x-axis
    bins = np.geomspace(min_count, max_count, 25)
    ax.hist(user_counts, bins=bins, color="#4C956C", edgecolor="#2C5E43", alpha=0.85)

    ax.set_xscale("log")
    ax.set_title("User Activity: Ratings per User", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Ratings Given (Log Scale)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Number of Users", fontsize=11, fontweight="medium")

    # Reference lines for median and mean
    ax.axvline(median_val, color="#D95F02", linestyle="--", linewidth=1.8, label=f"Median: {median_val:.0f}")
    ax.axvline(mean_val, color="#7570B3", linestyle=":", linewidth=2.0, label=f"Mean: {mean_val:.1f}")

    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    fig.tight_layout()
    return fig


def plot_item_long_tail(df: pd.DataFrame, num_total_items: int = 1682) -> plt.Figure:
    """Plot item popularity long tail:
    1) Movie rank vs rating count.
    2) Cumulative percentage of ratings covered by top X% of movies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Item counts including zero-rated items in catalog
    item_counts_series = df["item_id"].value_counts()
    # Fill in zero for any unrated items up to num_total_items
    all_counts = np.zeros(num_total_items, dtype=int)
    for item_id, count in item_counts_series.items():
        if item_id < num_total_items:
            all_counts[item_id] = count

    sorted_counts = np.sort(all_counts)[::-1]
    ranks = np.arange(1, num_total_items + 1)

    # Subplot 1: Rank vs Rating Count
    ax1.plot(ranks, sorted_counts, color="#2B5C8F", linewidth=2.0)
    ax1.fill_between(ranks, sorted_counts, color="#2B5C8F", alpha=0.25)
    ax1.set_title("Item Popularity Long Tail (Rank vs. Count)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Movie Rank (by Popularity)", fontsize=10.5, fontweight="medium")
    ax1.set_ylabel("Number of Ratings", fontsize=10.5, fontweight="medium")
    ax1.set_xlim(1, num_total_items)
    ax1.set_ylim(0, sorted_counts[0] * 1.05)

    # Subplot 2: Cumulative % Coverage
    cum_ratings = np.cumsum(sorted_counts)
    total_ratings = cum_ratings[-1]
    cum_pct_ratings = (cum_ratings / total_ratings) * 100.0
    movie_pct = (ranks / num_total_items) * 100.0

    ax2.plot(movie_pct, cum_pct_ratings, color="#D95F02", linewidth=2.2, label="Actual Concentration")
    ax2.plot([0, 100], [0, 100], color="#888888", linestyle="--", linewidth=1.2, label="Uniform Parity")

    # Mark top 20% movies point
    idx_20 = int(0.20 * num_total_items) - 1
    pct_at_20 = cum_pct_ratings[idx_20]
    ax2.scatter([20.0], [pct_at_20], color="#B22222", zorder=5, s=40)
    ax2.annotate(
        f"Top 20% movies =\n{pct_at_20:.1f}% of ratings",
        xy=(20.0, pct_at_20),
        xytext=(30, pct_at_20 - 15),
        arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.2),
        fontweight="semibold",
        fontsize=9.5,
        color="#B22222",
    )

    ax2.set_title("Cumulative Rating Share by Catalog %", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Top X% of Movies in Catalog", fontsize=10.5, fontweight="medium")
    ax2.set_ylabel("Cumulative % of Total Ratings", fontsize=10.5, fontweight="medium")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="lower right")

    fig.tight_layout()
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """Plot 2x2 grid of bar charts comparing models across all 4 metrics:
    RMSE, Precision@10, Recall@10, Catalog Coverage (%).
    Each model has a consistent color across all subplots, with values labeled on bars.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    models = list(metrics_df.index)
    colors = [MODEL_COLORS.get(m, "#4C72B0") for m in models]

    metric_configs = [
        ("RMSE", axes[0, 0], "RMSE (Lower is Better)", "{:.4f}", False),
        ("Precision@10", axes[0, 1], "Precision@10 (Higher is Better)", "{:.4f}", True),
        ("Recall@10", axes[1, 0], "Recall@10 (Higher is Better)", "{:.4f}", True),
        ("Coverage (%)", axes[1, 1], "Catalog Coverage % (Higher is Better)", "{:.2f}%", True),
    ]

    for col_name, ax, title, fmt, higher_better in metric_configs:
        values = metrics_df[col_name].values
        bars = ax.bar(models, values, color=colors, edgecolor="#333333", width=0.55, alpha=0.9)

        # Value labels on top of bars
        for bar in bars:
            h = bar.get_height()
            label_text = fmt.format(h)
            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_ylabel(col_name, fontsize=10, fontweight="medium")
        max_val = max(values) if max(values) > 0 else 1.0
        ax.set_ylim(0, max_val * 1.18)
        ax.tick_params(axis="x", labelsize=10.5)

    fig.suptitle("Recommendation Algorithms Performance Comparison", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
