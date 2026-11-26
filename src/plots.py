"""STAGE 6: DATA & MODEL VISUALIZATION MODULE
===========================================
Pipeline Stage: Stage 6 / 7 (Plots)
File: src/plots.py

What this file does:
    Generates all presentation-quality matplotlib/seaborn visualization figures:
    1. Rating distribution (1-5 stars with counts and percentages)
    2. User activity histogram (ratings per user with log-scale x-axis)
    3. Item popularity long-tail (movie rank vs. count + cumulative Pareto curve)
    4. 2x2 Model comparison grid (RMSE, Precision@10, Recall@10, Catalog Coverage)

Inputs:
    - total_df: pd.DataFrame of all 100,000 ratings
    - metrics_df: pd.DataFrame of benchmark results (3 rows, 4 columns)
    - num_total_items: Total catalog items (1682)

Outputs:
    - matplotlib.figure.Figure objects rendered in Streamlit and report figures

Why this stage exists:
    Visual charts allow examiners and stakeholders to visually verify data skewness,
    understand the long-tail dilemma in recommendation datasets, and intuitively
    compare the trade-offs between popularity bias and personalization.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for web apps and multi-threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Apply clean, modern seaborn aesthetic globally
sns.set_theme(style="whitegrid", font="sans-serif")

# Consistent color palette mapped to each model across all charts
MODEL_COLORS = {
    "Popularity": "#4C72B0",  # Muted Corporate Blue
    "ItemKNN": "#DD8452",     # Muted Coral / Orange
    "SVD": "#55A868",         # Muted Forest Green
}


# ---------------------------------------------------------------
# SECTION: EDA Figure 1 - Rating Distribution
# ---------------------------------------------------------------
def plot_rating_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plot the frequency and percentage breakdown of 1 to 5 star ratings.

    Args:
        df: Ratings DataFrame containing the 'rating' column.

    Returns:
        plt.Figure: Matplotlib figure ready for display.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Count occurrences of each rating value (1, 2, 3, 4, 5)
    rating_counts = df["rating"].value_counts().sort_index()
    total_ratings = len(df)
    ratings = rating_counts.index.astype(int)
    counts = rating_counts.values

    # Render bar chart
    bars = ax.bar(ratings, counts, color="#3470A3", edgecolor="#1D4367", width=0.6, alpha=0.9)

    # Annotate each bar with its exact count and percentage of total ratings
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


# ---------------------------------------------------------------
# SECTION: EDA Figure 2 - Ratings Per User (Log Scale)
# ---------------------------------------------------------------
def plot_ratings_per_user(df: pd.DataFrame) -> plt.Figure:
    """Plot histogram of rating activity per user with a logarithmic x-axis.

    Args:
        df: Ratings DataFrame containing 'user_id' and 'rating'.

    Returns:
        plt.Figure: Matplotlib figure ready for display.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Calculate total ratings given by each individual user
    user_counts = df.groupby("user_id")["rating"].count().values
    min_count = user_counts.min()
    max_count = user_counts.max()
    median_val = np.median(user_counts)
    mean_val = np.mean(user_counts)

    # Use logarithmically spaced bins so the heavy right tail is cleanly displayed
    bins = np.geomspace(min_count, max_count, 25)
    ax.hist(user_counts, bins=bins, color="#4C956C", edgecolor="#2C5E43", alpha=0.85)

    ax.set_xscale("log")
    ax.set_title("User Activity: Ratings per User", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Ratings Given (Log Scale)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Number of Users", fontsize=11, fontweight="medium")

    # Add vertical reference lines for median and mean to show activity skew
    ax.axvline(median_val, color="#D95F02", linestyle="--", linewidth=1.8, label=f"Median: {median_val:.0f}")
    ax.axvline(mean_val, color="#7570B3", linestyle=":", linewidth=2.0, label=f"Mean: {mean_val:.1f}")

    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------
# SECTION: EDA Figure 3 - Item Popularity Long Tail & Pareto Curve
# ---------------------------------------------------------------
def plot_item_long_tail(df: pd.DataFrame, num_total_items: int = 1682) -> plt.Figure:
    """Plot 2-panel chart: movie rank vs rating count, and cumulative catalog coverage.

    Args:
        df: Ratings DataFrame containing 'item_id'.
        num_total_items: Total movies in catalog (1682).

    Returns:
        plt.Figure: 1x2 subplots figure ready for display.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Tally ratings for all items, assigning 0 to unrated items in the catalog
    item_counts_series = df["item_id"].value_counts()
    all_counts = np.zeros(num_total_items, dtype=int)
    for item_id, count in item_counts_series.items():
        if item_id < num_total_items:
            all_counts[item_id] = count

    # Sort descending to visualize the classic power-law / long-tail curve
    sorted_counts = np.sort(all_counts)[::-1]
    ranks = np.arange(1, num_total_items + 1)

    # Subplot 1: Movie Rank vs Number of Ratings (The Long Tail)
    ax1.plot(ranks, sorted_counts, color="#2B5C8F", linewidth=2.0)
    ax1.fill_between(ranks, sorted_counts, color="#2B5C8F", alpha=0.25)
    ax1.set_title("Item Popularity Long Tail (Rank vs. Count)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Movie Rank (by Popularity)", fontsize=10.5, fontweight="medium")
    ax1.set_ylabel("Number of Ratings", fontsize=10.5, fontweight="medium")
    ax1.set_xlim(1, num_total_items)
    ax1.set_ylim(0, sorted_counts[0] * 1.05)

    # Subplot 2: Cumulative Rating Share (Lorenz / Pareto Curve)
    cum_ratings = np.cumsum(sorted_counts)
    total_ratings = cum_ratings[-1]
    cum_pct_ratings = (cum_ratings / total_ratings) * 100.0
    movie_pct = (ranks / num_total_items) * 100.0

    ax2.plot(movie_pct, cum_pct_ratings, color="#D95F02", linewidth=2.2, label="Actual Concentration")
    ax2.plot([0, 100], [0, 100], color="#888888", linestyle="--", linewidth=1.2, label="Uniform Parity")

    # Highlight the Top 20% point demonstrating the classic 80/20 Pareto rule
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


# ---------------------------------------------------------------
# SECTION: Model Comparison 2x2 Bar Grid
# ---------------------------------------------------------------
def plot_model_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """Plot 2x2 grid of bar charts comparing models across all 4 metrics.

    Metrics:
        1. RMSE (Lower is Better)
        2. Precision@10 (Higher is Better)
        3. Recall@10 (Higher is Better)
        4. Catalog Coverage % (Higher is Better)

    Args:
        metrics_df: Results DataFrame indexed by model name.

    Returns:
        plt.Figure: 2x2 subplots comparison grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    models = list(metrics_df.index)
    colors = [MODEL_COLORS.get(m, "#4C72B0") for m in models]

    metric_configs = [
        ("RMSE", axes[0, 0], "RMSE (Lower is Better)", "{:.4f}"),
        ("Precision@10", axes[0, 1], "Precision@10 (Higher is Better)", "{:.4f}"),
        ("Recall@10", axes[1, 0], "Recall@10 (Higher is Better)", "{:.4f}"),
        ("Coverage (%)", axes[1, 1], "Catalog Coverage % (Higher is Better)", "{:.2f}%"),
    ]

    for col_name, ax, title, fmt in metric_configs:
        values = metrics_df[col_name].values
        bars = ax.bar(models, values, color=colors, edgecolor="#333333", width=0.55, alpha=0.9)

        # Label each bar directly with its formatted numeric value
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
