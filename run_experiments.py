"""Main experiment runner for comparing recommendation algorithms on MovieLens 100K.
Executes data loading, model fitting, evaluation, and persists all results.
"""

import pickle
import time
from pathlib import Path
import numpy as np

from src.data import load_and_split_data, load_movies
from src.evaluate import evaluate_models
from src.models import ItemKNN, PopularityBaseline, SVDRecommender

RANDOM_SEED = 42


def run_pipeline(results_dir: Path | str = "results") -> None:
    """Run full recommendation benchmark pipeline and save artifacts."""
    start_time = time.time()
    np.random.seed(RANDOM_SEED)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" COMPARING RECOMMENDATION ALGORITHMS ON MOVIELENS 100K")
    print("=" * 65)

    # 1. Load data & perform temporal split
    print("\n[1/4] Loading dataset and performing per-user temporal split...")
    train_df, test_df, R_train = load_and_split_data()
    movies_df = load_movies()

    # 2. Initialize and fit models
    print("\n[2/4] Training recommendation models...")
    models = {
        "Popularity": PopularityBaseline(),
        "ItemKNN": ItemKNN(k=30),
        "SVD": SVDRecommender(k=20, random_state=RANDOM_SEED),
    }

    for name, model in models.items():
        t0 = time.time()
        print(f"  - Fitting {name}...", end="", flush=True)
        model.fit(R_train)
        print(f" done in {time.time() - t0:.2f}s")

    # 3. Evaluate models
    print("\n[3/4] Evaluating models (RMSE, Precision@10, Recall@10, Catalog Coverage)...")
    results_df, top10_recommendations = evaluate_models(
        models=models,
        R_train=R_train,
        test_df=test_df,
        num_items=R_train.shape[1],
    )

    # 4. Save artifacts
    print("\n[4/4] Saving results and experiment artifacts...")
    metrics_path = results_path / "metrics.csv"
    results_df.to_csv(metrics_path)
    print(f"  - Saved metrics to {metrics_path}")

    top10_path = results_path / "top10_recommendations.pkl"
    with open(top10_path, "wb") as f:
        pickle.dump(top10_recommendations, f)
    print(f"  - Saved top-10 recommendations to {top10_path}")

    train_path = results_path / "train.csv"
    test_path = results_path / "test.csv"
    movies_path = results_path / "movies.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    movies_df.to_csv(movies_path, index=False)
    print(f"  - Saved train/test splits and movie metadata to {results_path}")

    # Print metrics table to console
    print("\n" + "=" * 65)
    print(" FINAL RESULTS TABLE")
    print("=" * 65)
    formatted_df = results_df.copy()
    formatted_df["RMSE"] = formatted_df["RMSE"].map(lambda x: f"{x:.4f}")
    formatted_df["Precision@10"] = formatted_df["Precision@10"].map(lambda x: f"{x:.4f}")
    formatted_df["Recall@10"] = formatted_df["Recall@10"].map(lambda x: f"{x:.4f}")
    formatted_df["Coverage (%)"] = formatted_df["Coverage (%)"].map(lambda x: f"{x:.2f}%")
    print(formatted_df.to_string())
    print("=" * 65)

    elapsed = time.time() - start_time
    print(f"\nPipeline finished successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    run_pipeline()
