"""Data loading, auto-download, preprocessing, and temporal train/test splitting for MovieLens 100K."""

from pathlib import Path
import ssl
import urllib.request
import zipfile
import numpy as np
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
NUM_USERS = 943
NUM_ITEMS = 1682
NUM_RATINGS = 100000


def _get_ssl_context() -> ssl.SSLContext:
    """Create a resilient SSL context, falling back to unverified if certificates fail."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl.create_default_context()
        except Exception:
            return ssl._create_unverified_context()


def ensure_movielens_data(data_dir: Path | str = "data") -> Path:
    """Download and extract MovieLens 100K if not already present."""
    data_path = Path(data_dir)
    ml_100k_dir = data_path / "ml-100k"
    u_data = ml_100k_dir / "u.data"
    u_item = ml_100k_dir / "u.item"

    if ml_100k_dir.exists() and u_data.exists() and u_item.exists():
        return ml_100k_dir

    data_path.mkdir(parents=True, exist_ok=True)
    zip_path = data_path / "ml-100k.zip"

    print(f"Downloading MovieLens 100K from {MOVIELENS_URL}...")
    req = urllib.request.Request(MOVIELENS_URL, headers={"User-Agent": "Mozilla/5.0"})
    context = _get_ssl_context()
    try:
        with urllib.request.urlopen(req, context=context) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as e:
        print(f"Standard SSL failed ({e}), retrying with unverified context...")
        unverified_context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=unverified_context) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    print(f"Extracting {zip_path.name} to {data_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    if zip_path.exists():
        zip_path.unlink()

    return ml_100k_dir


def load_movies(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load movie catalog metadata (0-based item_id and title)."""
    ml_100k_dir = ensure_movielens_data(data_dir)
    u_item = ml_100k_dir / "u.item"

    movies_df = pd.read_csv(
        u_item,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"],
    )
    movies_df["item_id"] = movies_df["movie_id"] - 1
    movies_df = movies_df[["item_id", "title"]].sort_values("item_id").reset_index(drop=True)
    return movies_df


def load_and_split_data(
    data_dir: Path | str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load MovieLens 100K, apply per-user temporal split, and return train/test data and dense train matrix.

    Temporal split per user:
    - Sort each user's ratings by timestamp using a stable sort.
    - First floor(0.8 * n) ratings -> train_df.
    - Remaining ratings -> test_df.
    - Dense train matrix R_train of shape (943, 1682) where 0 indicates missing.
    """
    ml_100k_dir = ensure_movielens_data(data_dir)
    u_data = ml_100k_dir / "u.data"

    df = pd.read_csv(
        u_data,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    # Convert to 0-based indices
    df["user_id"] = df["user_id"] - 1
    df["item_id"] = df["item_id"] - 1

    # Sanity checks on raw data
    assert len(df) == NUM_RATINGS, f"Expected {NUM_RATINGS} ratings, got {len(df)}"
    assert df["user_id"].nunique() == NUM_USERS, f"Expected {NUM_USERS} users, got {df['user_id'].nunique()}"
    assert df["item_id"].max() < NUM_ITEMS, f"Expected items < {NUM_ITEMS}, got max {df['item_id'].max()}"
    assert df["user_id"].min() >= 0 and df["item_id"].min() >= 0

    # Per-user temporal split: sort stable by user_id and timestamp
    df_sorted = df.sort_values(by=["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Compute per-user rating index and train threshold floor(0.8 * n)
    df_sorted["user_rating_rank"] = df_sorted.groupby("user_id").cumcount()
    user_counts = df_sorted.groupby("user_id")["rating"].transform("count")
    df_sorted["train_threshold"] = np.floor(0.8 * user_counts).astype(int)

    is_train = df_sorted["user_rating_rank"] < df_sorted["train_threshold"]

    cols_to_drop = ["user_rating_rank", "train_threshold"]
    train_df = df_sorted[is_train].drop(columns=cols_to_drop).reset_index(drop=True)
    test_df = df_sorted[~is_train].drop(columns=cols_to_drop).reset_index(drop=True)

    assert len(train_df) + len(test_df) == NUM_RATINGS, "Train + test counts must equal total ratings"

    # Dense train matrix R_train (943 x 1682, 0 = missing)
    R_train = np.zeros((NUM_USERS, NUM_ITEMS), dtype=float)
    R_train[train_df["user_id"].values, train_df["item_id"].values] = train_df["rating"].values

    # Sanity check metrics
    sparsity = (1.0 - (len(train_df) / (NUM_USERS * NUM_ITEMS))) * 100.0
    print("--- MovieLens 100K Data Sanity Check ---")
    print(f"Total ratings: {NUM_RATINGS:,}")
    print(f"Train ratings: {len(train_df):,} ({len(train_df)/NUM_RATINGS*100:.2f}%)")
    print(f"Test ratings:  {len(test_df):,} ({len(test_df)/NUM_RATINGS*100:.2f}%)")
    print(f"Users: {NUM_USERS}, Items: {NUM_ITEMS}")
    print(f"Train matrix shape: {R_train.shape}")
    print(f"Train matrix sparsity: {sparsity:.2f}%")
    print("----------------------------------------")

    return train_df, test_df, R_train


if __name__ == "__main__":
    train_df, test_df, R_train = load_and_split_data()
