"""STAGE 1: DATA LOADING MODULE
==============================
Pipeline Stage: Stage 1 / 7
File: src/data_loading.py

What this file does:
    Downloads the MovieLens 100K dataset from GroupLens if it is not already present,
    unpacks it, loads the raw ratings (u.data) and movie catalog (u.item), and converts
    all 1-based user and item IDs into 0-based integers for efficient matrix indexing.

Inputs:
    - Path to data directory (default: 'data')
    - Remote URL: https://files.grouplens.org/datasets/movielens/ml-100k.zip

Outputs:
    - ratings_df: pd.DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
    - movies_df: pd.DataFrame with columns ['item_id', 'title']

Why this stage exists:
    Machine learning algorithms and NumPy matrices use zero-based array indexing, so we
    must ensure the raw data is safely acquired and indexed starting at 0 before any
    computations begin.
"""

from pathlib import Path
import ssl
import urllib.request
import zipfile
import pandas as pd

# ---------------------------------------------------------------
# SECTION: Dataset Constants
# ---------------------------------------------------------------
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
NUM_USERS = 943       # Exactly 943 unique users in MovieLens 100K
NUM_ITEMS = 1682      # Exactly 1682 unique movies in MovieLens 100K
NUM_RATINGS = 100000  # Exactly 100,000 ratings in MovieLens 100K


# ---------------------------------------------------------------
# SECTION: Resilient Download & Extraction
# ---------------------------------------------------------------
def get_ssl_context() -> ssl.SSLContext:
    """Create a resilient SSL context, falling back to unverified if certificates fail.

    Returns:
        ssl.SSLContext: Configured SSL context for secure HTTPS downloads.
    """
    # Try using the system/certifi CA bundle first for secure verification
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # Fall back to unverified context if local Windows CA certificates are missing
        try:
            return ssl.create_default_context()
        except Exception:
            return ssl._create_unverified_context()


def ensure_movielens_data(data_dir: Path | str = "data") -> Path:
    """Check if MovieLens 100K exists locally; download and extract it if missing.

    Args:
        data_dir: Directory where data should be stored.

    Returns:
        Path: Path to the extracted 'ml-100k' folder.
    """
    data_path = Path(data_dir)
    ml_100k_dir = data_path / "ml-100k"
    u_data = ml_100k_dir / "u.data"
    u_item = ml_100k_dir / "u.item"

    # Avoid re-downloading if the files already exist on disk
    if ml_100k_dir.exists() and u_data.exists() and u_item.exists():
        return ml_100k_dir

    # Create destination folder if it does not exist
    data_path.mkdir(parents=True, exist_ok=True)
    zip_path = data_path / "ml-100k.zip"

    print(f"Downloading MovieLens 100K from {MOVIELENS_URL}...")
    req = urllib.request.Request(MOVIELENS_URL, headers={"User-Agent": "Mozilla/5.0"})
    context = get_ssl_context()
    try:
        with urllib.request.urlopen(req, context=context) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as e:
        print(f"Standard SSL download failed ({e}), retrying with unverified context...")
        unverified_context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=unverified_context) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Extract all files from the zip archive into data/
    print(f"Extracting {zip_path.name} to {data_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    # Clean up the zip file to save disk space
    if zip_path.exists():
        zip_path.unlink()

    return ml_100k_dir


# ---------------------------------------------------------------
# SECTION: Loading & ID Conversion
# ---------------------------------------------------------------
def load_ratings(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load u.data ratings file and convert IDs to 0-based indices.

    Args:
        data_dir: Directory where the data folder is located.

    Returns:
        pd.DataFrame: DataFrame containing ['user_id', 'item_id', 'rating', 'timestamp'].
    """
    # Ensure files are present before reading
    ml_100k_dir = ensure_movielens_data(data_dir)
    u_data = ml_100k_dir / "u.data"

    # Raw u.data is tab-separated with no headers
    df = pd.read_csv(
        u_data,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    # Convert 1-based user and item IDs to 0-based integers
    # e.g., user 1 becomes user 0, item 1682 becomes item 1681
    df["user_id"] = df["user_id"] - 1
    df["item_id"] = df["item_id"] - 1

    # Sanity checks on shapes to ensure no corrupted rows exist
    if len(df) != NUM_RATINGS:
        print(f"  [DATA MODIFICATION NOTICE] File u.data contains {len(df):,} ratings (benchmark baseline is {NUM_RATINGS:,}).")
    assert len(df) > 0, "Ratings file is empty!"
    assert df["item_id"].max() < NUM_ITEMS, f"Expected items < {NUM_ITEMS}, got max {df['item_id'].max()}"
    assert df["user_id"].min() >= 0 and df["item_id"].min() >= 0, "IDs must be non-negative"

    return df


def load_movies(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load u.item movie catalog metadata and convert movie_id to 0-based item_id.

    Args:
        data_dir: Directory where the data folder is located.

    Returns:
        pd.DataFrame: DataFrame containing ['item_id', 'title'].
    """
    # Ensure files are present before reading
    ml_100k_dir = ensure_movielens_data(data_dir)
    u_item = ml_100k_dir / "u.item"

    # u.item is pipe-separated and encoded in latin-1 (contains French accents)
    movies_df = pd.read_csv(
        u_item,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"],
    )

    # Convert movie_id to 0-based item_id to align with ratings_df
    movies_df["item_id"] = movies_df["movie_id"] - 1
    movies_df = movies_df[["item_id", "title"]].sort_values("item_id").reset_index(drop=True)

    # Sanity check: must have exactly 1682 unique movies
    assert len(movies_df) == NUM_ITEMS, f"Expected {NUM_ITEMS} movies, got {len(movies_df)}"

    return movies_df


def load_raw_data(data_dir: Path | str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience helper to load both ratings and movies in one call.

    Args:
        data_dir: Base directory for dataset storage.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (ratings_df, movies_df).
    """
    ratings_df = load_ratings(data_dir)
    movies_df = load_movies(data_dir)
    return ratings_df, movies_df
