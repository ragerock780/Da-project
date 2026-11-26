# Comparing Recommendation Algorithms on MovieLens 100K

A clean, reproducible, and modular benchmark comparing three classical recommendation algorithms on MovieLens 100K using an evaluation protocol with per-user temporal train/test splitting.

---

## 7-Stage Pipeline Story
1. **Stage 1 (Data Loading)**: MovieLens 100K ratings (`u.data`) and movie catalog (`u.item`) are auto-downloaded, parsed, and converted to zero-based integer indices.
2. **Stage 2 (Preprocessing & Outliers)**: Validates schema integrity, audits activity outliers via IQR (retained for benchmark authenticity), and hashes the dataset with SHA-256 for change tracking.
3. **Stage 3 (Temporal Split)**: Chronological per-user 80/20 train/test split preventing future data leakage, followed by dense matrix construction ($R_{\text{train}}$, shape $943 \times 1682$, ~$95\%$ sparsity).
4. **Stage 4 (Models Training)**: Trains three classical algorithms from scratch: Bayesian Weighted Popularity, Item-based Collaborative Filtering (ItemKNN, top-30 neighbors), and Truncated SVD ($k=20$ latent factors).
5. **Stage 5 (Evaluation)**: Masks training items ($-\infty$), builds Top-10 lists, and evaluates RMSE, Precision@10, Recall@10 (on ratings $\ge 4.0$), and Catalog Coverage (%).
6. **Stage 6 (Save & Visualize)**: Persists benchmark artifacts to `results/` and generates presentation figures.
7. **Stage 7 (Refactor Check)**: Asserts that all evaluation metrics match pre-refactor baselines to 6 decimal places.

For an in-depth breakdown of every file, function, missing-value strategy, and outlier policy, see [**PROJECT_MAP.md**](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/PROJECT_MAP.md).

---

## Project Structure
```text
recsys-comparison/
├── src/
│   ├── data_loading.py       # [Stage 1] Download, parse, 0-based conversion
│   ├── preprocessing.py      # [Stage 2] Schema validation, IQR outlier audit, SHA-256
│   ├── splitting.py          # [Stage 3] Per-user temporal 80/20 split, R_train matrix
│   ├── models/               # [Stage 4] Recommendation models
│   │   ├── __init__.py
│   │   ├── popularity.py     # Bayesian weighted rating (IMDB formula)
│   │   ├── item_knn.py       # Cosine similarity Item-CF (top-30 neighbors)
│   │   └── svd.py            # Mean-centered Truncated SVD (k=20 factors)
│   ├── evaluation/           # [Stage 5] Ranking & Metrics
│   │   ├── __init__.py
│   │   ├── ranking.py        # Top-10 list generation & train-item masking
│   │   └── metrics.py        # RMSE, Precision@10, Recall@10, Catalog Coverage
│   └── plots.py              # [Stage 6] Presentation matplotlib/seaborn figures
├── run_experiments.py        # 7-stage story pipeline runner
├── app.py                    # Streamlit 3-tab interactive web app
├── PROJECT_MAP.md            # Detailed architecture and function mapping
├── requirements.txt          # Pinned minimum dependencies
└── README.md                 # Project documentation
```

---

## Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark Pipeline
```bash
python run_experiments.py
```
*Executes all 7 stages in under 2 seconds, prints the metrics table, and performs the refactor verification check.*

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
*(or `python -m streamlit run app.py`)*

---

## Key Talking Points & Search Tags
Search for these tags across the codebase (using `Ctrl+F`) to inspect core design decisions:
- `# [PREPROCESSING]`: Data validation, schema checks, and deduplication.
- `# [MISSING VALUES]`: How unrated matrix entries are represented as 0.0 and handled per model.
- `# [OUTLIERS]`: IQR detection of power users and blockbusters (report only).
- `# [EVALUATION]`: Train-item masking with $-\infty$ and evaluation metrics calculations.
- `# [ALGORITHM]`: Mathematical formulas for Popularity, ItemKNN, and SVD.
- `# [DATA CHANGE TRACKING]`: SHA-256 fingerprinting and dataset integrity checks.
