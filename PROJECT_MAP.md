# Project Architecture & Pipeline Map

This document provides an exhaustive overview of the codebase architecture, pipeline stages, responsibilities, and algorithm implementations for non-expert examiners and reviewers.

---

## 1. Pipeline Execution Table

| Stage | What Happens | File | Function(s) / Class(es) |
| :--- | :--- | :--- | :--- |
| **Stage 1: Loading** | Auto-downloads MovieLens 100K zip if missing, unpacks raw files, loads `u.data` (ratings) and `u.item` (titles), and converts all 1-based user/movie IDs to 0-based indices. | [`src/data_loading.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/data_loading.py) | `ensure_movielens_data()`, `load_ratings()`, `load_movies()`, `load_raw_data()` |
| **Stage 2: Validation** | Verifies expected schema, checks absence of null values, confirms rating range [1.0, 5.0], and validates exact count of 100,000 ratings across 943 users and 1,682 items. | [`src/preprocessing.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/preprocessing.py) | `validate()` |
| **Stage 2: Outliers** | Audits user rating activity and movie popularity distributions using the Interquartile Range (IQR); logs high-activity power users and blockbuster movies without deletion. | [`src/preprocessing.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/preprocessing.py) | `detect_outliers()`, `outlier_summary()` |
| **Stage 2: Cleaning** | Deduplicates records and casts column datatypes while guaranteeing zero data loss (exact 100,000 row count preserved). | [`src/preprocessing.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/preprocessing.py) | `clean()` |
| **Stage 2: Data Tracking** | Computes a deterministic SHA-256 hash of the cleaned dataset to act as a tamper-evident seal against silent data modification. | [`src/preprocessing.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/preprocessing.py) | `data_fingerprint()`, `save_reports()` |
| **Stage 3: Temporal Split** | Sorts each user's interactions chronologically using a stable sort; assigns the first $\lfloor 0.8 \cdot n \rfloor$ ratings to train and remaining to test. | [`src/splitting.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/splitting.py) | `temporal_train_test_split()` |
| **Stage 3: Matrix Build** | Constructs the dense training rating matrix $R_{\text{train}}$ of shape $(943, 1682)$, leaving unrated cells as $0.0$, and verifies sparsity (~$94.98\%$). | [`src/splitting.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/splitting.py) | `build_train_matrix()`, `compute_sparsity()`, `split_and_build_matrix()` |
| **Stage 4: Popularity Model** | Fits Bayesian weighted rating (IMDB formula) pulling movies with few ratings toward the global mean; broadcasts identical WR row to all users. | [`src/models/popularity.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/models/popularity.py) | `PopularityBaseline.fit()`, `PopularityBaseline.predict_matrix()` |
| **Stage 4: ItemKNN Model** | Computes pairwise cosine similarity between item column vectors, zeroes self-similarity, retains top-30 neighbors, and predicts ratings via vectorized neighbor aggregation. | [`src/models/item_knn.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/models/item_knn.py) | `ItemKNN.fit()`, `ItemKNN.predict_matrix()` |
| **Stage 4: SVD Model** | Centers ratings by subtracting user means over rated items only, extracts $k=20$ latent factors using `scipy.sparse.linalg.svds`, reconstructs scores, and restores user means. | [`src/models/svd.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/models/svd.py) | `SVDRecommender.fit()`, `SVDRecommender.predict_matrix()` |
| **Stage 5: Top-10 Masking** | Sets scores of all movies in that user's training set to $-\infty$, extracts top 10 candidates via `np.argpartition`, sorts descending, and verifies zero training item leakage. | [`src/evaluation/ranking.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/evaluation/ranking.py) | `compute_top10_recommendations()` |
| **Stage 5: Evaluation** | Evaluates RMSE on test pairs, Precision@10 & Recall@10 on relevant test items (rating $\ge 4.0$) across users with $\ge 1$ relevant item, and Catalog Coverage (%). | [`src/evaluation/metrics.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/evaluation/metrics.py) | `evaluate_models()` |
| **Stage 6: Visualizations** | Produces presentation-grade seaborn/matplotlib figures for rating distribution, log user activity, long-tail Pareto curves, and 2x2 model comparison grid. | [`src/plots.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/plots.py) | `plot_rating_distribution()`, `plot_ratings_per_user()`, `plot_item_long_tail()`, `plot_model_comparison()` |
| **Stage 6: Artifact Saving** | Persists metrics CSV, Top-10 recommendations pickle, train/test split CSVs, and movie catalog to `results/`. | [`run_experiments.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/run_experiments.py) | `stage_save()` |
| **Stage 7: Safety Check** | Asserts that refactored metrics match pre-refactor metrics to 6 decimal places, printing `REFACTOR CHECK PASSED`. | [`run_experiments.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/run_experiments.py) | `stage_check()` |
| **Interactive UI** | Multi-tab Streamlit dashboard: Tab 1 (EDA & SHA-256 integrity check), Tab 2 (Metrics table & 2x2 comparison chart), Tab 3 (User inspector with hit checkmarks). | [`app.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/app.py) | `load_cached_data()`, `main()` |

---

## 2. Where is Missing Value Handling?

In recommender systems, explicit feedback matrices are inherently sparse (~95% empty). Missing values represent **unobserved interactions** (the user has not watched or rated the movie), NOT zero-star ratings. We never impute or fabricate ratings:

1. **Matrix Construction** ([`src/splitting.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/splitting.py), function `build_train_matrix`):
   - Unrated cells are stored as `0.0` in $R_{\text{train}}$.
   - Code tag: `# [MISSING VALUES]`
2. **Popularity Baseline** ([`src/models/popularity.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/models/popularity.py), method `PopularityBaseline.fit`):
   - Unrated entries are filtered out when calculating movie vote counts $v = (R_{\text{train}} > 0).\text{sum}(\text{axis}=0)$ and movie means $R$.
   - Movies with zero training votes default strictly to the global catalog mean $C$.
3. **ItemKNN Collaborative Filtering** ([`src/models/item_knn.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/models/item_knn.py), method `ItemKNN.fit`):
   - User means are computed strictly over rated entries ($R_{\text{train}} > 0$).
   - If a user has not rated any of a movie's 30 nearest neighbors (denominator $= 0$), the model falls back safely to that user's personal mean rating.
4. **SVD Matrix Factorization** ([`src/models/svd.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/models/svd.py), method `SVDRecommender.fit`):
   - User means are computed only over rated items.
   - Missing entries are set to $0.0$ in the centered matrix, which represents neutral unobserved information in latent factor space.

---

## 3. Where are Outliers Handled?

Outliers are audited and reported in **Stage 2** ([`src/preprocessing.py`](file:///c:/Users/rutvi/OneDrive/Desktop/DA%20PROJECT/src/preprocessing.py), functions `detect_outliers()` and `outlier_summary()`):

- **User Activity Outliers**: Analyzes user rating counts via the Interquartile Range ($Q3 + 1.5 \cdot \text{IQR}$). Detects power users with 500+ ratings.
- **Movie Popularity Outliers**: Analyzes rating counts per movie via IQR. Detects blockbuster titles with 500+ ratings.
- **Handling Strategy (Report Only)**:
  In recommendation benchmarks, power users and blockbuster hits are **genuine human consumption behaviors**, not data errors. Deleting them would distort the true long-tail distribution and invalidate standard benchmark comparisons. Therefore, outliers are documented in `results/data_preprocessing_report.json` and printed to the console, while keeping all 100,000 ratings intact.
- Code tag: `# [OUTLIERS]`

---

## 4. What AI Model is Used?

**No pretrained AI models, Large Language Models (LLMs), or deep neural networks are used.**

This project uses **classical, interpretable recommendation algorithms trained strictly from scratch**:
1. **Popularity Baseline**: Bayesian probability / Empirical Bayes shrinkage (IMDB formula).
2. **ItemKNN**: Memory-based collaborative filtering using cosine vector similarity in Euclidean space with $k$-nearest neighbors.
3. **SVD Recommender**: Low-rank matrix factorization via Truncated Singular Value Decomposition (`scipy.sparse.linalg.svds`) extracting $k=20$ latent factors.

All algorithms are deterministic, fully vectorized using NumPy/SciPy, and trained purely on the local 79,619 training ratings without any external API calls or black-box embeddings.
