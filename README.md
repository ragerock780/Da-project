# Comparing Recommendation Algorithms on MovieLens 100K

A clean, reproducible benchmark comparing three core recommendation algorithms on MovieLens 100K using an evaluation protocol with per-user temporal train/test splitting.

## Methodology
1. **Data & Temporal Split**: MovieLens 100K ratings are sorted chronologically per user; the first 80% of each user's interactions form the training set while the remaining 20% form the test set.
2. **Popularity Baseline**: Computes Bayesian weighted ratings (IMDB formula with 75th percentile threshold) to rank movies globally.
3. **ItemKNN**: Item-based collaborative filtering using cosine similarity over rating vectors, keeping the top 30 neighbors per movie with user-mean fallback.
4. **SVD Recommender**: Matrix factorization via truncated SVD ($k=20$) on mean-centered user ratings with user-mean restoration.
5. **Evaluation Protocol**: Evaluates RMSE on test ratings alongside top-10 ranking metrics (Precision@10, Recall@10 on test ratings $\ge 4.0$) and Catalog Coverage (%).

## Setup & Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Benchmark Experiments
```bash
python run_experiments.py
```
This automatically downloads the dataset if missing, fits all models, evaluates metrics, and saves outputs to `results/`.

### 3. Launch Streamlit UI
```bash
streamlit run app.py
```
(Or `python -m streamlit run app.py`)
