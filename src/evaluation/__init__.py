"""STAGE 5: EVALUATION PACKAGE
============================
Exposes ranking and metrics evaluation functions:
- compute_top10_recommendations: Generates Top-10 lists with train items masked with -inf
- evaluate_models: Computes RMSE, Precision@10, Recall@10, and Catalog Coverage
"""

from src.evaluation.metrics import evaluate_models
from src.evaluation.ranking import compute_top10_recommendations

__all__ = ["compute_top10_recommendations", "evaluate_models"]
