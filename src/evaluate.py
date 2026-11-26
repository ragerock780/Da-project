"""BACKWARD COMPATIBILITY ADAPTER FOR EVALUATION MODULE
=======================================================
File: src/evaluate.py

Re-exports evaluation functions from the src.evaluation package:
- compute_top10_recommendations from src.evaluation.ranking
- evaluate_models from src.evaluation.metrics
"""

from src.evaluation.metrics import evaluate_models
from src.evaluation.ranking import compute_top10_recommendations

__all__ = ["compute_top10_recommendations", "evaluate_models"]
