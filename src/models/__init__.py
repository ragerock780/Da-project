"""STAGE 4: RECOMMENDATION MODELS PACKAGE
========================================
Exposes the three recommendation algorithm classes:
1. PopularityBaseline (Bayesian weighted rating, IMDB-style)
2. ItemKNN (Item-based collaborative filtering with top-30 neighbors)
3. SVDRecommender (Truncated SVD matrix factorization with user-mean centering)
"""

from src.models.item_knn import ItemKNN
from src.models.popularity import PopularityBaseline
from src.models.svd import SVDRecommender

__all__ = ["PopularityBaseline", "ItemKNN", "SVDRecommender"]
