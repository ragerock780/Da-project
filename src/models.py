"""BACKWARD COMPATIBILITY ADAPTER FOR MODELS MODULE
===================================================
File: src/models.py

Re-exports models from the src.models package:
- PopularityBaseline from src.models.popularity
- ItemKNN from src.models.item_knn
- SVDRecommender from src.models.svd
"""

from src.models.item_knn import ItemKNN
from src.models.popularity import PopularityBaseline
from src.models.svd import SVDRecommender

__all__ = ["PopularityBaseline", "ItemKNN", "SVDRecommender"]
