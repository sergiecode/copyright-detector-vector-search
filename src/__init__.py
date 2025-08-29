"""
🎵 Copyright Detector Vector Search Module

A FAISS-based vector indexing and similarity search system for audio embeddings.
Perfect for building copyright detection systems and music similarity search engines.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

try:
    from .indexer import VectorIndexer
    from .search import SimilaritySearcher
except ImportError:
    from indexer import VectorIndexer
    from search import SimilaritySearcher

__version__ = "1.0.0"
__author__ = "Sergie Code"

__all__ = ["VectorIndexer", "SimilaritySearcher"]
