"""
🎵 Similarity Search Module

FAISS-based similarity search for audio embeddings.
Find similar tracks, detect potential copyright matches, and perform batch searches.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

try:
    from .indexer import VectorIndexer
except ImportError:
    from indexer import VectorIndexer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilaritySearcher:
    """
    A FAISS-based similarity searcher for audio embeddings.
    
    This class provides functionality to search for similar audio tracks,
    detect potential copyright matches, and perform batch similarity searches.
    """
    
    def __init__(self, index_path: Optional[str] = None, indexer: Optional[VectorIndexer] = None):
        """
        Initialize the SimilaritySearcher.
        
        Args:
            index_path (str, optional): Path to a saved index to load
            indexer (VectorIndexer, optional): An existing VectorIndexer instance
        """
        self.indexer = None
        
        if indexer is not None:
            self.indexer = indexer
        elif index_path is not None:
            self.load_index(index_path)
        else:
            raise ValueError("Either index_path or indexer must be provided")
    
    def load_index(self, index_path: str):
        """
        Load an index from disk.
        
        Args:
            index_path (str): Path to the saved index
        """
        # Extract dimension from saved metadata if possible
        metadata_path = str(Path(index_path)) + "_metadata.pkl"
        
        if Path(metadata_path).exists():
            import pickle
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                dimension = data.get('dimension', 128)  # Default fallback
        else:
            # Fallback dimension - this should be adjusted based on your embeddings
            logger.warning("No metadata found, using default dimension of 128")
            dimension = 128
        
        self.indexer = VectorIndexer(dimension=dimension)
        self.indexer.load_index(index_path)
        logger.info(f"Loaded index with {self.indexer.index.ntotal} vectors")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10, 
                      return_distances: bool = True) -> List[Dict]:
        """
        Search for similar embeddings in the index.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of similar results to return
            return_distances (bool): Whether to include distances in results
            
        Returns:
            List[Dict]: List of similar results with metadata and distances
        """
        if self.indexer is None or self.indexer.index is None:
            raise ValueError("No index loaded")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Perform search
        distances, indices = self.indexer.index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No more results
                break
                
            result = {
                'rank': i + 1,
                'index': int(idx),
                'similarity_score': float(1.0 / (1.0 + distance)),  # Convert distance to similarity
                'distance': float(distance)
            }
            
            # Add metadata if available
            if idx < len(self.indexer.metadata):
                result.update(self.indexer.metadata[idx])
            
            if not return_distances:
                result.pop('distance', None)
                
            results.append(result)
        
        return results
    
    def find_similar_tracks(self, audio_file: str, k: int = 10, 
                          music_embeddings_path: str = "../copyright-detector-music-embeddings",
                          model_name: str = "spectrogram") -> List[Dict]:
        """
        Find tracks similar to an input audio file.
        
        Args:
            audio_file (str): Path to the audio file
            k (int): Number of similar results to return
            music_embeddings_path (str): Path to the music embeddings project
            model_name (str): Model name for embedding extraction
            
        Returns:
            List[Dict]: List of similar tracks
        """
        import sys
        sys.path.append(music_embeddings_path)
        
        try:
            from src.embeddings import AudioEmbeddingExtractor
            
            # Extract embeddings from the query audio file
            extractor = AudioEmbeddingExtractor(model_name=model_name)
            query_embedding = extractor.extract_embeddings(audio_file)
            
            # If multiple embeddings are returned, use the mean
            if query_embedding.ndim > 1 and query_embedding.shape[0] > 1:
                query_embedding = np.mean(query_embedding, axis=0)
            
            # Search for similar tracks
            results = self.search_similar(query_embedding, k=k)
            
            # Add query file info to results
            for result in results:
                result['query_file'] = audio_file
                result['query_model'] = model_name
            
            return results
            
        except ImportError as e:
            logger.error(f"Failed to import music embeddings module: {e}")
            raise
    
    def detect_copyright_matches(self, query_embedding: np.ndarray, 
                               similarity_threshold: float = 0.8,
                               max_results: int = 50) -> List[Dict]:
        """
        Detect potential copyright matches based on similarity threshold.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            similarity_threshold (float): Minimum similarity score for matches
            max_results (int): Maximum number of results to check
            
        Returns:
            List[Dict]: List of potential copyright matches
        """
        # Search for similar tracks
        all_results = self.search_similar(query_embedding, k=max_results)
        
        # Filter by similarity threshold
        matches = [
            result for result in all_results 
            if result['similarity_score'] >= similarity_threshold
        ]
        
        # Add copyright match indicators
        for match in matches:
            if match['similarity_score'] >= 0.95:
                match['match_confidence'] = 'HIGH'
                match['copyright_risk'] = 'VERY_HIGH'
            elif match['similarity_score'] >= 0.85:
                match['match_confidence'] = 'MEDIUM'
                match['copyright_risk'] = 'HIGH'
            else:
                match['match_confidence'] = 'LOW'
                match['copyright_risk'] = 'MEDIUM'
        
        logger.info(f"Found {len(matches)} potential copyright matches")
        return matches
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 10) -> List[List[Dict]]:
        """
        Perform batch similarity search for multiple query embeddings.
        
        Args:
            query_embeddings (np.ndarray): Array of query embeddings
            k (int): Number of similar results per query
            
        Returns:
            List[List[Dict]]: List of search results for each query
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        results = []
        for i, query in enumerate(query_embeddings):
            query_results = self.search_similar(query, k=k)
            
            # Add batch info
            for result in query_results:
                result['batch_query_id'] = i
            
            results.append(query_results)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(query_embeddings)} batch queries")
        
        return results
    
    def find_duplicates(self, similarity_threshold: float = 0.95) -> List[Dict]:
        """
        Find potential duplicate tracks in the index.
        
        Args:
            similarity_threshold (float): Minimum similarity for duplicates
            
        Returns:
            List[Dict]: List of potential duplicate pairs
        """
        if self.indexer is None or self.indexer.index.ntotal == 0:
            raise ValueError("No index loaded or index is empty")
        
        duplicates = []
        checked_pairs = set()
        
        # Sample some vectors for duplicate detection (to avoid O(n²) complexity)
        sample_size = min(1000, self.indexer.index.ntotal)
        indices = np.random.choice(self.indexer.index.ntotal, sample_size, replace=False)
        
        for idx in indices:
            # Get the vector
            vector = self.indexer.index.reconstruct(int(idx)).reshape(1, -1)
            
            # Search for similar vectors
            results = self.search_similar(vector, k=10)
            
            for result in results[1:]:  # Skip the first result (itself)
                other_idx = result['index']
                
                # Create a unique pair identifier
                pair_id = tuple(sorted([idx, other_idx]))
                
                if (pair_id not in checked_pairs and 
                    result['similarity_score'] >= similarity_threshold):
                    
                    duplicate_info = {
                        'file1_index': idx,
                        'file2_index': other_idx,
                        'similarity_score': result['similarity_score'],
                    }
                    
                    # Add metadata if available
                    if idx < len(self.indexer.metadata):
                        duplicate_info['file1_metadata'] = self.indexer.metadata[idx]
                    if other_idx < len(self.indexer.metadata):
                        duplicate_info['file2_metadata'] = self.indexer.metadata[other_idx]
                    
                    duplicates.append(duplicate_info)
                    checked_pairs.add(pair_id)
        
        logger.info(f"Found {len(duplicates)} potential duplicate pairs")
        return duplicates
    
    def get_statistics(self) -> Dict:
        """
        Get search statistics and index information.
        
        Returns:
            Dict: Dictionary containing search statistics
        """
        if self.indexer is None:
            return {'status': 'No index loaded'}
        
        stats = self.indexer.get_stats()
        
        # Add search-specific stats
        stats.update({
            'search_ready': self.indexer.index is not None,
            'supports_batch_search': True,
            'supports_copyright_detection': True
        })
        
        return stats


class CopyrightDetector:
    """
    Specialized class for copyright detection using similarity search.
    """
    
    def __init__(self, searcher: SimilaritySearcher):
        """
        Initialize the CopyrightDetector.
        
        Args:
            searcher (SimilaritySearcher): Initialized similarity searcher
        """
        self.searcher = searcher
    
    def analyze_track(self, audio_file: str, 
                     music_embeddings_path: str = "../copyright-detector-music-embeddings",
                     model_name: str = "spectrogram") -> Dict:
        """
        Perform comprehensive copyright analysis on a track.
        
        Args:
            audio_file (str): Path to the audio file to analyze
            music_embeddings_path (str): Path to the music embeddings project
            model_name (str): Model name for embedding extraction
            
        Returns:
            Dict: Comprehensive copyright analysis results
        """
        # Find similar tracks
        similar_tracks = self.searcher.find_similar_tracks(
            audio_file, k=20, 
            music_embeddings_path=music_embeddings_path,
            model_name=model_name
        )
        
        # Extract the query embedding for copyright detection
        import sys
        sys.path.append(music_embeddings_path)
        
        try:
            from src.embeddings import AudioEmbeddingExtractor
            
            # Extract embeddings from the query audio file
            extractor = AudioEmbeddingExtractor(model_name=model_name)
            query_embedding = extractor.extract_embeddings(audio_file)
            
            # If multiple embeddings are returned, use the mean
            if query_embedding.ndim > 1 and query_embedding.shape[0] > 1:
                query_embedding = np.mean(query_embedding, axis=0)
            
        except ImportError as e:
            logger.error(f"Failed to import music embeddings module: {e}")
            raise ImportError(
                "Music embeddings module not found. Make sure it's installed and accessible. "
                "This feature requires the music-embeddings project to be integrated."
            )
        
        return self._analyze_embedding(query_embedding, similar_tracks)
    
    def analyze_embedding(self, query_embedding: np.ndarray) -> Dict:
        """
        Perform comprehensive copyright analysis on an embedding directly.
        
        Args:
            query_embedding (np.ndarray): The audio embedding to analyze
            
        Returns:
            Dict: Comprehensive copyright analysis results
        """
        # Find similar tracks using the embedding directly
        similar_tracks = self.searcher.search_similar(query_embedding, k=20)
        
        return self._analyze_embedding(query_embedding, similar_tracks)
    
    def _analyze_embedding(self, query_embedding: np.ndarray, similar_tracks: List[Dict]) -> Dict:
        """
        Internal method to perform copyright analysis on an embedding.
        
        Args:
            query_embedding (np.ndarray): The audio embedding to analyze
            similar_tracks (List[Dict]): List of similar tracks found
            
        Returns:
            Dict: Comprehensive copyright analysis results
        """
        # Detect copyright matches
        copyright_matches = self.searcher.detect_copyright_matches(query_embedding)
        
        # Analyze results
        analysis = {
            'query_embedding_shape': query_embedding.shape,
            'total_similar_tracks': len(similar_tracks),
            'total_copyright_matches': len(copyright_matches),
            'highest_similarity': max([t['similarity_score'] for t in similar_tracks]) if similar_tracks else 0,
            'average_similarity': np.mean([t['similarity_score'] for t in similar_tracks]) if similar_tracks else 0,
            'similar_tracks': similar_tracks,
            'copyright_matches': copyright_matches,
        }
        
        # Determine overall risk level
        if copyright_matches:
            highest_match = max(copyright_matches, key=lambda x: x['similarity_score'])
            analysis['overall_risk'] = highest_match['copyright_risk']
            analysis['risk_score'] = highest_match['similarity_score']
        else:
            analysis['overall_risk'] = 'LOW'
            analysis['risk_score'] = analysis['highest_similarity']
        
        return analysis
