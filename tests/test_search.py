"""
🎵 Test Suite for SimilaritySearcher

Comprehensive tests for the FAISS-based similarity search functionality.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import unittest
import tempfile
import shutil
import numpy as np
import os
from pathlib import Path
import sys

# Add src to path
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))

from indexer import VectorIndexer
from search import SimilaritySearcher, CopyrightDetector


class TestSimilaritySearcher(unittest.TestCase):
    """Test cases for the SimilaritySearcher class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dimension = 64  # Smaller for faster tests
        self.num_tracks = 20
        
        # Create test embeddings with known patterns
        np.random.seed(42)  # For reproducible tests
        self.test_embeddings = np.random.rand(self.num_tracks, self.test_dimension).astype(np.float32)
        
        # Create test metadata
        self.test_metadata = [
            {
                'filename': f'track_{i:03d}.wav',
                'artist': f'Artist_{i // 5}',  # 4 tracks per artist
                'genre': ['Rock', 'Jazz', 'Pop', 'Electronic'][i % 4],
                'track_id': i,
                'similarity_group': i // 5  # Groups of similar tracks
            }
            for i in range(self.num_tracks)
        ]
        
        # Create and populate indexer
        self.indexer = VectorIndexer(dimension=self.test_dimension)
        self.indexer.add_embeddings(self.test_embeddings, self.test_metadata)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self.cleanup_temp_dir)
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_searcher_initialization_with_indexer(self):
        """Test SimilaritySearcher initialization with indexer."""
        searcher = SimilaritySearcher(indexer=self.indexer)
        self.assertIsNotNone(searcher.indexer)
        self.assertEqual(searcher.indexer.get_stats()['total_vectors'], self.num_tracks)
    
    def test_searcher_initialization_with_index_path(self):
        """Test SimilaritySearcher initialization with index path."""
        # Save index first
        index_path = os.path.join(self.temp_dir, "test_search_index")
        self.indexer.save_index(index_path)
        
        # Create searcher from path
        searcher = SimilaritySearcher(index_path=index_path)
        self.assertIsNotNone(searcher.indexer)
        self.assertEqual(searcher.indexer.get_stats()['total_vectors'], self.num_tracks)
    
    def test_searcher_initialization_error(self):
        """Test error when no indexer or path provided."""
        with self.assertRaises(ValueError):
            SimilaritySearcher()
    
    def test_basic_similarity_search(self):
        """Test basic similarity search functionality."""
        searcher = SimilaritySearcher(indexer=self.indexer)
        
        # Use first embedding as query
        query_embedding = self.test_embeddings[0]
        results = searcher.search_similar(query_embedding, k=5)
        
        # Verify results structure
        self.assertEqual(len(results), 5)
        
        for i, result in enumerate(results):
            self.assertIn('rank', result)
            self.assertIn('index', result)
            self.assertIn('similarity_score', result)
            self.assertIn('distance', result)
            self.assertIn('filename', result)  # From metadata
            
            # Check rank order
            self.assertEqual(result['rank'], i + 1)
            
            # Check similarity score is reasonable
            self.assertGreaterEqual(result['similarity_score'], 0)
            self.assertLessEqual(result['similarity_score'], 1)
    
    def test_search_with_different_k_values(self):
        """Test search with different k values."""
        searcher = SimilaritySearcher(indexer=self.indexer)
        query_embedding = self.test_embeddings[0]
        
        # Test various k values
        for k in [1, 3, 5, 10, 20]:
            results = searcher.search_similar(query_embedding, k=k)
            expected_length = min(k, self.num_tracks)
            self.assertEqual(len(results), expected_length)
    
    def test_search_without_distances(self):
        """Test search without returning distances."""
        searcher = SimilaritySearcher(indexer=self.indexer)
        query_embedding = self.test_embeddings[0]
        
        results = searcher.search_similar(query_embedding, k=3, return_distances=False)
        
        for result in results:
            self.assertNotIn('distance', result)
            self.assertIn('similarity_score', result)
    
    def test_search_with_1d_query(self):
        """Test search with 1D query embedding."""
        searcher = SimilaritySearcher(indexer=self.indexer)
        query_embedding = self.test_embeddings[0].flatten()  # 1D
        
        results = searcher.search_similar(query_embedding, k=3)
        self.assertEqual(len(results), 3)
    
    def test_copyright_detection_basic(self):
        """Test basic copyright detection functionality."""
        searcher = SimilaritySearcher(indexer=self.indexer)
        
        # Create a query very similar to an existing track
        similar_query = self.test_embeddings[5] + np.random.normal(0, 0.01, self.test_dimension).astype(np.float32)
        
        matches = searcher.detect_copyright_matches(similar_query, similarity_threshold=0.7)
        
        # Should find some matches due to high similarity
        self.assertGreater(len(matches), 0)
        
        # Check match structure
        for match in matches:
            self.assertIn('copyright_risk', match)
            self.assertIn('match_confidence', match)
            self.assertIn('similarity_score', match)
            self.assertGreaterEqual(match['similarity_score'], 0.7)
    
    def test_copyright_detection_thresholds(self):
        """Test copyright detection with different thresholds."""
        searcher = SimilaritySearcher(indexer=self.indexer)
        query_embedding = self.test_embeddings[0]
        
        # Test with high threshold (should find fewer matches)
        high_threshold_matches = searcher.detect_copyright_matches(query_embedding, similarity_threshold=0.95)
        
        # Test with low threshold (should find more matches)
        low_threshold_matches = searcher.detect_copyright_matches(query_embedding, similarity_threshold=0.5)
        
        # Low threshold should find at least as many as high threshold
        self.assertGreaterEqual(len(low_threshold_matches), len(high_threshold_matches))
    
    def test_copyright_risk_levels(self):
        """Test copyright risk level assignment."""
        searcher = SimilaritySearcher(indexer=self.indexer)
        
        # Create queries with different similarity levels
        test_cases = [
            (self.test_embeddings[0], "VERY_HIGH"),  # Exact match
            (self.test_embeddings[0] + 0.1, "HIGH"),  # High similarity
            (self.test_embeddings[0] + 0.3, "MEDIUM"),  # Medium similarity
        ]
        
        for query, expected_min_risk in test_cases:
            matches = searcher.detect_copyright_matches(query, similarity_threshold=0.5)
            
            if matches:  # If any matches found
                highest_risk_match = max(matches, key=lambda x: x['similarity_score'])
                self.assertIn(highest_risk_match['copyright_risk'], 
                            ['VERY_HIGH', 'HIGH', 'MEDIUM'])
    
    def test_search_with_empty_index(self):
        """Test search behavior with empty index."""
        empty_indexer = VectorIndexer(dimension=self.test_dimension)
        searcher = SimilaritySearcher(indexer=empty_indexer)
        
        query_embedding = np.random.rand(self.test_dimension).astype(np.float32)
        
        with self.assertRaises(ValueError):
            searcher.search_similar(query_embedding, k=5)
    
    def test_get_statistics(self):
        """Test getting search statistics."""
        searcher = SimilaritySearcher(indexer=self.indexer)
        stats = searcher.get_statistics()
        
        # Verify statistics structure
        self.assertIn('total_vectors', stats)
        self.assertIn('dimension', stats)
        self.assertIn('search_ready', stats)
        self.assertIn('supports_batch_search', stats)
        self.assertIn('supports_copyright_detection', stats)
        
        # Verify values
        self.assertEqual(stats['total_vectors'], self.num_tracks)
        self.assertEqual(stats['dimension'], self.test_dimension)
        self.assertTrue(stats['search_ready'])
        self.assertTrue(stats['supports_batch_search'])
        self.assertTrue(stats['supports_copyright_detection'])


class TestCopyrightDetector(unittest.TestCase):
    """Test cases for the CopyrightDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dimension = 64
        self.num_tracks = 30
        
        # Create test data with copyright scenarios
        np.random.seed(123)
        self.test_embeddings = []
        self.test_metadata = []
        
        # Create original tracks
        for i in range(20):
            embedding = np.random.rand(self.test_dimension).astype(np.float32)
            self.test_embeddings.append(embedding)
            self.test_metadata.append({
                'filename': f'original_{i:03d}.wav',
                'artist': f'Original Artist {i}',
                'is_original': True,
                'track_id': i
            })
        
        # Create some similar tracks (potential copyright issues)
        for i in range(10):
            base_idx = i % 20
            similar_embedding = self.test_embeddings[base_idx] + np.random.normal(0, 0.1, self.test_dimension).astype(np.float32)
            self.test_embeddings.append(similar_embedding)
            self.test_metadata.append({
                'filename': f'similar_{i:03d}.wav',
                'artist': f'Cover Artist {i}',
                'is_original': False,
                'original_reference': base_idx,
                'track_id': 20 + i
            })
        
        self.test_embeddings = np.array(self.test_embeddings)
        
        # Create indexer and searcher
        self.indexer = VectorIndexer(dimension=self.test_dimension)
        self.indexer.add_embeddings(self.test_embeddings, self.test_metadata)
        self.searcher = SimilaritySearcher(indexer=self.indexer)
        self.detector = CopyrightDetector(self.searcher)
    
    def test_detector_initialization(self):
        """Test CopyrightDetector initialization."""
        self.assertIsNotNone(self.detector.searcher)
        self.assertEqual(self.detector.searcher.indexer.get_stats()['total_vectors'], 30)
    
    def test_analyze_track_structure(self):
        """Test the structure of track analysis results."""
        query_embedding = self.test_embeddings[0]
        analysis = self.detector.analyze_embedding(query_embedding)
        
        # Check required fields
        required_fields = [
            'total_similar_tracks', 'total_copyright_matches',
            'highest_similarity', 'average_similarity',
            'similar_tracks', 'copyright_matches',
            'overall_risk', 'risk_score'
        ]
        
        for field in required_fields:
            self.assertIn(field, analysis)
    
    def test_analyze_original_track(self):
        """Test analysis of an original track."""
        original_query = self.test_embeddings[5]  # An original track
        analysis = self.detector.analyze_embedding(original_query)
        
        # Should find itself and possibly similar tracks
        self.assertGreater(analysis['total_similar_tracks'], 0)
        self.assertGreaterEqual(analysis['highest_similarity'], 0.8)  # Should be high for self-match
    
    def test_analyze_copyright_infringing_track(self):
        """Test analysis of a potentially infringing track."""
        # Use one of the similar tracks we created
        infringing_query = self.test_embeddings[25]  # A similar track
        analysis = self.detector.analyze_embedding(infringing_query)
        
        # Should detect potential copyright issues
        self.assertGreater(analysis['total_copyright_matches'], 0)
        self.assertIn(analysis['overall_risk'], ['MEDIUM', 'HIGH', 'VERY_HIGH'])
    
    def test_analyze_completely_different_track(self):
        """Test analysis of a completely different track."""
        different_query = np.random.rand(self.test_dimension).astype(np.float32) + 10  # Very different
        analysis = self.detector.analyze_embedding(different_query)
        
        # Should have low risk
        self.assertEqual(analysis['overall_risk'], 'LOW')
        self.assertLessEqual(analysis['total_copyright_matches'], 2)  # Very few or no matches
    
    def test_risk_score_consistency(self):
        """Test that risk scores are consistent with risk levels."""
        test_queries = [
            self.test_embeddings[0],  # Original
            self.test_embeddings[25],  # Similar
            np.random.rand(self.test_dimension).astype(np.float32)  # Random
        ]
        
        for query in test_queries:
            analysis = self.detector.analyze_embedding(query)
            risk_score = analysis['risk_score']
            risk_level = analysis['overall_risk']
            
            # Verify risk score is between 0 and 1
            self.assertGreaterEqual(risk_score, 0)
            self.assertLessEqual(risk_score, 1)
            
            # Verify consistency between score and level
            if risk_level == 'VERY_HIGH':
                self.assertGreaterEqual(risk_score, 0.8)
            elif risk_level == 'HIGH':
                self.assertGreaterEqual(risk_score, 0.7)
            elif risk_level == 'LOW':
                self.assertLessEqual(risk_score, 0.8)


class TestSimilaritySearchIntegration(unittest.TestCase):
    """Integration tests for the complete similarity search system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self.cleanup_temp_dir)
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Step 1: Create a music collection
        dimension = 64
        genres = ['Rock', 'Jazz', 'Pop']
        tracks_per_genre = 10
        
        all_embeddings = []
        all_metadata = []
        
        np.random.seed(999)
        for genre_idx, genre in enumerate(genres):
            for track_idx in range(tracks_per_genre):
                # Create genre-specific patterns
                embedding = np.random.rand(dimension).astype(np.float32)
                embedding[genre_idx * 20:(genre_idx + 1) * 20] += 0.3  # Genre signature
                
                all_embeddings.append(embedding)
                all_metadata.append({
                    'filename': f'{genre.lower()}_{track_idx:02d}.wav',
                    'genre': genre,
                    'artist': f'{genre} Artist {track_idx // 3}',
                    'album': f'{genre} Album {track_idx // 5}'
                })
        
        all_embeddings = np.array(all_embeddings)
        
        # Step 2: Build index
        indexer = VectorIndexer(dimension=dimension, index_type="FlatL2")
        indexer.add_embeddings(all_embeddings, all_metadata)
        
        # Step 3: Save and reload index
        index_path = os.path.join(self.temp_dir, "integration_test")
        indexer.save_index(index_path)
        
        searcher = SimilaritySearcher(index_path=index_path)
        
        # Step 4: Test similarity search
        # Query with a Rock-like pattern
        rock_query = np.random.rand(dimension).astype(np.float32)
        rock_query[0:20] += 0.3  # Rock signature
        
        results = searcher.search_similar(rock_query, k=5)
        
        # Should find mostly Rock tracks
        rock_count = sum(1 for r in results if r['genre'] == 'Rock')
        self.assertGreaterEqual(rock_count, 3)  # At least 3 out of 5 should be Rock
        
        # Step 5: Test copyright detection
        detector = CopyrightDetector(searcher)
        
        # Test with a track very similar to an existing one
        similar_to_existing = all_embeddings[5] + np.random.normal(0, 0.05, dimension).astype(np.float32)
        analysis = detector.analyze_embedding(similar_to_existing)
        
        # Should detect potential copyright issues
        self.assertGreater(analysis['total_copyright_matches'], 0)
        self.assertIn(analysis['overall_risk'], ['MEDIUM', 'HIGH', 'VERY_HIGH'])
        
        # Step 6: Test with completely original content
        original_track = np.random.rand(dimension).astype(np.float32) + 5  # Very different
        original_analysis = detector.analyze_embedding(original_track)
        
        # Should have lower risk
        self.assertLessEqual(original_analysis['total_copyright_matches'], 2)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        import time
        
        # Create a larger dataset for performance testing
        dimension = 128
        num_tracks = 500
        
        embeddings = np.random.rand(num_tracks, dimension).astype(np.float32)
        metadata = [
            {'filename': f'perf_track_{i:04d}.wav', 'track_id': i}
            for i in range(num_tracks)
        ]
        
        # Build index
        start_time = time.time()
        indexer = VectorIndexer(dimension=dimension, index_type="FlatL2")
        indexer.add_embeddings(embeddings, metadata)
        build_time = time.time() - start_time
        
        # Test search performance
        searcher = SimilaritySearcher(indexer=indexer)
        query = np.random.rand(dimension).astype(np.float32)
        
        # Warm up
        searcher.search_similar(query, k=10)
        
        # Measure search time
        search_times = []
        for _ in range(10):
            start_time = time.time()
            searcher.search_similar(query, k=10)
            search_times.append(time.time() - start_time)
        
        avg_search_time = np.mean(search_times)
        
        # Performance assertions (generous bounds for CI)
        self.assertLess(build_time, 10.0)  # Should build in under 10 seconds
        self.assertLess(avg_search_time, 0.1)  # Should search in under 100ms
        
        print(f"Performance Results:")
        print(f"  Build time for {num_tracks} tracks: {build_time:.3f}s")
        print(f"  Average search time: {avg_search_time*1000:.2f}ms")


if __name__ == '__main__':
    unittest.main(verbosity=2)
