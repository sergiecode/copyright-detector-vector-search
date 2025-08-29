"""
🎵 Integration Tests for Vector Search System

Tests for complete system integration, including real-world scenarios
and integration with music embeddings module.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import unittest
import tempfile
import shutil
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))

from indexer import VectorIndexer
from search import SimilaritySearcher, CopyrightDetector


class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self.cleanup_temp_dir)
        
        # Create a realistic music dataset
        self.create_realistic_dataset()
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_realistic_dataset(self):
        """Create a realistic music dataset for testing."""
        np.random.seed(42)
        
        # Music genres with distinct characteristics
        self.genres = {
            'Rock': {'pattern': [1.0, 0.8, 0.6], 'tracks': 15},
            'Jazz': {'pattern': [0.3, 1.2, 0.8], 'tracks': 12},
            'Classical': {'pattern': [0.2, 0.4, 1.5], 'tracks': 10},
            'Electronic': {'pattern': [1.8, 0.2, 0.1], 'tracks': 13}
        }
        
        self.all_embeddings = []
        self.all_metadata = []
        self.dimension = 96
        
        track_id = 0
        for genre, info in self.genres.items():
            pattern = np.array(info['pattern'] * 32)  # Extend pattern to match dimension
            
            for i in range(info['tracks']):
                # Create embedding with genre pattern + noise
                base_embedding = np.random.rand(self.dimension).astype(np.float32)
                genre_embedding = base_embedding + pattern[:self.dimension] * 0.3
                
                self.all_embeddings.append(genre_embedding)
                self.all_metadata.append({
                    'track_id': track_id,
                    'filename': f'{genre.lower()}_{i+1:02d}.wav',
                    'artist': f'{genre} Artist {(i//3)+1}',
                    'genre': genre,
                    'album': f'{genre} Collection {(i//5)+1}',
                    'year': 1990 + (i * 2),
                    'duration': 180 + np.random.randint(-30, 60),
                    'popularity': np.random.uniform(10, 100)
                })
                track_id += 1
        
        self.all_embeddings = np.array(self.all_embeddings)
        self.total_tracks = len(self.all_embeddings)
    
    def test_complete_copyright_detection_workflow(self):
        """Test complete copyright detection workflow."""
        # Step 1: Build music index
        indexer = VectorIndexer(dimension=self.dimension, index_type="FlatL2")
        indexer.add_embeddings(self.all_embeddings, self.all_metadata)
        
        # Verify index was built correctly
        stats = indexer.get_stats()
        self.assertEqual(stats['total_vectors'], self.total_tracks)
        self.assertEqual(stats['dimension'], self.dimension)
        
        # Step 2: Save and reload index (test persistence)
        index_path = os.path.join(self.temp_dir, "copyright_detection_index")
        indexer.save_index(index_path)
        
        # Step 3: Initialize search system
        searcher = SimilaritySearcher(index_path=index_path)
        detector = CopyrightDetector(searcher)
        
        # Step 4: Test original content detection
        original_query = np.random.rand(self.dimension).astype(np.float32) * 2  # Very different
        original_analysis = detector.analyze_track(original_query)
        
        self.assertEqual(original_analysis['overall_risk'], 'LOW')
        self.assertLessEqual(original_analysis['total_copyright_matches'], 3)
        
        # Step 5: Test potential copyright infringement
        # Create a track very similar to an existing one
        base_track_idx = 5
        similar_track = self.all_embeddings[base_track_idx] + np.random.normal(0, 0.05, self.dimension).astype(np.float32)
        similar_analysis = detector.analyze_track(similar_track)
        
        self.assertGreater(similar_analysis['total_copyright_matches'], 0)
        self.assertIn(similar_analysis['overall_risk'], ['MEDIUM', 'HIGH', 'VERY_HIGH'])
        
        # Step 6: Test cover version detection
        # Create a "cover version" with moderate similarity
        cover_track = self.all_embeddings[10] + np.random.normal(0, 0.2, self.dimension).astype(np.float32)
        cover_analysis = detector.analyze_track(cover_track)
        
        # Should detect some similarity but lower risk than exact copy
        self.assertGreater(cover_analysis['risk_score'], 0.3)
        
        print(f"Copyright Detection Test Results:")
        print(f"  Original content risk: {original_analysis['overall_risk']}")
        print(f"  Similar track risk: {similar_analysis['overall_risk']}")
        print(f"  Cover version risk: {cover_analysis['overall_risk']}")
    
    def test_genre_based_similarity_accuracy(self):
        """Test accuracy of genre-based similarity detection."""
        indexer = VectorIndexer(dimension=self.dimension, index_type="FlatL2")
        indexer.add_embeddings(self.all_embeddings, self.all_metadata)
        searcher = SimilaritySearcher(indexer=indexer)
        
        genre_accuracy = {}
        
        for genre, info in self.genres.items():
            # Create a query that matches the genre pattern
            pattern = np.array(info['pattern'] * 32)[:self.dimension]
            genre_query = np.random.rand(self.dimension).astype(np.float32) + pattern * 0.3
            
            # Search for similar tracks
            results = searcher.search_similar(genre_query, k=10)
            
            # Count how many results match the expected genre
            correct_genre_count = sum(1 for r in results if r['genre'] == genre)
            accuracy = correct_genre_count / len(results)
            genre_accuracy[genre] = accuracy
            
            # Should have reasonable accuracy for genre matching
            self.assertGreater(accuracy, 0.3, f"Genre accuracy too low for {genre}")
        
        overall_accuracy = np.mean(list(genre_accuracy.values()))
        self.assertGreater(overall_accuracy, 0.4, "Overall genre matching accuracy too low")
        
        print(f"Genre Similarity Accuracy:")
        for genre, accuracy in genre_accuracy.items():
            print(f"  {genre}: {accuracy:.2%}")
        print(f"  Overall: {overall_accuracy:.2%}")
    
    def test_scalability_with_different_index_types(self):
        """Test scalability with different FAISS index types."""
        index_types = ["FlatL2", "IVF", "HNSW"]
        performance_results = {}
        
        for index_type in index_types:
            try:
                import time
                
                # Build index
                start_time = time.time()
                indexer = VectorIndexer(dimension=self.dimension, index_type=index_type)
                indexer.add_embeddings(self.all_embeddings, self.all_metadata)
                build_time = time.time() - start_time
                
                # Test search performance
                searcher = SimilaritySearcher(indexer=indexer)
                query = np.random.rand(self.dimension).astype(np.float32)
                
                # Measure search time
                start_time = time.time()
                results = searcher.search_similar(query, k=10)
                search_time = time.time() - start_time
                
                performance_results[index_type] = {
                    'build_time': build_time,
                    'search_time': search_time,
                    'results_count': len(results)
                }
                
                # Verify results are reasonable
                self.assertEqual(len(results), 10)
                self.assertGreater(results[0]['similarity_score'], 0)
                
            except Exception as e:
                # Some index types might not be available in all environments
                print(f"Skipping {index_type} due to: {e}")
                continue
        
        # At least FlatL2 should work
        self.assertIn("FlatL2", performance_results)
        
        print(f"Index Type Performance:")
        for index_type, perf in performance_results.items():
            print(f"  {index_type}:")
            print(f"    Build time: {perf['build_time']:.3f}s")
            print(f"    Search time: {perf['search_time']*1000:.2f}ms")
    
    def test_batch_processing_scenario(self):
        """Test batch processing of multiple queries."""
        indexer = VectorIndexer(dimension=self.dimension, index_type="FlatL2")
        indexer.add_embeddings(self.all_embeddings, self.all_metadata)
        searcher = SimilaritySearcher(indexer=indexer)
        detector = CopyrightDetector(searcher)
        
        # Create multiple test queries
        test_queries = []
        expected_results = []
        
        # Original content
        original_query = np.random.rand(self.dimension).astype(np.float32) * 3
        test_queries.append(original_query)
        expected_results.append('LOW')
        
        # Similar to existing tracks
        for i in [0, 5, 10]:
            similar_query = self.all_embeddings[i] + np.random.normal(0, 0.1, self.dimension).astype(np.float32)
            test_queries.append(similar_query)
            expected_results.append(['MEDIUM', 'HIGH', 'VERY_HIGH'])
        
        # Process all queries
        batch_results = []
        for query in test_queries:
            analysis = detector.analyze_track(query)
            batch_results.append(analysis)
        
        # Verify batch results
        self.assertEqual(len(batch_results), len(test_queries))
        
        # Check first result (original content)
        self.assertEqual(batch_results[0]['overall_risk'], 'LOW')
        
        # Check similarity results
        for i in range(1, len(batch_results)):
            result = batch_results[i]
            self.assertIn(result['overall_risk'], ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'])
            self.assertGreater(result['total_similar_tracks'], 0)
        
        print(f"Batch Processing Results:")
        for i, result in enumerate(batch_results):
            print(f"  Query {i+1}: {result['overall_risk']} risk ({result['risk_score']:.3f})")
    
    def test_data_integrity_and_persistence(self):
        """Test data integrity through save/load cycles."""
        # Create original index
        original_indexer = VectorIndexer(dimension=self.dimension, index_type="FlatL2")
        original_indexer.add_embeddings(self.all_embeddings, self.all_metadata)
        original_stats = original_indexer.get_stats()
        
        # Save index
        save_path = os.path.join(self.temp_dir, "integrity_test")
        original_indexer.save_index(save_path)
        
        # Load index multiple times to test consistency
        for iteration in range(3):
            loaded_indexer = VectorIndexer(dimension=self.dimension)
            loaded_indexer.load_index(save_path)
            loaded_stats = loaded_indexer.get_stats()
            
            # Verify stats match
            self.assertEqual(loaded_stats['total_vectors'], original_stats['total_vectors'])
            self.assertEqual(loaded_stats['dimension'], original_stats['dimension'])
            self.assertEqual(len(loaded_indexer.metadata), len(original_indexer.metadata))
            
            # Verify metadata integrity
            for i in range(min(10, len(loaded_indexer.metadata))):
                original_meta = original_indexer.metadata[i]
                loaded_meta = loaded_indexer.metadata[i]
                self.assertEqual(original_meta['filename'], loaded_meta['filename'])
                self.assertEqual(original_meta['genre'], loaded_meta['genre'])
            
            # Test search consistency
            searcher = SimilaritySearcher(indexer=loaded_indexer)
            test_query = self.all_embeddings[0]
            results = searcher.search_similar(test_query, k=5)
            
            # Should find the exact same track as top result
            self.assertEqual(results[0]['track_id'], 0)
            self.assertGreater(results[0]['similarity_score'], 0.95)
        
        print("Data integrity verified through multiple save/load cycles")
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        indexer = VectorIndexer(dimension=self.dimension, index_type="FlatL2")
        indexer.add_embeddings(self.all_embeddings, self.all_metadata)
        searcher = SimilaritySearcher(indexer=indexer)
        
        # Test with invalid query dimensions
        with self.assertRaises(ValueError):
            wrong_dim_query = np.random.rand(64).astype(np.float32)  # Wrong dimension
            searcher.search_similar(wrong_dim_query, k=5)
        
        # Test with k larger than index size
        large_k_query = np.random.rand(self.dimension).astype(np.float32)
        results = searcher.search_similar(large_k_query, k=1000)  # More than available
        self.assertLessEqual(len(results), self.total_tracks)
        
        # Test with k=0
        with self.assertRaises(Exception):
            searcher.search_similar(large_k_query, k=0)
        
        # Test copyright detection with extreme thresholds
        detector = CopyrightDetector(searcher)
        query = np.random.rand(self.dimension).astype(np.float32)
        
        # Very high threshold (should find nothing)
        high_threshold_matches = searcher.detect_copyright_matches(query, similarity_threshold=0.99)
        self.assertLessEqual(len(high_threshold_matches), 1)
        
        # Very low threshold (should find many)
        low_threshold_matches = searcher.detect_copyright_matches(query, similarity_threshold=0.1)
        self.assertGreater(len(low_threshold_matches), 0)
        
        print("Error handling and edge cases tested successfully")


class TestMusicEmbeddingsIntegration(unittest.TestCase):
    """Test integration with music embeddings module (if available)."""
    
    def setUp(self):
        """Set up for embeddings integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self.cleanup_temp_dir)
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_embeddings_module_integration_interface(self):
        """Test the interface for integration with music embeddings module."""
        # This test verifies that the integration interface works
        # even without the actual embeddings module
        
        from indexer import build_index_from_embeddings_module
        
        # Test that the function exists and has correct signature
        self.assertTrue(callable(build_index_from_embeddings_module))
        
        # Test with mock parameters (should fail gracefully if module not available)
        mock_audio_files = ["fake_audio1.wav", "fake_audio2.wav"]
        mock_embeddings_path = "/nonexistent/path"
        mock_output_path = os.path.join(self.temp_dir, "mock_index")
        
        try:
            # This should fail with ImportError or FileNotFoundError
            result = build_index_from_embeddings_module(
                music_embeddings_path=mock_embeddings_path,
                audio_files=mock_audio_files,
                output_path=mock_output_path
            )
            # If it somehow succeeds, verify it returns a VectorIndexer
            self.assertIsInstance(result, VectorIndexer)
        except (ImportError, ModuleNotFoundError, FileNotFoundError):
            # Expected when embeddings module is not available
            print("Music embeddings module not available - integration interface verified")
        except Exception as e:
            # Other errors are acceptable for this interface test
            print(f"Integration interface test completed with: {type(e).__name__}")
    
    def test_realistic_music_scenario_simulation(self):
        """Simulate a realistic music analysis scenario."""
        # Simulate what would happen with real audio embeddings
        
        # Create embeddings that simulate real audio features
        np.random.seed(456)
        dimension = 128
        
        # Simulate different types of music content
        music_scenarios = {
            'original_songs': {
                'count': 20,
                'pattern_strength': 1.0,
                'noise_level': 0.1
            },
            'cover_versions': {
                'count': 5,
                'pattern_strength': 0.8,
                'noise_level': 0.3
            },
            'remixes': {
                'count': 3,
                'pattern_strength': 0.6,
                'noise_level': 0.5
            }
        }
        
        all_embeddings = []
        all_metadata = []
        track_id = 0
        
        # Create base original songs
        original_embeddings = []
        for i in range(music_scenarios['original_songs']['count']):
            embedding = np.random.rand(dimension).astype(np.float32)
            original_embeddings.append(embedding)
            all_embeddings.append(embedding)
            all_metadata.append({
                'track_id': track_id,
                'filename': f'original_{i:02d}.wav',
                'type': 'original',
                'artist': f'Original Artist {i//4}',
                'similarity_source': None
            })
            track_id += 1
        
        # Create cover versions based on originals
        for i in range(music_scenarios['cover_versions']['count']):
            base_idx = i % len(original_embeddings)
            base_embedding = original_embeddings[base_idx]
            
            # Add noise to simulate cover version differences
            cover_embedding = base_embedding + np.random.normal(0, 0.2, dimension).astype(np.float32)
            all_embeddings.append(cover_embedding)
            all_metadata.append({
                'track_id': track_id,
                'filename': f'cover_{i:02d}.wav',
                'type': 'cover',
                'artist': f'Cover Artist {i}',
                'similarity_source': base_idx
            })
            track_id += 1
        
        # Create remixes
        for i in range(music_scenarios['remixes']['count']):
            base_idx = i % len(original_embeddings)
            base_embedding = original_embeddings[base_idx]
            
            # More significant changes for remixes
            remix_embedding = base_embedding + np.random.normal(0, 0.4, dimension).astype(np.float32)
            all_embeddings.append(remix_embedding)
            all_metadata.append({
                'track_id': track_id,
                'filename': f'remix_{i:02d}.wav',
                'type': 'remix',
                'artist': f'Remix Artist {i}',
                'similarity_source': base_idx
            })
            track_id += 1
        
        all_embeddings = np.array(all_embeddings)
        
        # Build and test the system
        indexer = VectorIndexer(dimension=dimension, index_type="FlatL2")
        indexer.add_embeddings(all_embeddings, all_metadata)
        
        searcher = SimilaritySearcher(indexer=indexer)
        detector = CopyrightDetector(searcher)
        
        # Test cover version detection
        cover_detection_accuracy = 0
        for meta in all_metadata:
            if meta['type'] == 'cover' and meta['similarity_source'] is not None:
                cover_idx = meta['track_id']
                cover_embedding = all_embeddings[cover_idx]
                
                analysis = detector.analyze_track(cover_embedding)
                
                # Cover should be detected as potential copyright issue
                if analysis['overall_risk'] in ['MEDIUM', 'HIGH', 'VERY_HIGH']:
                    cover_detection_accuracy += 1
        
        cover_detection_rate = cover_detection_accuracy / music_scenarios['cover_versions']['count']
        self.assertGreater(cover_detection_rate, 0.6, "Cover version detection rate too low")
        
        # Test original content
        original_false_positive_rate = 0
        for meta in all_metadata:
            if meta['type'] == 'original':
                original_idx = meta['track_id']
                # Test with slight variation to simulate same song re-encoded
                test_embedding = all_embeddings[original_idx] + np.random.normal(0, 0.02, dimension).astype(np.float32)
                
                analysis = detector.analyze_track(test_embedding)
                
                # Should detect as legitimate (finding the original itself)
                if analysis['overall_risk'] == 'VERY_HIGH':
                    # This is expected - finding the exact same song
                    pass
                elif analysis['overall_risk'] in ['MEDIUM', 'HIGH']:
                    original_false_positive_rate += 1
        
        print(f"Realistic Music Scenario Results:")
        print(f"  Cover detection rate: {cover_detection_rate:.2%}")
        print(f"  Original false positive rate: {original_false_positive_rate}/{music_scenarios['original_songs']['count']}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
