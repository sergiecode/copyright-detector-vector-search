"""
🎵 Test Suite for VectorIndexer

Comprehensive tests for the FAISS-based vector indexing functionality.

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


class TestVectorIndexer(unittest.TestCase):
    """Test cases for the VectorIndexer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dimension = 128
        self.test_embeddings = np.random.rand(10, self.test_dimension).astype(np.float32)
        self.test_metadata = [
            {
                'filename': f'test_track_{i}.wav',
                'artist': f'Test Artist {i}',
                'genre': 'Test Genre',
                'track_id': i
            }
            for i in range(10)
        ]
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self.cleanup_temp_dir)
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_indexer_initialization(self):
        """Test proper initialization of VectorIndexer."""
        # Test FlatL2 index
        indexer = VectorIndexer(dimension=self.test_dimension, index_type="FlatL2")
        self.assertEqual(indexer.dimension, self.test_dimension)
        self.assertEqual(indexer.index_type, "FlatL2")
        self.assertEqual(indexer.metric, "L2")
        self.assertTrue(indexer.is_trained)  # FlatL2 doesn't need training
        self.assertIsNotNone(indexer.index)
        
        # Test IVF index
        indexer_ivf = VectorIndexer(dimension=self.test_dimension, index_type="IVF")
        self.assertEqual(indexer_ivf.index_type, "IVF")
        self.assertFalse(indexer_ivf.is_trained)  # IVF needs training
        
        # Test HNSW index
        indexer_hnsw = VectorIndexer(dimension=self.test_dimension, index_type="HNSW")
        self.assertEqual(indexer_hnsw.index_type, "HNSW")
        self.assertTrue(indexer_hnsw.is_trained)  # HNSW doesn't need training
    
    def test_invalid_index_type(self):
        """Test error handling for invalid index types."""
        with self.assertRaises(ValueError):
            VectorIndexer(dimension=128, index_type="INVALID_TYPE")
    
    def test_add_embeddings(self):
        """Test adding embeddings to the index."""
        indexer = VectorIndexer(dimension=self.test_dimension)
        
        # Test successful addition
        indexer.add_embeddings(self.test_embeddings, self.test_metadata)
        
        stats = indexer.get_stats()
        self.assertEqual(stats['total_vectors'], 10)
        self.assertEqual(len(indexer.metadata), 10)
        
        # Test metadata content
        self.assertEqual(indexer.metadata[0]['filename'], 'test_track_0.wav')
        self.assertEqual(indexer.metadata[5]['artist'], 'Test Artist 5')
    
    def test_add_embeddings_validation(self):
        """Test validation when adding embeddings."""
        indexer = VectorIndexer(dimension=self.test_dimension)
        
        # Test mismatched embedding and metadata count
        with self.assertRaises(ValueError):
            indexer.add_embeddings(self.test_embeddings, self.test_metadata[:-1])
        
        # Test wrong embedding dimension
        wrong_dim_embeddings = np.random.rand(5, 64).astype(np.float32)
        with self.assertRaises(ValueError):
            indexer.add_embeddings(wrong_dim_embeddings, self.test_metadata[:5])
    
    def test_save_and_load_index(self):
        """Test saving and loading index functionality."""
        # Create and populate index
        indexer = VectorIndexer(dimension=self.test_dimension)
        indexer.add_embeddings(self.test_embeddings, self.test_metadata)
        
        # Save index
        save_path = os.path.join(self.temp_dir, "test_index")
        indexer.save_index(save_path)
        
        # Verify files were created
        self.assertTrue(os.path.exists(f"{save_path}.faiss"))
        self.assertTrue(os.path.exists(f"{save_path}_metadata.pkl"))
        
        # Load index into new indexer
        new_indexer = VectorIndexer(dimension=self.test_dimension)
        new_indexer.load_index(save_path)
        
        # Verify loaded index
        self.assertEqual(new_indexer.get_stats()['total_vectors'], 10)
        self.assertEqual(len(new_indexer.metadata), 10)
        self.assertEqual(new_indexer.metadata[0]['filename'], 'test_track_0.wav')
    
    def test_save_empty_index(self):
        """Test error when trying to save empty index."""
        indexer = VectorIndexer(dimension=self.test_dimension)
        save_path = os.path.join(self.temp_dir, "empty_index")
        
        with self.assertRaises(ValueError):
            indexer.save_index(save_path)
    
    def test_load_nonexistent_index(self):
        """Test error when trying to load nonexistent index."""
        indexer = VectorIndexer(dimension=self.test_dimension)
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent")
        
        with self.assertRaises(FileNotFoundError):
            indexer.load_index(nonexistent_path)
    
    def test_get_stats(self):
        """Test getting index statistics."""
        indexer = VectorIndexer(dimension=self.test_dimension)
        
        # Test empty index stats
        stats = indexer.get_stats()
        self.assertEqual(stats['total_vectors'], 0)
        self.assertEqual(stats['dimension'], self.test_dimension)
        self.assertEqual(stats['metadata_count'], 0)
        
        # Test populated index stats
        indexer.add_embeddings(self.test_embeddings, self.test_metadata)
        stats = indexer.get_stats()
        self.assertEqual(stats['total_vectors'], 10)
        self.assertEqual(stats['metadata_count'], 10)
        self.assertTrue(stats['is_trained'])
    
    def test_ivf_training(self):
        """Test IVF index training."""
        indexer = VectorIndexer(dimension=self.test_dimension, index_type="IVF")
        
        # Should not be trained initially
        self.assertFalse(indexer.is_trained)
        
        # Add embeddings (this should trigger training)
        indexer.add_embeddings(self.test_embeddings, self.test_metadata)
        
        # Should be trained after adding embeddings
        self.assertTrue(indexer.is_trained)
        self.assertEqual(indexer.get_stats()['total_vectors'], 10)
    
    def test_optimize_index(self):
        """Test index optimization."""
        indexer = VectorIndexer(dimension=self.test_dimension, index_type="IVF")
        indexer.add_embeddings(self.test_embeddings, self.test_metadata)
        
        # Test optimization (should not raise any errors)
        try:
            indexer.optimize_index()
        except Exception as e:
            self.fail(f"Index optimization failed: {e}")
    
    def test_multiple_additions(self):
        """Test adding embeddings in multiple batches."""
        indexer = VectorIndexer(dimension=self.test_dimension)
        
        # Add first batch
        batch1_embeddings = self.test_embeddings[:5]
        batch1_metadata = self.test_metadata[:5]
        indexer.add_embeddings(batch1_embeddings, batch1_metadata)
        
        self.assertEqual(indexer.get_stats()['total_vectors'], 5)
        
        # Add second batch
        batch2_embeddings = self.test_embeddings[5:]
        batch2_metadata = self.test_metadata[5:]
        indexer.add_embeddings(batch2_embeddings, batch2_metadata)
        
        self.assertEqual(indexer.get_stats()['total_vectors'], 10)
        self.assertEqual(len(indexer.metadata), 10)
    
    def test_large_batch_processing(self):
        """Test handling larger batches of embeddings."""
        indexer = VectorIndexer(dimension=64)  # Smaller dimension for speed
        
        # Create larger test data
        large_embeddings = np.random.rand(1000, 64).astype(np.float32)
        large_metadata = [
            {'filename': f'large_track_{i}.wav', 'track_id': i}
            for i in range(1000)
        ]
        
        # Add large batch
        indexer.add_embeddings(large_embeddings, large_metadata)
        
        stats = indexer.get_stats()
        self.assertEqual(stats['total_vectors'], 1000)
        self.assertEqual(len(indexer.metadata), 1000)


class TestVectorIndexerIntegration(unittest.TestCase):
    """Integration tests for VectorIndexer with realistic scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self.cleanup_temp_dir)
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_music_collection_scenario(self):
        """Test a realistic music collection scenario."""
        # Simulate a music collection with different genres
        genres = ["Rock", "Jazz", "Classical", "Electronic"]
        all_embeddings = []
        all_metadata = []
        
        # Create embeddings with genre-specific patterns
        for genre_idx, genre in enumerate(genres):
            for track_idx in range(25):  # 25 tracks per genre
                # Create embeddings with slight genre-based patterns
                base_pattern = np.random.rand(128).astype(np.float32)
                base_pattern[genre_idx * 32:(genre_idx + 1) * 32] += 0.5  # Genre signature
                
                all_embeddings.append(base_pattern)
                all_metadata.append({
                    'filename': f'{genre.lower()}_track_{track_idx:03d}.wav',
                    'artist': f'{genre} Artist {track_idx // 5 + 1}',
                    'genre': genre,
                    'album': f'{genre} Album {track_idx // 10 + 1}',
                    'track_id': len(all_embeddings)
                })
        
        all_embeddings = np.array(all_embeddings)
        
        # Build index
        indexer = VectorIndexer(dimension=128, index_type="FlatL2")
        indexer.add_embeddings(all_embeddings, all_metadata)
        
        # Verify collection was indexed correctly
        stats = indexer.get_stats()
        self.assertEqual(stats['total_vectors'], 100)  # 25 tracks × 4 genres
        
        # Save and reload to test persistence
        save_path = os.path.join(self.temp_dir, "music_collection")
        indexer.save_index(save_path)
        
        # Load in new indexer
        new_indexer = VectorIndexer(dimension=128)
        new_indexer.load_index(save_path)
        
        # Verify loaded collection
        new_stats = new_indexer.get_stats()
        self.assertEqual(new_stats['total_vectors'], 100)
        
        # Verify metadata integrity
        rock_tracks = [meta for meta in new_indexer.metadata if meta['genre'] == 'Rock']
        self.assertEqual(len(rock_tracks), 25)


if __name__ == '__main__':
    unittest.main(verbosity=2)
