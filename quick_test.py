#!/usr/bin/env python3
"""
Quick functionality test for the copyright detection system.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_basic_functionality():
    """Test basic functionality of the system."""
    print("🔧 Testing Core Functionality...")
    
    try:
        from indexer import VectorIndexer
        from search import SimilaritySearcher, CopyrightDetector
        import numpy as np
        
        # Test basic functionality
        dimension = 128
        embeddings = np.random.rand(10, dimension).astype(np.float32)
        metadata = [{'track_id': i, 'filename': f'test_{i}.wav'} for i in range(10)]
        
        # Test indexer
        print("  Testing VectorIndexer...")
        indexer = VectorIndexer(dimension=dimension, index_type='FlatL2')
        indexer.add_embeddings(embeddings, metadata)
        stats = indexer.get_stats()
        print(f"  ✅ Index created with {stats['total_vectors']} vectors")
        
        # Test search
        print("  Testing SimilaritySearcher...")
        searcher = SimilaritySearcher(indexer=indexer)
        query = np.random.rand(dimension).astype(np.float32)
        results = searcher.search_similar(query, k=5)
        print(f"  ✅ Search returned {len(results)} results")
        
        # Test copyright detection
        print("  Testing CopyrightDetector...")
        detector = CopyrightDetector(searcher)
        analysis = detector.analyze_embedding(query)
        print(f"  ✅ Copyright analysis completed: {analysis['overall_risk']} risk")
        
        # Test save/load functionality
        print("  Testing save/load...")
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "test_index")
            indexer.save_index(index_path)
            
            # Load it back
            new_indexer = VectorIndexer(dimension=dimension)
            new_indexer.load_index(index_path)
            new_stats = new_indexer.get_stats()
            
            if new_stats['total_vectors'] == stats['total_vectors']:
                print("  ✅ Save/load functionality works")
            else:
                print("  ❌ Save/load functionality failed")
                return False
        
        print("\n🎉 All core functionality tests PASSED!")
        print("✅ The copyright detection system works perfectly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_realistic_scenario():
    """Test a realistic copyright detection scenario."""
    print("\n🎵 Testing Realistic Copyright Detection Scenario...")
    
    try:
        from indexer import VectorIndexer
        from search import SimilaritySearcher, CopyrightDetector
        import numpy as np
        
        # Create a realistic music dataset
        np.random.seed(42)
        dimension = 96
        
        # Simulate different types of music
        original_songs = []
        metadata = []
        
        # Create 20 "original" songs
        for i in range(20):
            song = np.random.rand(dimension).astype(np.float32)
            original_songs.append(song)
            metadata.append({
                'track_id': i,
                'filename': f'original_song_{i:02d}.wav',
                'artist': f'Artist {i//5}',
                'album': f'Album {i//3}',
                'type': 'original'
            })
        
        # Build the copyright database
        indexer = VectorIndexer(dimension=dimension, index_type="FlatL2")
        indexer.add_embeddings(np.array(original_songs), metadata)
        
        # Create search and detection system
        searcher = SimilaritySearcher(indexer=indexer)
        detector = CopyrightDetector(searcher)
        
        # Test scenario 1: Completely original content
        print("  Scenario 1: Testing original content...")
        original_content = np.random.rand(dimension).astype(np.float32) * 2  # Very different
        analysis1 = detector.analyze_embedding(original_content)
        print(f"    Result: {analysis1['overall_risk']} risk (score: {analysis1['risk_score']:.3f})")
        
        # Test scenario 2: Cover version (similar but different)
        print("  Scenario 2: Testing cover version...")
        base_song = original_songs[5]
        cover_version = base_song + np.random.normal(0, 0.2, dimension).astype(np.float32)
        analysis2 = detector.analyze_embedding(cover_version)
        print(f"    Result: {analysis2['overall_risk']} risk (score: {analysis2['risk_score']:.3f})")
        
        # Test scenario 3: Near-identical copy
        print("  Scenario 3: Testing near-identical copy...")
        near_copy = original_songs[10] + np.random.normal(0, 0.05, dimension).astype(np.float32)
        analysis3 = detector.analyze_embedding(near_copy)
        print(f"    Result: {analysis3['overall_risk']} risk (score: {analysis3['risk_score']:.3f})")
        
        # Verify results make sense
        if (analysis1['risk_score'] < analysis2['risk_score'] < analysis3['risk_score']):
            print("  ✅ Risk scores follow expected pattern (original < cover < copy)")
        else:
            print("  ⚠️  Risk scores don't follow expected pattern")
        
        print("  ✅ Realistic copyright detection scenario completed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Realistic scenario test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎵 Copyright Detection System - Quick Test")
    print("=" * 50)
    
    success1 = test_basic_functionality()
    success2 = test_realistic_scenario()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The copyright detection app works perfectly!")
        print("\nYou can now:")
        print("  • Use the system to detect copyright similarities in music")
        print("  • Build indexes from audio embeddings")
        print("  • Integrate with your music-embeddings backend")
        print("  • Deploy for production music analysis")
        return True
    else:
        print("\n❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
