"""
🎵 Test Script for Vector Search Module

Quick test to verify that the vector search module is working correctly.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import os
import sys
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.indexer import VectorIndexer
    from src.search import SimilaritySearcher
    print("✅ Successfully imported vector search modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic indexing and search functionality."""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 35)
    
    try:
        # Create test data
        num_tracks = 10
        embedding_dim = 64
        embeddings = np.random.rand(num_tracks, embedding_dim).astype(np.float32)
        metadata = [{'filename': f'test_track_{i}.wav', 'id': i} for i in range(num_tracks)]
        
        # Test indexing
        print("📦 Testing indexing...")
        indexer = VectorIndexer(dimension=embedding_dim, index_type="FlatL2")
        indexer.add_embeddings(embeddings, metadata)
        
        stats = indexer.get_stats()
        print(f"   Created index with {stats['total_vectors']} vectors")
        
        # Test searching
        print("🔍 Testing search...")
        searcher = SimilaritySearcher(indexer=indexer)
        query = np.random.rand(embedding_dim).astype(np.float32)
        results = searcher.search_similar(query, k=3)
        
        print(f"   Found {len(results)} similar tracks")
        for i, result in enumerate(results):
            print(f"   {i+1}. {result['filename']} (similarity: {result['similarity_score']:.3f})")
        
        print("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_save_load_index():
    """Test saving and loading indexes."""
    print("\n🧪 Testing Save/Load Functionality")
    print("=" * 38)
    
    try:
        # Create and save index
        embedding_dim = 32
        embeddings = np.random.rand(5, embedding_dim).astype(np.float32)
        metadata = [{'filename': f'save_test_{i}.wav'} for i in range(5)]
        
        print("💾 Testing save...")
        indexer = VectorIndexer(dimension=embedding_dim)
        indexer.add_embeddings(embeddings, metadata)
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        test_path = "data/test_index"
        indexer.save_index(test_path)
        print(f"   Saved index to {test_path}")
        
        print("📂 Testing load...")
        new_indexer = VectorIndexer(dimension=embedding_dim)
        new_indexer.load_index(test_path)
        
        stats = new_indexer.get_stats()
        print(f"   Loaded index with {stats['total_vectors']} vectors")
        
        # Clean up test files
        try:
            os.remove(f"{test_path}.faiss")
            os.remove(f"{test_path}_metadata.pkl")
            print("   Cleaned up test files")
        except:
            pass
        
        print("✅ Save/Load functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Save/Load functionality test failed: {e}")
        return False


def test_copyright_detection():
    """Test copyright detection functionality."""
    print("\n🧪 Testing Copyright Detection")
    print("=" * 33)
    
    try:
        # Create test data with similar embeddings
        embedding_dim = 32
        base_embedding = np.random.rand(embedding_dim).astype(np.float32)
        
        embeddings = []
        metadata = []
        
        # Add original track
        embeddings.append(base_embedding)
        metadata.append({'filename': 'original.wav', 'is_copyrighted': True})
        
        # Add similar track (potential copyright issue)
        similar_embedding = base_embedding + np.random.normal(0, 0.05, embedding_dim).astype(np.float32)
        embeddings.append(similar_embedding)
        metadata.append({'filename': 'similar.wav', 'is_copyrighted': False})
        
        # Add different tracks
        for i in range(3):
            different_embedding = np.random.rand(embedding_dim).astype(np.float32)
            embeddings.append(different_embedding)
            metadata.append({'filename': f'different_{i}.wav', 'is_copyrighted': False})
        
        # Build index
        all_embeddings = np.array(embeddings)
        indexer = VectorIndexer(dimension=embedding_dim)
        indexer.add_embeddings(all_embeddings, metadata)
        
        # Test copyright detection
        print("⚖️  Testing copyright detection...")
        searcher = SimilaritySearcher(indexer=indexer)
        
        # Use the similar track as query
        query_embedding = similar_embedding
        matches = searcher.detect_copyright_matches(query_embedding, similarity_threshold=0.7)
        
        print(f"   Found {len(matches)} potential copyright matches")
        for match in matches:
            print(f"   - {match['filename']}: {match['similarity_score']:.3f} ({match['copyright_risk']})")
        
        print("✅ Copyright detection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Copyright detection test failed: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking Dependencies")
    print("=" * 25)
    
    dependencies = [
        ('numpy', 'numpy'),
        ('faiss', 'faiss'),
        ('pickle', 'pickle'),
        ('pathlib', 'pathlib')
    ]
    
    all_good = True
    
    for dep_name, import_name in dependencies:
        try:
            if import_name == 'numpy':
                import numpy
                print(f"✅ {dep_name} - version {numpy.__version__}")
            elif import_name == 'faiss':
                import faiss
                print(f"✅ {dep_name} - version {faiss.__version__}")
            else:
                __import__(import_name)
                print(f"✅ {dep_name}")
        except ImportError:
            print(f"❌ {dep_name} - Not installed")
            all_good = False
    
    return all_good


if __name__ == "__main__":
    print("🎵 Vector Search Module Test Suite")
    print("Created by Sergie Code - AI Tools for Musicians")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Some dependencies are missing. Please run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Run tests
    tests = [
        test_basic_functionality,
        test_save_load_index,
        test_copyright_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your vector search module is ready to use.")
        print("\n📚 Next steps:")
        print("   1. Run: python examples/build_index_example.py")
        print("   2. Run: python examples/search_example.py")
        print("   3. Try with your own audio files and embeddings")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)
