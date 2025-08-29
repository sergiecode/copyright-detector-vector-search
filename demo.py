#!/usr/bin/env python3
"""
🎵 Copyright Detection System - Usage Demo

Demonstrates how to use the copyright detection system for music analysis.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from indexer import VectorIndexer
from search import SimilaritySearcher, CopyrightDetector

def demo_basic_usage():
    """Demonstrate basic usage of the copyright detection system."""
    print("🎵 Copyright Detection System - Usage Demo")
    print("=" * 50)
    
    print("\n1️⃣ Creating a music database...")
    
    # Step 1: Create a VectorIndexer
    dimension = 128  # Typical dimension for audio embeddings
    indexer = VectorIndexer(dimension=dimension, index_type="FlatL2")
    
    # Step 2: Simulate a music database with embeddings
    # In real usage, these would come from your audio embedding extraction
    music_database = []
    metadata_database = []
    
    # Add some "songs" to the database
    np.random.seed(42)  # For reproducible results
    
    artists = ["The Beatles", "Queen", "Led Zeppelin", "Pink Floyd", "The Rolling Stones"]
    albums = ["Abbey Road", "Bohemian Rhapsody", "IV", "Dark Side", "Sticky Fingers"]
    
    for i in range(20):
        # Create a random embedding (in practice, this comes from audio analysis)
        embedding = np.random.rand(dimension).astype(np.float32)
        music_database.append(embedding)
        
        # Create metadata for the track
        metadata = {
            'track_id': i,
            'filename': f'song_{i:02d}.wav',
            'artist': artists[i % len(artists)],
            'album': albums[i % len(albums)],
            'title': f'Song {i+1}',
            'year': 1960 + (i * 2),
            'duration': 180 + np.random.randint(-30, 60)
        }
        metadata_database.append(metadata)
    
    # Add all embeddings to the index
    music_database = np.array(music_database)
    indexer.add_embeddings(music_database, metadata_database)
    
    print(f"✅ Created database with {len(music_database)} songs")
    
    print("\n2️⃣ Setting up search and detection...")
    
    # Step 3: Create searcher and detector
    searcher = SimilaritySearcher(indexer=indexer)
    detector = CopyrightDetector(searcher)
    
    print("✅ Search and detection systems ready")
    
    print("\n3️⃣ Testing different scenarios...")
    
    # Scenario 1: Completely original content
    print("\n🎨 Scenario 1: Original content")
    original_embedding = np.random.rand(dimension).astype(np.float32) * 2  # Very different
    analysis = detector.analyze_embedding(original_embedding)
    
    print(f"   Risk Level: {analysis['overall_risk']}")
    print(f"   Risk Score: {analysis['risk_score']:.3f}")
    print(f"   Similar Tracks: {analysis['total_similar_tracks']}")
    print(f"   Copyright Matches: {analysis['total_copyright_matches']}")
    
    # Scenario 2: Cover version (similar but not identical)
    print("\n🎭 Scenario 2: Cover version")
    base_song = music_database[5]  # Take an existing song
    cover_embedding = base_song + np.random.normal(0, 0.3, dimension).astype(np.float32)
    analysis = detector.analyze_embedding(cover_embedding)
    
    print(f"   Risk Level: {analysis['overall_risk']}")
    print(f"   Risk Score: {analysis['risk_score']:.3f}")
    print(f"   Similar Tracks: {analysis['total_similar_tracks']}")
    print(f"   Copyright Matches: {analysis['total_copyright_matches']}")
    
    # Scenario 3: Near-identical copy
    print("\n⚠️  Scenario 3: Near-identical copy")
    copied_song = music_database[10] + np.random.normal(0, 0.05, dimension).astype(np.float32)
    analysis = detector.analyze_embedding(copied_song)
    
    print(f"   Risk Level: {analysis['overall_risk']}")
    print(f"   Risk Score: {analysis['risk_score']:.3f}")
    print(f"   Similar Tracks: {analysis['total_similar_tracks']}")
    print(f"   Copyright Matches: {analysis['total_copyright_matches']}")
    
    print("\n4️⃣ Finding similar tracks...")
    
    # Demonstrate similarity search
    query_embedding = music_database[0]  # Use first song as query
    similar_tracks = searcher.search_similar(query_embedding, k=5)
    
    print(f"\nTop 5 similar tracks to '{metadata_database[0]['title']}':")
    for i, track in enumerate(similar_tracks, 1):
        print(f"   {i}. {track['title']} by {track['artist']} "
              f"(similarity: {track['similarity_score']:.3f})")
    
    print("\n5️⃣ Saving the database...")
    
    # Save the index for later use
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "music_database")
        indexer.save_index(index_path)
        print("✅ Database saved successfully")
        
        # Demonstrate loading
        new_indexer = VectorIndexer(dimension=dimension)
        new_indexer.load_index(index_path)
        print("✅ Database loaded successfully")
        
        # Verify the loaded index works
        new_searcher = SimilaritySearcher(indexer=new_indexer)
        test_results = new_searcher.search_similar(query_embedding, k=3)
        print(f"✅ Loaded database working ({len(test_results)} results)")
    
    print("\n🎉 Demo completed successfully!")
    
    return True

def demo_advanced_usage():
    """Demonstrate advanced features."""
    print("\n🚀 Advanced Features Demo")
    print("=" * 30)
    
    # Different index types
    print("\n🔧 Testing different index types...")
    
    dimension = 64
    test_embeddings = np.random.rand(100, dimension).astype(np.float32)
    test_metadata = [{'track_id': i, 'filename': f'test_{i}.wav'} for i in range(100)]
    
    index_types = ["FlatL2", "IVF", "HNSW"]
    
    for index_type in index_types:
        try:
            print(f"\n   Testing {index_type} index...")
            indexer = VectorIndexer(dimension=dimension, index_type=index_type)
            indexer.add_embeddings(test_embeddings, test_metadata)
            
            searcher = SimilaritySearcher(indexer=indexer)
            query = np.random.rand(dimension).astype(np.float32)
            
            import time
            start_time = time.time()
            results = searcher.search_similar(query, k=10)
            search_time = time.time() - start_time
            
            print(f"   ✅ {index_type}: {len(results)} results in {search_time*1000:.2f}ms")
            
        except Exception as e:
            print(f"   ⚠️  {index_type}: {e}")
    
    print("\n📊 Performance characteristics:")
    print("   • FlatL2: Exact search, best accuracy")
    print("   • IVF: Fast approximate search, good for large datasets")
    print("   • HNSW: Memory-efficient, fast queries")

def main():
    """Run the complete demo."""
    try:
        demo_basic_usage()
        demo_advanced_usage()
        
        print("\n" + "="*60)
        print("🎵 COPYRIGHT DETECTION SYSTEM DEMO COMPLETE")
        print("="*60)
        print("\n✨ Ready for integration with your music projects! ✨")
        print("\nNext steps:")
        print("1. Integrate with music-embeddings module for audio processing")
        print("2. Connect to your backend API")
        print("3. Scale to your full music catalog")
        print("4. Deploy for production use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
