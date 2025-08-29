"""
🎵 Example: Vector Similarity Search for Music

This example demonstrates how to perform similarity search on audio embeddings
for finding similar tracks and detecting potential copyright matches.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import os
import sys
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from search import SimilaritySearcher, CopyrightDetector
from indexer import VectorIndexer


def example_basic_similarity_search():
    """
    Example: Basic similarity search using a pre-built index.
    """
    print("🎵 Basic Similarity Search Example")
    print("=" * 40)
    
    # First, let's create a demo index for this example
    print("📦 Creating demo index...")
    
    # Create dummy embeddings and metadata
    num_tracks = 50
    embedding_dim = 128
    
    embeddings = np.random.rand(num_tracks, embedding_dim).astype(np.float32)
    metadata = []
    
    # Create realistic metadata
    artists = ["The Beatles", "Queen", "Led Zeppelin", "Pink Floyd", "Bob Dylan"]
    albums = ["Album A", "Album B", "Greatest Hits", "Live Concert", "Studio Sessions"]
    
    for i in range(num_tracks):
        metadata.append({
            'filename': f'track_{i+1:03d}',
            'file_path': f'/music/collection/track_{i+1:03d}.wav',
            'artist': artists[i % len(artists)],
            'album': albums[i % len(albums)],
            'track_number': (i % 12) + 1,
            'duration': np.random.randint(180, 420),  # 3-7 minutes
            'file_id': i
        })
    
    # Build index
    indexer = VectorIndexer(dimension=embedding_dim, index_type="FlatL2")
    indexer.add_embeddings(embeddings, metadata)
    
    # Create searcher
    searcher = SimilaritySearcher(indexer=indexer)
    
    # Perform similarity search with a random query
    query_embedding = np.random.rand(embedding_dim).astype(np.float32)
    
    print("🔍 Searching for similar tracks...")
    results = searcher.search_similar(query_embedding, k=5)
    
    print("Top 5 similar tracks:")
    for result in results:
        print(f"  Rank {result['rank']}: {result['artist']} - {result['filename']}")
        print(f"    Similarity: {result['similarity_score']:.3f}")
        print(f"    Album: {result['album']}")
        print()


def example_find_similar_tracks():
    """
    Example: Find similar tracks to an input audio file.
    """
    print("🎵 Find Similar Tracks Example")
    print("=" * 35)
    
    # Load an existing index (demo index from previous example)
    index_path = "../data/demo_index"
    
    try:
        searcher = SimilaritySearcher(index_path=index_path)
        
        # This would normally be an actual audio file
        # For demo, we'll show how it would work
        audio_file = "path/to/query_song.wav"
        
        print(f"🎧 Query: {audio_file}")
        print("📝 Note: This is a demo - replace with actual audio file path")
        print()
        
        # In a real scenario, this would extract embeddings and search
        print("🔍 Would perform similarity search...")
        print("Expected output:")
        print("  Top similar tracks with similarity scores")
        print("  Metadata: artist, album, duration, etc.")
        
    except FileNotFoundError:
        print("❌ Demo index not found. Run build_index_example.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_copyright_detection():
    """
    Example: Copyright detection using similarity thresholds.
    """
    print("\n🎵 Copyright Detection Example")
    print("=" * 35)
    
    # Create a demo setup
    print("⚖️  Setting up copyright detection demo...")
    
    # Create dummy embeddings with some "copyrighted" content
    num_tracks = 30
    embedding_dim = 128
    
    embeddings = []
    metadata = []
    
    # Create some original tracks
    for i in range(20):
        embedding = np.random.rand(embedding_dim).astype(np.float32)
        embeddings.append(embedding)
        metadata.append({
            'filename': f'original_track_{i+1}',
            'artist': f'Artist_{i%5 + 1}',
            'is_copyrighted': True,
            'copyright_owner': f'Record Label {i%3 + 1}',
            'file_id': i
        })
    
    # Create some similar tracks (potential copyright infringement)
    for i in range(10):
        base_embedding = embeddings[i]  # Use an existing track as base
        # Add small variations to simulate similar but not identical tracks
        variation = np.random.normal(0, 0.1, embedding_dim).astype(np.float32)
        similar_embedding = base_embedding + variation
        
        embeddings.append(similar_embedding)
        metadata.append({
            'filename': f'suspicious_track_{i+1}',
            'artist': f'Unknown Artist {i+1}',
            'is_copyrighted': False,
            'file_id': 20 + i
        })
    
    # Build index
    all_embeddings = np.array(embeddings)
    indexer = VectorIndexer(dimension=embedding_dim, index_type="FlatL2")
    indexer.add_embeddings(all_embeddings, metadata)
    
    # Create copyright detector
    searcher = SimilaritySearcher(indexer=indexer)
    detector = CopyrightDetector(searcher)
    
    # Test copyright detection with a suspicious track
    suspicious_query = embeddings[25]  # One of the similar tracks
    
    print("🔍 Analyzing suspicious track for copyright matches...")
    matches = searcher.detect_copyright_matches(suspicious_query, similarity_threshold=0.7)
    
    print(f"Found {len(matches)} potential copyright matches:")
    for match in matches[:3]:  # Show top 3 matches
        print(f"  Match: {match['filename']}")
        print(f"    Similarity: {match['similarity_score']:.3f}")
        print(f"    Risk Level: {match['copyright_risk']}")
        print(f"    Confidence: {match['match_confidence']}")
        if 'copyright_owner' in match:
            print(f"    Copyright Owner: {match['copyright_owner']}")
        print()


def example_batch_search():
    """
    Example: Batch similarity search for multiple queries.
    """
    print("\n🎵 Batch Search Example")
    print("=" * 25)
    
    # Create demo index
    num_tracks = 100
    embedding_dim = 128
    
    embeddings = np.random.rand(num_tracks, embedding_dim).astype(np.float32)
    metadata = [{'filename': f'track_{i}', 'file_id': i} for i in range(num_tracks)]
    
    indexer = VectorIndexer(dimension=embedding_dim)
    indexer.add_embeddings(embeddings, metadata)
    
    searcher = SimilaritySearcher(indexer=indexer)
    
    # Create multiple query embeddings
    num_queries = 5
    query_embeddings = np.random.rand(num_queries, embedding_dim).astype(np.float32)
    
    print(f"🔍 Performing batch search for {num_queries} queries...")
    batch_results = searcher.batch_search(query_embeddings, k=3)
    
    print("Batch search results:")
    for i, results in enumerate(batch_results):
        print(f"  Query {i+1}:")
        for result in results:
            print(f"    {result['filename']} (similarity: {result['similarity_score']:.3f})")
        print()


def example_find_duplicates():
    """
    Example: Find potential duplicate tracks in the index.
    """
    print("\n🎵 Duplicate Detection Example")
    print("=" * 32)
    
    # Create demo data with some intentional duplicates
    embedding_dim = 128
    base_embeddings = np.random.rand(20, embedding_dim).astype(np.float32)
    
    embeddings = []
    metadata = []
    
    # Add original tracks
    for i, embedding in enumerate(base_embeddings):
        embeddings.append(embedding)
        metadata.append({
            'filename': f'original_song_{i+1}',
            'artist': f'Artist_{i+1}',
            'file_id': i
        })
    
    # Add some duplicates with slight variations
    for i in range(5):
        base_idx = i * 2  # Use every other track
        duplicate = base_embeddings[base_idx] + np.random.normal(0, 0.01, embedding_dim).astype(np.float32)
        embeddings.append(duplicate)
        metadata.append({
            'filename': f'duplicate_song_{i+1}',
            'artist': f'Different Artist {i+1}',
            'file_id': 20 + i
        })
    
    # Build index
    all_embeddings = np.array(embeddings)
    indexer = VectorIndexer(dimension=embedding_dim)
    indexer.add_embeddings(all_embeddings, metadata)
    
    searcher = SimilaritySearcher(indexer=indexer)
    
    print("🔍 Searching for potential duplicates...")
    duplicates = searcher.find_duplicates(similarity_threshold=0.9)
    
    print(f"Found {len(duplicates)} potential duplicate pairs:")
    for dup in duplicates[:3]:  # Show first 3 pairs
        file1 = dup.get('file1_metadata', {})
        file2 = dup.get('file2_metadata', {})
        print(f"  Pair: {file1.get('filename', 'Unknown')} <-> {file2.get('filename', 'Unknown')}")
        print(f"    Similarity: {dup['similarity_score']:.3f}")
        print()


def example_search_statistics():
    """
    Example: Get search statistics and index information.
    """
    print("\n🎵 Search Statistics Example")
    print("=" * 30)
    
    # Create a demo index
    embedding_dim = 128
    embeddings = np.random.rand(50, embedding_dim).astype(np.float32)
    metadata = [{'filename': f'track_{i}', 'file_id': i} for i in range(50)]
    
    indexer = VectorIndexer(dimension=embedding_dim, index_type="FlatL2")
    indexer.add_embeddings(embeddings, metadata)
    
    searcher = SimilaritySearcher(indexer=indexer)
    
    # Get statistics
    stats = searcher.get_statistics()
    
    print("📊 Index Statistics:")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Dimension: {stats['dimension']}")
    print(f"  Index type: {stats['index_type']}")
    print(f"  Search ready: {stats['search_ready']}")
    print(f"  Supports batch search: {stats['supports_batch_search']}")
    print(f"  Supports copyright detection: {stats['supports_copyright_detection']}")


if __name__ == "__main__":
    print("🎵 Vector Similarity Search Examples")
    print("Created by Sergie Code - AI Tools for Musicians")
    print("=" * 60)
    
    # Run all examples
    example_basic_similarity_search()
    example_find_similar_tracks()
    example_copyright_detection()
    example_batch_search()
    example_find_duplicates()
    example_search_statistics()
    
    print("\n🎉 All search examples completed!")
    print("\n💡 Next steps:")
    print("   1. Replace demo data with your actual music index")
    print("   2. Experiment with different similarity thresholds")
    print("   3. Integrate with audio files using the embeddings module")
    print("   4. Build the backend API for production deployment")
