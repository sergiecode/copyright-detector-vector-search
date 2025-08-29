"""
🎵 Example: Building a Vector Index from Audio Embeddings

This example demonstrates how to build a FAISS index from audio files
using the music embeddings extraction module.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from indexer import VectorIndexer, build_index_from_embeddings_module


def example_build_index_from_audio_files():
    """
    Example: Build index directly from audio files using the embeddings module.
    """
    print("🎵 Building Vector Index from Audio Files")
    print("=" * 50)
    
    # Configuration
    music_embeddings_path = "../copyright-detector-music-embeddings"
    audio_files = [
        # Add paths to your audio files here
        # "path/to/song1.wav",
        # "path/to/song2.mp3",
        # "path/to/song3.flac",
    ]
    
    # For demo purposes, let's create some dummy paths
    # In real usage, replace these with actual audio file paths
    if not audio_files:
        print("⚠️  No audio files specified. Please add audio file paths to the audio_files list.")
        print("Example audio files:")
        for i in range(5):
            print(f"   - example_song_{i+1}.wav")
        return
    
    output_path = "../data/music_index"
    
    try:
        # Build the index
        indexer = build_index_from_embeddings_module(
            music_embeddings_path=music_embeddings_path,
            audio_files=audio_files,
            output_path=output_path,
            model_name="spectrogram",  # or "openl3", "audioclip"
            index_type="FlatL2"  # or "IVF", "HNSW"
        )
        
        # Display statistics
        stats = indexer.get_stats()
        print(f"✅ Index built successfully!")
        print(f"   Total vectors: {stats['total_vectors']}")
        print(f"   Dimension: {stats['dimension']}")
        print(f"   Index type: {stats['index_type']}")
        print(f"   Saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        print("Make sure the music embeddings project is in the correct path.")


def example_build_index_from_embeddings():
    """
    Example: Build index from pre-computed embeddings.
    """
    print("\n🎵 Building Vector Index from Pre-computed Embeddings")
    print("=" * 55)
    
    # Create some dummy embeddings for demonstration
    # In real usage, you would load your actual embeddings
    num_tracks = 100
    embedding_dim = 128
    
    embeddings = np.random.rand(num_tracks, embedding_dim).astype(np.float32)
    
    # Create metadata for each embedding
    metadata = []
    for i in range(num_tracks):
        metadata.append({
            'filename': f'song_{i+1}',
            'file_path': f'/music/collection/song_{i+1}.wav',
            'artist': f'Artist_{i%10 + 1}',
            'album': f'Album_{i%20 + 1}',
            'duration': np.random.randint(120, 300),  # seconds
            'file_id': i
        })
    
    # Create and build index
    indexer = VectorIndexer(dimension=embedding_dim, index_type="FlatL2")
    indexer.add_embeddings(embeddings, metadata)
    
    # Save the index
    output_path = "../data/demo_index"
    indexer.save_index(output_path)
    
    # Display statistics
    stats = indexer.get_stats()
    print(f"✅ Demo index built successfully!")
    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Dimension: {stats['dimension']}")
    print(f"   Index type: {stats['index_type']}")
    print(f"   Saved to: {output_path}")


def example_load_and_optimize_index():
    """
    Example: Load an existing index and optimize it.
    """
    print("\n🎵 Loading and Optimizing Index")
    print("=" * 35)
    
    index_path = "../data/demo_index"
    
    try:
        # Load the index
        indexer = VectorIndexer(dimension=128)  # Dimension should match the saved index
        indexer.load_index(index_path)
        
        print(f"✅ Index loaded successfully!")
        
        # Get statistics before optimization
        stats = indexer.get_stats()
        print(f"   Total vectors: {stats['total_vectors']}")
        print(f"   Index type: {stats['index_type']}")
        
        # Optimize the index
        indexer.optimize_index()
        print(f"🚀 Index optimized for better search performance!")
        
    except FileNotFoundError:
        print(f"❌ Index not found at {index_path}")
        print("Run the previous examples first to create a demo index.")
    except Exception as e:
        print(f"❌ Error loading index: {e}")


def example_build_index_from_directory():
    """
    Example: Build index from a directory of embedding files.
    """
    print("\n🎵 Building Index from Embedding Files Directory")
    print("=" * 50)
    
    # Create a demo directory with some dummy embedding files
    embeddings_dir = "../data/embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Create some dummy embedding files
    embedding_dim = 128
    for i in range(10):
        embedding = np.random.rand(embedding_dim).astype(np.float32)
        np.save(os.path.join(embeddings_dir, f"track_{i+1}.npy"), embedding)
    
    print(f"📁 Created {10} demo embedding files in {embeddings_dir}")
    
    # Build index from directory
    indexer = VectorIndexer(dimension=embedding_dim, index_type="FlatL2")
    
    try:
        indexer.add_from_directory(embeddings_dir, file_extension=".npy")
        
        # Save the index
        output_path = "../data/directory_index"
        indexer.save_index(output_path)
        
        # Display statistics
        stats = indexer.get_stats()
        print(f"✅ Index built from directory successfully!")
        print(f"   Total vectors: {stats['total_vectors']}")
        print(f"   Dimension: {stats['dimension']}")
        print(f"   Saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error building index from directory: {e}")


if __name__ == "__main__":
    print("🎵 Vector Indexing Examples")
    print("Created by Sergie Code - AI Tools for Musicians")
    print("=" * 60)
    
    # Run all examples
    example_build_index_from_audio_files()
    example_build_index_from_embeddings()
    example_load_and_optimize_index()
    example_build_index_from_directory()
    
    print("\n🎉 All examples completed!")
    print("\n💡 Next steps:")
    print("   1. Replace demo data with your actual audio files and embeddings")
    print("   2. Experiment with different index types (IVF, HNSW)")
    print("   3. Try the search examples to query your index")
    print("   4. Integrate with the backend API for production use")
