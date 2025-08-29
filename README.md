# 🎵 Copyright Detector Vector Search

**A FAISS-based vector indexing and similarity search system for audio embeddings**

Created by **Sergie Code** - Software Engineer & YouTube Programming Educator  
**AI Tools for Musicians Series**

## 🎯 Purpose

This project provides a fast and scalable vector indexing and similarity search system specifically designed for music copyright detection and audio similarity analysis. It uses **FAISS** (Facebook AI Similarity Search) to build efficient indexes of audio embeddings and perform lightning-fast similarity searches.

### Key Features

- ⚡ **Fast Similarity Search**: Sub-second search times even with millions of audio tracks
- 🎵 **Music-Focused**: Optimized for audio embedding vectors and music metadata
- 📈 **Scalable**: Handles large music collections with efficient indexing
- 🔍 **Copyright Detection**: Built-in similarity thresholds for copyright matching
- 🔄 **Batch Processing**: Process multiple audio files simultaneously
- 💾 **Persistent Storage**: Save and load indexes for production use
- 🎯 **Easy Integration**: Seamlessly works with the music embeddings extraction module

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Vector Search Module                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   VectorIndexer │───▶│ SimilaritySearcher │               │
│  │                 │    │                 │                │
│  │ • Build Index   │    │ • Find Similar  │                │
│  │ • Save/Load     │    │ • Copyright     │                │
│  │ • Optimize      │    │ • Batch Search  │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│      FAISS Index            Search Results                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/copyright-detector-vector-search.git
cd copyright-detector-vector-search
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# For GPU support (optional, for large datasets)
pip install faiss-gpu
```

### 3. Verify Installation

```bash
python -c "import faiss; print('FAISS version:', faiss.__version__)"
```

## 🚀 Quick Start

### Basic Usage Example

```python
import numpy as np
from src.indexer import VectorIndexer
from src.search import SimilaritySearcher

# 1. Create embeddings (normally from audio files)
embeddings = np.random.rand(100, 128).astype(np.float32)
metadata = [{'filename': f'song_{i}.wav', 'artist': f'Artist_{i}'} 
           for i in range(100)]

# 2. Build the index
indexer = VectorIndexer(dimension=128, index_type="FlatL2")
indexer.add_embeddings(embeddings, metadata)
indexer.save_index("music_index")

# 3. Search for similar tracks
searcher = SimilaritySearcher(index_path="music_index")
query = np.random.rand(128).astype(np.float32)
results = searcher.search_similar(query, k=5)

print("Top 5 similar tracks:")
for result in results:
    print(f"  {result['filename']} - Similarity: {result['similarity_score']:.3f}")
```

## 🎵 Integration with Music Embeddings

This project is designed to work seamlessly with the [copyright-detector-music-embeddings](../copyright-detector-music-embeddings) module.

### Step-by-Step Integration

```python
from src.indexer import build_index_from_embeddings_module

# Build index directly from audio files
audio_files = [
    "path/to/song1.wav",
    "path/to/song2.mp3",
    "path/to/song3.flac"
]

indexer = build_index_from_embeddings_module(
    music_embeddings_path="../copyright-detector-music-embeddings",
    audio_files=audio_files,
    output_path="my_music_index",
    model_name="spectrogram"  # or "openl3", "audioclip"
)

print(f"Built index with {indexer.get_stats()['total_vectors']} tracks")
```

### Find Similar Tracks to Audio File

```python
from src.search import SimilaritySearcher

# Load your music index
searcher = SimilaritySearcher(index_path="my_music_index")

# Find tracks similar to a new audio file
similar_tracks = searcher.find_similar_tracks(
    audio_file="query_song.wav",
    k=10,
    music_embeddings_path="../copyright-detector-music-embeddings"
)

for track in similar_tracks:
    print(f"{track['filename']} - {track['similarity_score']:.3f}")
```

## ⚖️ Copyright Detection

### Detect Potential Copyright Matches

```python
from src.search import SimilaritySearcher, CopyrightDetector

# Initialize copyright detector
searcher = SimilaritySearcher(index_path="my_music_index")
detector = CopyrightDetector(searcher)

# Analyze a track for copyright issues
analysis = detector.analyze_track(
    audio_file="suspicious_song.wav",
    music_embeddings_path="../copyright-detector-music-embeddings"
)

print(f"Overall Risk: {analysis['overall_risk']}")
print(f"Risk Score: {analysis['risk_score']:.3f}")
print(f"Potential Matches: {analysis['total_copyright_matches']}")

for match in analysis['copyright_matches']:
    print(f"  Match: {match['filename']} - Risk: {match['copyright_risk']}")
```

## 📊 Advanced Features

### Batch Processing

```python
# Process multiple audio files at once
audio_files = ["song1.wav", "song2.wav", "song3.wav"]
batch_results = []

for audio_file in audio_files:
    results = searcher.find_similar_tracks(audio_file, k=5)
    batch_results.append(results)

print(f"Processed {len(batch_results)} files")
```

### Index Optimization

```python
# Optimize index for better performance
indexer = VectorIndexer(dimension=128, index_type="IVF")  # Use IVF for large datasets
indexer.load_index("my_music_index")
indexer.optimize_index()
indexer.save_index("my_music_index_optimized")
```

### Duplicate Detection

```python
# Find potential duplicate tracks
duplicates = searcher.find_duplicates(similarity_threshold=0.95)

print(f"Found {len(duplicates)} potential duplicate pairs")
for dup in duplicates:
    print(f"Duplicate: {dup['file1_metadata']['filename']} <-> {dup['file2_metadata']['filename']}")
```

## 🔧 Configuration Options

### Index Types

- **FlatL2**: Exact search, best for small to medium datasets (<100K tracks)
- **IVF**: Approximate search, faster for large datasets (>100K tracks)
- **HNSW**: Graph-based search, excellent for real-time applications

### Similarity Thresholds

- **0.95+**: Very high similarity (likely duplicates or copyright issues)
- **0.85-0.94**: High similarity (potential copyright concerns)
- **0.70-0.84**: Medium similarity (similar style or genre)
- **<0.70**: Low similarity (different tracks)

## 📁 Project Structure

```
copyright-detector-vector-search/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── indexer.py               # FAISS index building and management
│   └── search.py                # Similarity search and copyright detection
├── examples/
│   ├── build_index_example.py   # Index building examples
│   └── search_example.py        # Search and detection examples
├── data/                        # Storage for indexes and embeddings
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎓 Educational Examples

### Example 1: Build Index from Music Collection

```bash
cd examples
python build_index_example.py
```

This example shows how to:
- Build an index from audio files
- Save the index for later use
- Load and optimize existing indexes

### Example 2: Similarity Search and Copyright Detection

```bash
cd examples
python search_example.py
```

This example demonstrates:
- Basic similarity search
- Copyright detection with thresholds
- Batch processing
- Duplicate detection

## 🌐 Backend API Integration

This module serves as the foundation for the **copyright-detector-music-backend** project, which provides a REST API for large-scale music similarity analysis.

### Preparing for Backend Integration

```python
# Your vector search module is ready for backend integration!
# The backend will use these classes:

from src.indexer import VectorIndexer
from src.search import SimilaritySearcher, CopyrightDetector

# These will be called from .NET Core Web API
# via Python script wrappers for production use
```

## 📈 Performance Tips

### For Large Datasets (>100K tracks)

1. **Use IVF Index**: Better performance for large collections
2. **GPU Acceleration**: Install `faiss-gpu` for faster processing  
3. **Batch Processing**: Process multiple files simultaneously
4. **Index Optimization**: Tune nprobe parameter for IVF indexes

### Memory Management

```python
# For very large datasets, use index sharding
def create_sharded_index(embeddings, shard_size=100000):
    shards = []
    for i in range(0, len(embeddings), shard_size):
        shard_embeddings = embeddings[i:i+shard_size]
        indexer = VectorIndexer(dimension=embeddings.shape[1])
        indexer.add_embeddings(shard_embeddings, metadata[i:i+shard_size])
        shards.append(indexer)
    return shards
```

## 🔬 Technical Details

### FAISS Index Types Explained

- **IndexFlatL2**: Brute force L2 distance search (exact results)
- **IndexIVFFlat**: Inverted file index (approximate but fast)
- **IndexHNSWFlat**: Hierarchical NSW graph (excellent recall/speed trade-off)

### Distance Metrics

- **L2 (Euclidean)**: Good for normalized embeddings
- **Inner Product**: For embeddings with different magnitudes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## 📄 License

This project is part of the "AI Tools for Musicians" educational series by Sergie Code.

## 👨‍💻 About the Author

**Sergie Code** is a Software Engineer and YouTube Programming Educator specializing in AI tools for creative industries. This project is part of his educational series teaching musicians how to leverage AI technology.

### Connect with Sergie Code
- 📸 Instagram: https://www.instagram.com/sergiecode

- 🧑🏼‍💻 LinkedIn: https://www.linkedin.com/in/sergiecode/

- 📽️Youtube: https://www.youtube.com/@SergieCode

- 😺 Github: https://github.com/sergiecode

- 👤 Facebook: https://www.facebook.com/sergiecodeok

- 🎞️ Tiktok: https://www.tiktok.com/@sergiecode

- 🕊️Twitter: https://twitter.com/sergiecode

- 🧵Threads: https://www.threads.net/@sergiecode

## 🎯 Next Steps

1. **Try the Examples**: Run the provided examples to understand the functionality
2. **Build Your Index**: Use your own music collection to create a vector index
3. **Experiment**: Try different index types and similarity thresholds
4. **Integrate**: Connect with the music embeddings module for end-to-end processing
5. **Scale Up**: Move to the backend API module for production deployment

## 🆘 Support

If you encounter issues or have questions:

1. Check the [examples](./examples/) for common use cases
2. Review the error messages and ensure all dependencies are installed
3. Verify that the music embeddings module is properly installed
4. Open an issue on GitHub with detailed error information

---

**Happy Music Analysis! 🎵**

*Built with ❤️ by Sergie Code for the music and AI community*
