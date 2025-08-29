"""
🎵 Vector Indexer Module

FAISS-based vector indexing for audio embeddings.
Build, save, and load indexes efficiently for fast similarity search.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorIndexer:
    """
    A FAISS-based vector indexer for audio embeddings.
    
    This class provides functionality to build, save, and load FAISS indexes
    for efficient similarity search of audio embeddings.
    """
    
    def __init__(self, dimension: int, index_type: str = "FlatL2", metric: str = "L2"):
        """
        Initialize the VectorIndexer.
        
        Args:
            dimension (int): The dimension of the embeddings
            index_type (str): Type of FAISS index ('FlatL2', 'IVF', 'HNSW')
            metric (str): Distance metric ('L2' or 'IP' for inner product)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.metadata = []
        self.is_trained = False
        
        # Initialize the FAISS index
        self._create_index()
        
    def _create_index(self):
        """Create the FAISS index based on the specified type."""
        if self.index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
            self.is_trained = True
        elif self.index_type == "IVF":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 centroids
        elif self.index_type == "HNSW":
            # HNSW index for fast approximate search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.is_trained = True
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
            
        logger.info(f"Created {self.index_type} index with dimension {self.dimension}")
    
    def train_index(self, training_vectors: np.ndarray):
        """
        Train the index (required for some index types like IVF).
        
        Args:
            training_vectors (np.ndarray): Training vectors for the index
        """
        if not self.is_trained and hasattr(self.index, 'train'):
            logger.info(f"Training index with {len(training_vectors)} vectors...")
            self.index.train(training_vectors.astype(np.float32))
            self.is_trained = True
            logger.info("Index training completed")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add embeddings to the index with associated metadata.
        
        Args:
            embeddings (np.ndarray): Array of embeddings to add
            metadata (List[Dict]): List of metadata dictionaries for each embedding
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
            
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Train index if needed
        if not self.is_trained:
            self.train_index(embeddings)
        
        # Add embeddings to index
        embeddings_f32 = embeddings.astype(np.float32)
        self.index.add(embeddings_f32)
        
        # Store metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")
    
    def add_from_directory(self, embeddings_dir: str, file_extension: str = ".npy"):
        """
        Add embeddings from a directory of saved embedding files.
        
        Args:
            embeddings_dir (str): Directory containing embedding files
            file_extension (str): File extension of embedding files
        """
        embeddings_path = Path(embeddings_dir)
        if not embeddings_path.exists():
            raise ValueError(f"Directory not found: {embeddings_dir}")
        
        embedding_files = list(embeddings_path.glob(f"*{file_extension}"))
        if not embedding_files:
            raise ValueError(f"No embedding files found in {embeddings_dir}")
        
        all_embeddings = []
        all_metadata = []
        
        for file_path in embedding_files:
            try:
                # Load embedding
                embedding = np.load(file_path)
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                
                all_embeddings.append(embedding)
                
                # Create metadata
                metadata = {
                    'file_path': str(file_path),
                    'filename': file_path.stem,
                    'file_id': len(all_metadata)
                }
                all_metadata.extend([metadata] * len(embedding))
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            self.add_embeddings(combined_embeddings, all_metadata)
            logger.info(f"Successfully loaded embeddings from {len(embedding_files)} files")
        else:
            logger.error("No valid embeddings found in directory")
    
    def save_index(self, save_path: str):
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            save_path (str): Path to save the index (without extension)
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No index to save or index is empty")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = str(save_path) + ".faiss"
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = str(save_path) + "_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'metric': self.metric
            }, f)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_index(self, load_path: str):
        """
        Load a FAISS index and metadata from disk.
        
        Args:
            load_path (str): Path to load the index from (without extension)
        """
        load_path = Path(load_path)
        
        # Load FAISS index
        index_path = str(load_path) + ".faiss"
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = str(load_path) + "_metadata.pkl"
        if Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get('metadata', [])
                self.dimension = data.get('dimension', self.dimension)
                self.index_type = data.get('index_type', self.index_type)
                self.metric = data.get('metric', self.metric)
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            self.metadata = []
        
        self.is_trained = True
        logger.info(f"Loaded index with {self.index.ntotal} vectors from {index_path}")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dict: Dictionary containing index statistics
        """
        if self.index is None:
            return {'status': 'No index created'}
        
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'is_trained': self.is_trained,
            'metadata_count': len(self.metadata)
        }
    
    def optimize_index(self):
        """
        Optimize the index for better search performance.
        This is mainly useful for IVF-type indexes.
        """
        if hasattr(self.index, 'nprobe'):
            # Set nprobe for IVF indexes
            self.index.nprobe = min(10, self.index.nlist)
            logger.info(f"Set nprobe to {self.index.nprobe}")
        
        logger.info("Index optimization completed")


def build_index_from_embeddings_module(music_embeddings_path: str, 
                                     audio_files: List[str], 
                                     output_path: str,
                                     model_name: str = "spectrogram",
                                     index_type: str = "FlatL2") -> VectorIndexer:
    """
    Build an index using the music embeddings extraction module.
    
    Args:
        music_embeddings_path (str): Path to the music embeddings project
        audio_files (List[str]): List of audio file paths
        output_path (str): Path to save the index
        model_name (str): Model name for embedding extraction
        index_type (str): Type of FAISS index to create
        
    Returns:
        VectorIndexer: The built and saved index
    """
    import sys
    sys.path.append(music_embeddings_path)
    
    try:
        from src.embeddings import AudioEmbeddingExtractor
        
        # Initialize extractor
        extractor = AudioEmbeddingExtractor(model_name=model_name)
        
        # Extract embeddings for all files
        all_embeddings = []
        all_metadata = []
        
        logger.info(f"Extracting embeddings for {len(audio_files)} files...")
        
        for i, audio_file in enumerate(audio_files):
            try:
                embeddings = extractor.extract_embeddings(audio_file)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                all_embeddings.append(embeddings)
                
                # Create metadata for each embedding
                for j in range(len(embeddings)):
                    metadata = {
                        'file_path': audio_file,
                        'filename': Path(audio_file).stem,
                        'file_id': i,
                        'segment_id': j,
                        'model_name': model_name
                    }
                    all_metadata.append(metadata)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(audio_files)} files")
                    
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")
                continue
        
        if not all_embeddings:
            raise ValueError("No embeddings were successfully extracted")
        
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        # Create and build index
        indexer = VectorIndexer(
            dimension=combined_embeddings.shape[1],
            index_type=index_type
        )
        
        indexer.add_embeddings(combined_embeddings, all_metadata)
        indexer.save_index(output_path)
        
        logger.info(f"Successfully built index with {len(combined_embeddings)} embeddings")
        return indexer
        
    except ImportError as e:
        logger.error(f"Failed to import music embeddings module: {e}")
        logger.error("Make sure the music embeddings project is in the correct path")
        raise
