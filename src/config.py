"""
🎵 Configuration for Vector Search Module

Default configuration settings for the copyright detector vector search system.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
SRC_DIR = PROJECT_ROOT / "src"

# Default embeddings module path (relative to this project)
DEFAULT_EMBEDDINGS_PATH = "../copyright-detector-music-embeddings"

# FAISS Configuration
FAISS_CONFIG = {
    'default_index_type': 'FlatL2',  # FlatL2, IVF, HNSW
    'default_metric': 'L2',  # L2 or IP (Inner Product)
    'ivf_centroids': 100,  # Number of centroids for IVF index
    'hnsw_m': 32,  # Number of connections for HNSW
    'default_nprobe': 10,  # Search parameter for IVF
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    'default_dimension': 128,
    'supported_models': ['spectrogram', 'openl3', 'audioclip'],
    'default_model': 'spectrogram',
    'batch_size': 32,
}

# Search Configuration
SEARCH_CONFIG = {
    'default_k': 10,  # Number of results to return
    'max_k': 100,  # Maximum number of results
    'default_similarity_threshold': 0.8,
    'copyright_thresholds': {
        'very_high_risk': 0.95,
        'high_risk': 0.85,
        'medium_risk': 0.70,
        'low_risk': 0.50
    }
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'small_dataset_threshold': 10000,  # Use FlatL2 for datasets smaller than this
    'large_dataset_threshold': 100000,  # Use IVF for datasets larger than this
    'batch_processing_size': 100,  # Process embeddings in batches of this size
    'max_memory_gb': 8,  # Maximum memory usage in GB
}

# File Configuration
FILE_CONFIG = {
    'supported_audio_formats': ['.wav', '.mp3', '.flac', '.m4a', '.aac'],
    'embedding_file_extension': '.npy',
    'index_file_extension': '.faiss',
    'metadata_file_extension': '_metadata.pkl',
    'max_file_size_mb': 100,
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': False,
    'log_file': 'vector_search.log'
}

# Copyright Detection Configuration
COPYRIGHT_CONFIG = {
    'default_threshold': 0.8,
    'high_confidence_threshold': 0.95,
    'medium_confidence_threshold': 0.85,
    'low_confidence_threshold': 0.70,
    'max_matches_to_check': 50,
    'duplicate_threshold': 0.95,
}

# Development Configuration
DEV_CONFIG = {
    'debug_mode': False,
    'verbose_logging': False,
    'save_intermediate_results': False,
    'profile_performance': False,
}


def get_config():
    """
    Get the complete configuration dictionary.
    
    Returns:
        dict: Complete configuration settings
    """
    return {
        'faiss': FAISS_CONFIG,
        'embedding': EMBEDDING_CONFIG,
        'search': SEARCH_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'file': FILE_CONFIG,
        'logging': LOGGING_CONFIG,
        'copyright': COPYRIGHT_CONFIG,
        'development': DEV_CONFIG,
        'paths': {
            'project_root': str(PROJECT_ROOT),
            'data_dir': str(DATA_DIR),
            'examples_dir': str(EXAMPLES_DIR),
            'src_dir': str(SRC_DIR),
            'embeddings_path': DEFAULT_EMBEDDINGS_PATH,
        }
    }


def get_index_config_for_size(dataset_size):
    """
    Get recommended index configuration based on dataset size.
    
    Args:
        dataset_size (int): Number of vectors in the dataset
        
    Returns:
        dict: Recommended index configuration
    """
    if dataset_size < PERFORMANCE_CONFIG['small_dataset_threshold']:
        return {
            'index_type': 'FlatL2',
            'metric': 'L2',
            'description': 'Exact search for small dataset'
        }
    elif dataset_size < PERFORMANCE_CONFIG['large_dataset_threshold']:
        return {
            'index_type': 'IVF',
            'metric': 'L2',
            'nlist': min(int(dataset_size / 100), 1000),
            'nprobe': 10,
            'description': 'Approximate search for medium dataset'
        }
    else:
        return {
            'index_type': 'HNSW',
            'metric': 'L2',
            'm': 32,
            'description': 'Graph-based search for large dataset'
        }


def get_similarity_thresholds():
    """
    Get copyright detection similarity thresholds.
    
    Returns:
        dict: Similarity thresholds for different risk levels
    """
    return COPYRIGHT_CONFIG.copy()


def create_data_directories():
    """Create necessary data directories if they don't exist."""
    directories = [
        DATA_DIR,
        DATA_DIR / "indexes",
        DATA_DIR / "embeddings",
        DATA_DIR / "temp",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Environment-based configuration overrides
def load_environment_config():
    """Load configuration from environment variables if available."""
    env_config = {}
    
    # FAISS configuration from environment
    if os.getenv('FAISS_INDEX_TYPE'):
        env_config['faiss_index_type'] = os.getenv('FAISS_INDEX_TYPE')
    
    # Search configuration from environment
    if os.getenv('SIMILARITY_THRESHOLD'):
        env_config['similarity_threshold'] = float(os.getenv('SIMILARITY_THRESHOLD'))
    
    # Performance configuration from environment
    if os.getenv('MAX_MEMORY_GB'):
        env_config['max_memory_gb'] = int(os.getenv('MAX_MEMORY_GB'))
    
    # Paths from environment
    if os.getenv('EMBEDDINGS_MODULE_PATH'):
        env_config['embeddings_path'] = os.getenv('EMBEDDINGS_MODULE_PATH')
    
    return env_config


# Initialize directories on import
create_data_directories()
