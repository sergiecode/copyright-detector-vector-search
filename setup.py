"""
Setup script for Copyright Detector Vector Search

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'faiss-cpu>=1.7.4',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
    ]

setup(
    name="copyright-detector-vector-search",
    version="1.0.0",
    author="Sergie Code",
    author_email="your.email@example.com",
    description="FAISS-based vector indexing and similarity search for audio embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/copyright-detector-vector-search",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["faiss-gpu>=1.7.4"],
        "dev": [
            "pytest>=6.2.4",
            "pytest-cov>=2.12.0",
            "black>=21.6.0",
            "flake8>=3.9.2",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vector-search-test=test_installation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="audio, music, copyright, similarity, search, faiss, embeddings, vector",
    project_urls={
        "Bug Reports": "https://github.com/your-username/copyright-detector-vector-search/issues",
        "Source": "https://github.com/your-username/copyright-detector-vector-search",
        "Documentation": "https://github.com/your-username/copyright-detector-vector-search/blob/main/README.md",
    },
)
