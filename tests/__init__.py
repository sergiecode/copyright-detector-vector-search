"""
🎵 Test Package for Copyright Detector Vector Search

Comprehensive test suite to ensure all functionality works correctly.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path for testing
test_dir = Path(__file__).parent
project_root = test_dir.parent
src_dir = project_root / "src"

sys.path.insert(0, str(src_dir))

__version__ = "1.0.0"
__author__ = "Sergie Code"
