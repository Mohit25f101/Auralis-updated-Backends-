# ==============================
# 📄 tests/__init__.py
# ==============================
"""
Auralis Ultimate Test Suite
Comprehensive testing for all components
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = [
    "test_analysis",
    "test_semantic",
    "test_entities",
    "test_transcription",
    "test_emotions"
]