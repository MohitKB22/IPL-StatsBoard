"""
conftest.py
===========
Pytest configuration — adds project root to sys.path and
suppresses Streamlit cache warnings during test runs.
"""
import sys
import os
import warnings

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))

# Silence noisy non-test warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*copy.*on.*write.*")
warnings.filterwarnings("ignore", message=".*ChainedAssignment.*")
warnings.filterwarnings("ignore", message=".*No runtime found.*")
