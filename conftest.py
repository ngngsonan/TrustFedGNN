"""Ensure the repository root is importable as ``src`` regardless of how pytest
is invoked (`pytest` vs `python -m pytest`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
