"""
AIrganizer - Intelligent File Organization Tool

A cross-platform tool that uses AI to automatically categorize and organize files.
Works on macOS, Linux, and Windows.
"""

__version__ = "1.0.0"
__author__ = "AIrganizer Team"
__email__ = ""

from .core import AI_rganize
from .cli import main

__all__ = ["AI_rganize", "main"]
