"""
File utility functions for common operations.
"""

import re
from pathlib import Path


def extract_person_name(filename: str) -> str:
    """Extract person name from filename for intelligent grouping."""
    # Remove file extension
    name = Path(filename).stem
    
    # Common patterns for person names
    patterns = [
        r'([A-Z][a-z]+_[A-Z][a-z]+)',  # FirstName_LastName
        r'([A-Z][a-z]+\.[A-Z][a-z]+)',  # FirstName.LastName
        r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # FirstName LastName
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1).replace('_', ' ').replace('.', ' ')
    
    return None


def is_system_file(file_path: Path) -> bool:
    return file_path.name in SYSTEM_FILES or file_path.name.startswith('.')


def get_file_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)


def get_file_size_kb(file_path: Path) -> float:
    return file_path.stat().st_size / 1024
