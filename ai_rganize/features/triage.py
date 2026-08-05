"""Lightweight, non-AI triage helpers for quickly understanding a pile of files.

These are cheap heuristics (no network calls, no LLM usage) meant to give a
user or the CLI a quick sense of what's in a directory before spending money
on AI categorization.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .archives import is_rar_archive, is_supported_archive

_CATEGORY_BY_EXTENSION = {
    "documents": {".pdf", ".doc", ".docx", ".txt", ".rtf", ".pages", ".odt"},
    "images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".heic", ".webp", ".svg"},
    "videos": {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"},
    "audio": {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"},
    "archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".xz"},
    "code": {
        ".py", ".js", ".ts", ".html", ".css", ".java", ".cpp", ".c",
        ".php", ".rb", ".go", ".rs",
    },
    "spreadsheets": {".xls", ".xlsx", ".csv", ".numbers"},
    "presentations": {".ppt", ".pptx", ".key"},
}

_EXTENSION_TO_CATEGORY = {ext: cat for cat, exts in _CATEGORY_BY_EXTENSION.items() for ext in exts}


def explain_file(path: Path) -> dict[str, Any]:
    """Produce a short, human-readable explanation of what a file is and why
    it might belong in a particular category -- without calling any AI."""
    path = Path(path)
    try:
        stat = path.stat()
        size_bytes = stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
    except OSError:
        size_bytes = 0
        modified = None

    extension = path.suffix.lower()
    category = _EXTENSION_TO_CATEGORY.get(extension, "other")

    is_archive = is_supported_archive(path) or is_rar_archive(path)

    reasons = [f"Extension '{extension or '(none)'}' maps to category '{category}'."]
    if size_bytes == 0:
        reasons.append("File is empty (0 bytes).")
    elif size_bytes > 1_000_000_000:
        reasons.append("File is very large (> 1 GB).")

    return {
        "path": str(path),
        "name": path.name,
        "extension": extension,
        "category": category,
        "size_bytes": size_bytes,
        "modified": modified,
        "is_archive": is_archive,
        "reasons": reasons,
    }


def suggest_archives(files: list[Path]) -> list[dict[str, Any]]:
    """Filter *files* down to the ones that look like archives worth inspecting."""
    suggestions = []
    for f in files:
        f = Path(f)
        if is_rar_archive(f):
            suggestions.append({
                "path": f,
                "supported": False,
                "reason": "RAR archives are not auto-extractable",
            })
        elif is_supported_archive(f):
            suggestions.append({
                "path": f,
                "supported": True,
                "reason": "Zip/Tar archive available for inspection",
            })
    return suggestions


def cluster_by_extension(files: list[Path]) -> dict[str, list[Path]]:
    """Group files by their (lowercased) extension, e.g. '.pdf' -> [list of files]."""
    clusters: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        f = Path(f)
        ext = f.suffix.lower() or "(no extension)"
        clusters[ext].append(f)
    return dict(clusters)


def cluster_by_category(files: list[Path]) -> dict[str, list[Path]]:
    """Group files by broad category (documents/images/videos/etc.)."""
    clusters: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        f = Path(f)
        category = _EXTENSION_TO_CATEGORY.get(f.suffix.lower(), "other")
        clusters[category].append(f)
    return dict(clusters)
