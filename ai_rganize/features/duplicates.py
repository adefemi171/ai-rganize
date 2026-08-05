"""Exact duplicate file detection using size prefiltering + sha256 hashing."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _hash_file(path: Path, chunk_size: int = 65536) -> str | None:
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


def find_exact_duplicates(files: Iterable[Path]) -> dict[str, list[Path]]:
    """Find files with identical content, keyed by sha256 hash.

    Uses a cheap file-size prefilter so we only hash files that share a size
    with at least one other file, avoiding needless I/O on unique files.
    """
    by_size: dict[int, list[Path]] = defaultdict(list)
    for f in files:
        f = Path(f)
        try:
            if not f.is_file():
                continue
            size = f.stat().st_size
        except OSError:
            continue
        by_size[size].append(f)

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for size, candidates in by_size.items():
        if len(candidates) < 2:
            continue
        for candidate in candidates:
            digest = _hash_file(candidate)
            if digest is not None:
                by_hash[digest].append(candidate)

    return {digest: paths for digest, paths in by_hash.items() if len(paths) > 1}


def find_duplicate_groups(files: Iterable[Path]) -> list[dict[str, object]]:
    """Return duplicate groups (size > 1) with summary metadata for each group.

    Each group is a dict with ``hash``, ``paths``, ``count``, and
    ``size_bytes`` (the size of a single copy) and ``wasted_bytes`` (bytes
    that could be reclaimed by keeping only one copy).
    """
    duplicates = find_exact_duplicates(files)
    groups = []
    for digest, paths in duplicates.items():
        try:
            size_bytes = paths[0].stat().st_size
        except OSError:
            size_bytes = 0
        groups.append(
            {
                "hash": digest,
                "paths": paths,
                "count": len(paths),
                "size_bytes": size_bytes,
                "wasted_bytes": size_bytes * (len(paths) - 1),
            }
        )
    groups.sort(key=lambda g: g["wasted_bytes"], reverse=True)
    return groups
