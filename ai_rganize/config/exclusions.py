"""Gitignore-like exclusion matching for AI-rganize.

Supports a ``.airganizeignore`` file (one glob pattern per line, ``#`` comments
and blank lines ignored) plus a set of always-on protected patterns that
prevent AI-rganize from ever touching sensitive locations.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

DEFAULT_PROTECTED_PATTERNS: tuple[str, ...] = (
    "**/.git/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/Library/**",
    "**/.ai_rganize_manifest.json",
)

IGNORE_FILENAME = ".airganizeignore"


def _normalize_pattern(pattern: str) -> str:
    pattern = pattern.strip()
    # A trailing slash means "directory", which we treat as "everything under it".
    if pattern.endswith("/"):
        pattern = pattern.rstrip("/") + "/**"
    return pattern


def _candidate_strings(relative_posix: str) -> list[str]:
    """Build the set of strings we'll test a pattern against.

    This lets a pattern like ``*.tmp`` match ``foo/bar.tmp`` (matching just the
    basename) as well as a full-path pattern like ``**/.git/**``.
    """
    candidates = {relative_posix, "/" + relative_posix}
    name = relative_posix.rsplit("/", 1)[-1]
    candidates.add(name)
    return list(candidates)


def _pattern_matches(pattern: str, relative_posix: str) -> bool:
    pattern = _normalize_pattern(pattern)
    if not pattern:
        return False

    for candidate in _candidate_strings(relative_posix):
        if fnmatch.fnmatch(candidate, pattern):
            return True
        # Also allow a leading "**/" pattern to match at the root (no prefix dirs).
        if pattern.startswith("**/") and fnmatch.fnmatch(candidate, pattern[3:]):
            return True
    return False


class ExclusionMatcher:
    """Matches paths against a combined set of ignore patterns."""

    def __init__(self, patterns: list[str] | None = None, include_defaults: bool = True):
        self.patterns: list[str] = []
        if include_defaults:
            self.patterns.extend(DEFAULT_PROTECTED_PATTERNS)
        if patterns:
            self.patterns.extend(p for p in patterns if p and not p.strip().startswith("#"))

    def is_excluded(self, path: Path, root: Path) -> bool:
        """Return True if *path* should be excluded, relative to *root*."""
        try:
            relative = Path(path).resolve().relative_to(Path(root).resolve())
        except ValueError:
            # Not under root at all -- treat as excluded to be safe.
            return True
        relative_posix = relative.as_posix()

        for pattern in self.patterns:
            if _pattern_matches(pattern, relative_posix):
                return True
        return False

    def add_pattern(self, pattern: str) -> None:
        self.patterns.append(pattern)

    @classmethod
    def from_ignore_file(cls, path: Path, include_defaults: bool = True) -> "ExclusionMatcher":
        return cls(load_ignore_file(path), include_defaults=include_defaults)


def load_ignore_file(path: Path) -> list[str]:
    """Read a ``.airganizeignore``-style file into a list of glob patterns."""
    path = Path(path)
    if not path.exists():
        return []
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def find_ignore_file(directory: Path) -> Path | None:
    """Look for an ``.airganizeignore`` file directly inside *directory*."""
    candidate = Path(directory) / IGNORE_FILENAME
    return candidate if candidate.exists() else None


def build_matcher_for_directory(
    directory: Path, extra_patterns: list[str] | None = None
) -> ExclusionMatcher:
    """Convenience helper: build a matcher combining defaults, an
    ``.airganizeignore`` file (if present) and *extra_patterns* (e.g. from a
    Profile's ``exclusions`` field).
    """
    ignore_file = find_ignore_file(directory)
    patterns = load_ignore_file(ignore_file) if ignore_file else []
    if extra_patterns:
        patterns.extend(extra_patterns)
    return ExclusionMatcher(patterns)
