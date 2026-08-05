"""Filesystem safety helpers for path confinement and no-clobber moves."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _realpath(path: str | Path) -> str:
    """Expand ``~`` and resolve to a canonical absolute path."""
    text = os.fspath(path)
    if "\0" in text:
        raise ValueError("Invalid path: NUL byte")
    return os.path.realpath(os.path.expanduser(text))


def _is_under_root(resolved: str, root_real: str) -> bool:
    """Return True if *resolved* is *root_real* or a descendant.

    Uses ``startswith`` — the containment pattern CodeQL recognizes for
    ``py/path-injection`` (``os.path.commonpath`` / ``Path.is_relative_to``
    are not modeled as sanitizers).
    """
    return resolved == root_real or resolved.startswith(root_real + os.sep)


def default_allowed_roots() -> list[Path]:
    """Roots under which user-supplied paths may resolve.

    Includes the home directory (primary), the process temp dir (tests / scratch),
    and on macOS ``/Volumes`` for external drives.
    """
    roots = [Path.home(), Path(tempfile.gettempdir())]
    if sys.platform == "darwin":
        volumes = Path("/Volumes")
        if volumes.is_dir():
            roots.append(volumes)
    return roots


def is_within_directory(path: Path, directory: Path) -> bool:
    """Return True if *path* resolves inside *directory* (or is the directory)."""
    try:
        resolved = _realpath(path)
        root = _realpath(directory)
        return _is_under_root(resolved, root)
    except (ValueError, OSError):
        return False


def normalize_user_path(
    raw: str | Path,
    *,
    allowed_roots: list[Path] | None = None,
) -> Path:
    """Validate and resolve a user-supplied path under allowed roots.

    Confinement uses ``os.path.realpath`` + ``startswith`` so path-injection
    analyzers (e.g. CodeQL ``py/path-injection``) treat the result as sanitized.
    """
    if raw is None:
        raise ValueError("Path is required")
    text = os.fspath(raw).strip()
    if not text:
        raise ValueError("Path is required")
    if "\0" in text:
        raise ValueError("Invalid path")

    # Keep realpath at this call site (not only via a helper) so taint analysis
    # can see the canonicalization + startswith barrier on the same value.
    resolved = os.path.realpath(os.path.expanduser(text))
    roots = allowed_roots if allowed_roots is not None else default_allowed_roots()
    for root in roots:
        try:
            root_real = os.path.realpath(os.path.expanduser(os.fspath(root)))
        except (ValueError, OSError, TypeError):
            continue
        if resolved == root_real or resolved.startswith(root_real + os.sep):
            return Path(resolved)
    raise ValueError("Path outside allowed directories")


def ensure_destination_safe(dest: Path, root: Path) -> Path:
    """Resolve *dest* and raise if it would escape *root*."""
    root_resolved = os.path.realpath(os.path.expanduser(os.fspath(root)))
    parent = os.path.realpath(os.path.expanduser(os.fspath(dest.parent)))
    if not (parent == root_resolved or parent.startswith(root_resolved + os.sep)):
        raise ValueError(f"Destination escapes root: {dest}")
    return Path(parent) / Path(dest).name


def unique_destination(dest: Path) -> Path:
    """If *dest* exists, return a non-colliding path with _1, _2, ... suffix."""
    if not dest.exists():
        return dest
    stem, suffix, parent = dest.stem, dest.suffix, dest.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
        if counter > 10_000:
            raise FileExistsError(f"Could not find unique name for {dest}")


def is_symlink_or_through_symlink(path: Path) -> bool:
    """True if the path is a symlink or any parent component is a symlink."""
    try:
        current = Path(path)
        for _ in range(len(current.parts) + 1):
            if current.is_symlink():
                return True
            if current == current.parent:
                break
            current = current.parent
    except OSError:
        return True
    return False


def validate_restore_path(path: Path, allowed_root: Path) -> Path:
    """Ensure a restore destination stays under *allowed_root*."""
    return normalize_user_path(path, allowed_roots=[Path(allowed_root)])


def sanitize_folder_name(name: str) -> str:
    """Sanitize LLM-provided folder names to block traversal and invalid chars."""
    if not name:
        return "Unnamed_Folder"
    invalid_chars = '<>:"/\\|?*\x00'
    cleaned = "".join("_" if c in invalid_chars else c for c in name)
    cleaned = cleaned.replace("..", "_")
    cleaned = cleaned.strip(" .")
    if "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
        return "Unnamed_Folder"
    if len(cleaned) > 100:
        cleaned = cleaned[:100].rstrip(" .")
    return cleaned or "Unnamed_Folder"


def is_protected_path(path: Path) -> bool:
    """Return True for sensitive system/credential locations that must not be scanned."""
    try:
        resolved = os.path.realpath(os.path.expanduser(os.fspath(path)))
    except (ValueError, OSError, TypeError):
        return True

    home = os.path.realpath(os.path.expanduser(os.fspath(Path.home())))
    protected_roots = [
        os.path.join(home, "Library"),
        os.path.join(home, ".ssh"),
        os.path.join(home, ".gnupg"),
        os.path.join(home, ".aws"),
        os.path.join(home, ".config", "gcloud"),
    ]
    for protected in protected_roots:
        if resolved == protected or resolved.startswith(protected + os.sep):
            return True
    return False
