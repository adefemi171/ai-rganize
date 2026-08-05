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
        return os.path.commonpath([resolved, root]) == root
    except (ValueError, OSError):
        return False


def normalize_user_path(
    raw: str | Path,
    *,
    allowed_roots: list[Path] | None = None,
) -> Path:
    """Validate and resolve a user-supplied path under allowed roots.

    Confinement uses ``os.path.realpath`` + ``os.path.commonpath`` so path-injection
    analyzers (e.g. CodeQL) can see the check as a sanitizer.
    """
    if raw is None:
        raise ValueError("Path is required")
    text = os.fspath(raw).strip()
    if not text:
        raise ValueError("Path is required")
    if "\0" in text:
        raise ValueError("Invalid path")

    resolved = _realpath(text)
    roots = allowed_roots if allowed_roots is not None else default_allowed_roots()
    for root in roots:
        try:
            root_real = _realpath(root)
        except (ValueError, OSError):
            continue
        try:
            if os.path.commonpath([resolved, root_real]) == root_real:
                return Path(resolved)
        except ValueError:
            continue
    raise ValueError("Path outside allowed directories")


def ensure_destination_safe(dest: Path, root: Path) -> Path:
    """Resolve *dest* and raise if it would escape *root*."""
    root_resolved = _realpath(root)
    parent = _realpath(dest.parent)
    try:
        if os.path.commonpath([parent, root_resolved]) != root_resolved:
            raise ValueError(f"Destination escapes root: {dest}")
    except ValueError as exc:
        # commonpath raises ValueError when paths are on different drives /
        # when the destination is outside root.
        if str(exc).startswith("Destination escapes root:"):
            raise
        raise ValueError(f"Destination escapes root: {dest}") from exc
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
        resolved = _realpath(path)
    except (ValueError, OSError):
        return True

    home = _realpath(Path.home())
    protected_roots = [
        os.path.join(home, "Library"),
        os.path.join(home, ".ssh"),
        os.path.join(home, ".gnupg"),
        os.path.join(home, ".aws"),
        os.path.join(home, ".config", "gcloud"),
    ]
    for protected in protected_roots:
        try:
            if os.path.commonpath([resolved, protected]) == protected:
                return True
        except ValueError:
            continue
    return False
