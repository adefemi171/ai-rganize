"""Filesystem safety helpers for path confinement and no-clobber moves."""

from __future__ import annotations

from pathlib import Path


def is_within_directory(path: Path, directory: Path) -> bool:
    """Return True if *path* resolves inside *directory* (or is the directory)."""
    try:
        resolved = path.expanduser().resolve()
        root = directory.expanduser().resolve()
        resolved.relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def ensure_destination_safe(dest: Path, root: Path) -> Path:
    """Resolve *dest* and raise if it would escape *root*."""
    root_resolved = root.expanduser().resolve()
    # Resolve via parent so not-yet-created destinations still work
    parent = dest.parent.expanduser().resolve()
    candidate = parent / dest.name
    if not is_within_directory(parent, root_resolved) and parent != root_resolved:
        raise ValueError(f"Destination escapes root: {dest}")
    try:
        candidate.resolve().relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Destination escapes root: {dest}") from exc
    return candidate


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
        current = path
        # Check the path and each parent without fully resolving first
        for _ in range(len(path.parts) + 1):
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
    resolved = path.expanduser().resolve()
    root = allowed_root.expanduser().resolve()
    if not is_within_directory(resolved, root) and resolved != root:
        raise ValueError(f"Restore path not under allowed root {allowed_root}: {path}")
    return resolved


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
        resolved = path.expanduser().resolve()
    except OSError:
        return True
    home = Path.home().resolve()
    protected_roots = [
        home / "Library",
        home / ".ssh",
        home / ".gnupg",
        home / ".aws",
        home / ".config" / "gcloud",
    ]
    for protected in protected_roots:
        if resolved == protected or is_within_directory(resolved, protected):
            return True
    return False
