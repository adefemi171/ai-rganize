"""Detection of local cloud-sync folders.

This module never talks to any remote API -- it only looks for the
well-known local directories that Dropbox, iCloud Drive, and Google Drive
create on disk when their desktop sync clients are installed.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional


def _existing(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _dropbox_root(home: Path) -> Optional[Path]:
    return _existing(
        home / "Dropbox",
        home / "Library" / "CloudStorage" / "Dropbox",
    )


def _icloud_root(home: Path) -> Optional[Path]:
    return _existing(home / "Library" / "Mobile Documents" / "com~apple~CloudDocs")


def _google_drive_root(home: Path) -> Optional[Path]:
    direct = _existing(home / "Google Drive")
    if direct:
        return direct
    matches = sorted(glob.glob(str(home / "Library" / "CloudStorage" / "GoogleDrive-*")))
    for match in matches:
        path = Path(match)
        if path.is_dir():
            return path
    return None


_PROVIDER_DETECTORS = {
    "dropbox": _dropbox_root,
    "icloud": _icloud_root,
    "google_drive": _google_drive_root,
}


def discover_cloud_roots(home: Optional[Path] = None) -> dict[str, Path]:
    """Return a mapping of provider name -> local sync folder, for providers detected."""
    home = Path(home) if home is not None else Path.home()
    found: dict[str, Path] = {}
    for name, detector in _PROVIDER_DETECTORS.items():
        root = detector(home)
        if root is not None:
            found[name] = root
    return found


def get_cloud_root(provider_name: str, home: Optional[Path] = None) -> Optional[Path]:
    """Return the local sync folder for *provider_name* (e.g. 'dropbox'), or None."""
    home = Path(home) if home is not None else Path.home()
    detector = _PROVIDER_DETECTORS.get(provider_name.lower().replace(" ", "_"))
    if detector is None:
        return None
    return detector(home)


def known_providers() -> list[str]:
    return list(_PROVIDER_DETECTORS.keys())
