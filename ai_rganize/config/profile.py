"""User profile configuration for AI-rganize.

Profiles capture a reusable set of organization preferences (roots to scan,
destination, AI provider/model, budgets, exclusions, scheduling, etc.) and are
persisted as YAML (preferred) or JSON (fallback when PyYAML is unavailable)
under ``~/.ai_rganize/profiles/<name>.yaml``.

Project-local overrides are supported via a ``.airganize.yaml`` file that can
live in a project directory (or any ancestor of it), mirroring tools like
``.eslintrc`` / ``.flake8``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml as _yaml

    HAS_YAML = True
except ImportError:  # pragma: no cover - exercised when PyYAML isn't installed
    _yaml = None
    HAS_YAML = False

PROFILE_HOME = Path.home() / ".ai_rganize"
PROFILES_DIR = PROFILE_HOME / "profiles"
PROJECT_LOCAL_FILENAME = ".airganize.yaml"

DEFAULT_PROFILE_NAME = "default"


@dataclass
class Profile:
    """A reusable AI-rganize configuration profile."""

    name: str = DEFAULT_PROFILE_NAME
    roots: list[str] = field(default_factory=list)
    destination: Optional[str] = None
    provider: str = "openai"
    model: Optional[str] = None
    max_cost: float = 1.0
    max_folders: Optional[int] = None
    exclusions: list[str] = field(default_factory=list)
    schedule: Optional[str] = None
    enable_council: bool = True
    auto_unpack_archives: bool = False
    cloud_providers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Profile":
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in (data or {}).items() if k in known_fields}
        return cls(**filtered)


def _profiles_dir() -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return PROFILES_DIR


def resolve_profile_path(name: str, ensure_parent: bool = False) -> Path:
    """Return the path a profile named *name* would be stored at.

    Prefers ``.yaml`` when PyYAML is available; otherwise resolves to
    ``.json``. If a profile already exists on disk under either extension,
    that existing path is returned instead.
    """
    safe_name = name.strip() or DEFAULT_PROFILE_NAME
    directory = _profiles_dir() if ensure_parent else PROFILES_DIR

    yaml_path = directory / f"{safe_name}.yaml"
    json_path = directory / f"{safe_name}.json"

    if yaml_path.exists():
        return yaml_path
    if json_path.exists():
        return json_path

    return yaml_path if HAS_YAML else json_path


def _dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".yaml", ".yml") and HAS_YAML:
        with open(path, "w", encoding="utf-8") as fh:
            _yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        if HAS_YAML:
            return _yaml.safe_load(text) or {}
        # Fall back to a best-effort JSON parse in case the file is actually JSON.
        return json.loads(text) if text.strip() else {}
    return json.loads(text) if text.strip() else {}


def save_profile(profile: Profile) -> Path:
    """Persist *profile* to disk and return the path written."""
    path = resolve_profile_path(profile.name, ensure_parent=True)
    _dump(profile.to_dict(), path)
    return path


def load_profile(name: str) -> Profile:
    """Load a profile by name, raising ``FileNotFoundError`` if missing."""
    path = resolve_profile_path(name)
    if not path.exists():
        raise FileNotFoundError(f"No profile named '{name}' found at {path}")
    return Profile.from_dict(_load(path))


def list_profiles() -> list[str]:
    """Return the names of all saved profiles, sorted alphabetically."""
    if not PROFILES_DIR.exists():
        return []
    names = set()
    for p in PROFILES_DIR.iterdir():
        if p.suffix in (".yaml", ".yml", ".json") and p.is_file():
            names.add(p.stem)
    return sorted(names)


def default_profile() -> Profile:
    """Return the default profile, creating it on disk if it doesn't exist."""
    try:
        return load_profile(DEFAULT_PROFILE_NAME)
    except FileNotFoundError:
        profile = Profile(name=DEFAULT_PROFILE_NAME)
        save_profile(profile)
        return profile


def delete_profile(name: str) -> bool:
    """Delete a profile by name. Returns True if a file was removed."""
    removed = False
    for suffix in (".yaml", ".yml", ".json"):
        candidate = PROFILES_DIR / f"{name}{suffix}"
        if candidate.exists():
            candidate.unlink()
            removed = True
    return removed


def find_project_profile(start_dir: Optional[Path] = None) -> Optional[Path]:
    """Search *start_dir* and its ancestors for a ``.airganize.yaml`` file."""
    current = Path(start_dir or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / PROJECT_LOCAL_FILENAME
        if candidate.exists():
            return candidate
    return None


def load_project_profile(start_dir: Optional[Path] = None) -> Optional[Profile]:
    """Load the nearest project-local ``.airganize.yaml`` profile, if any."""
    path = find_project_profile(start_dir)
    if path is None:
        return None
    return Profile.from_dict(_load(path))


def save_project_profile(profile: Profile, directory: Optional[Path] = None) -> Path:
    """Save *profile* as a project-local ``.airganize.yaml`` in *directory*."""
    target_dir = Path(directory or Path.cwd())
    path = target_dir / PROJECT_LOCAL_FILENAME
    if not HAS_YAML:
        path = target_dir / ".airganize.json"
    _dump(profile.to_dict(), path)
    return path


def resolve_effective_profile(
    name: Optional[str] = None, start_dir: Optional[Path] = None
) -> Profile:
    """Resolve the profile that should be used, preferring an explicit *name*.

    Resolution order: explicit named profile > project-local ``.airganize.yaml``
    > global default profile.
    """
    if name:
        return load_profile(name)
    project_profile = load_project_profile(start_dir)
    if project_profile is not None:
        return project_profile
    return default_profile()
