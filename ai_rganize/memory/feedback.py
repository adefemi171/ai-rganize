"""Per-profile feedback memory.

Stores lightweight records of user decisions (e.g. "I moved invoice.pdf to
Finance/Invoices and approved it" or "I rejected the AI's guess for
photo.png") so future runs can be biased toward previously-approved
folder/file patterns. This is intentionally simple JSON storage -- no
embeddings or vector search -- to keep the feature dependency-free.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MEMORY_HOME = Path.home() / ".ai_rganize" / "memory"


@dataclass
class Decision:
    """A single recorded user decision about a file's categorization."""

    filename: str
    extension: str
    folder: str
    action: str  # "approved" | "rejected" | "edited"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _memory_path(profile: str) -> Path:
    MEMORY_HOME.mkdir(parents=True, exist_ok=True)
    safe_name = profile.strip() or "default"
    return MEMORY_HOME / f"{safe_name}.json"


def _load_all(profile: str) -> list[dict[str, Any]]:
    path = _memory_path(profile)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _save_all(profile: str, decisions: list[dict[str, Any]]) -> None:
    path = _memory_path(profile)
    path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")


def record_decision(
    profile: str,
    filename: str,
    folder: str,
    action: str,
    reason: Optional[str] = None,
) -> Decision:
    """Record a single user decision for *profile*, appending to its memory file."""
    extension = Path(filename).suffix.lower()
    decision = Decision(
        filename=filename, extension=extension, folder=folder, action=action, reason=reason
    )

    decisions = _load_all(profile)
    decisions.append(decision.to_dict())
    _save_all(profile, decisions)
    return decision


def get_exemplars(profile: str, extension: Optional[str] = None, limit: int = 20) -> list[Decision]:
    """Return past approved/edited decisions for *profile*, most recent first.

    If *extension* is given, only decisions for files with that extension are
    returned -- useful for biasing future categorization of similar files.
    """
    decisions = [Decision.from_dict(d) for d in _load_all(profile)]
    decisions = [d for d in decisions if d.action in ("approved", "edited")]
    if extension is not None:
        ext = extension.lower()
        decisions = [d for d in decisions if d.extension == ext]
    decisions.sort(key=lambda d: d.timestamp, reverse=True)
    return decisions[:limit]


def folder_affinity(profile: str, extension: str) -> dict[str, int]:
    """Return a count of how many times each folder was used/approved for a
    given file extension, useful for suggesting a default destination."""
    counts: dict[str, int] = {}
    for decision in get_exemplars(profile, extension=extension, limit=10_000):
        counts[decision.folder] = counts.get(decision.folder, 0) + 1
    return counts


def forget(profile: str, filename: Optional[str] = None) -> int:
    """Remove memory for *profile*. If *filename* is given, only remove
    decisions about that filename; otherwise wipe the whole profile's memory.

    Returns the number of records removed.
    """
    if filename is None:
        path = _memory_path(profile)
        if path.exists():
            existing = _load_all(profile)
            path.unlink()
            return len(existing)
        return 0

    decisions = _load_all(profile)
    remaining = [d for d in decisions if d.get("filename") != filename]
    removed = len(decisions) - len(remaining)
    _save_all(profile, remaining)
    return removed
