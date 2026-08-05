"""Append-only provenance ledger for AI-rganize operations.

Every file move made by AI-rganize can be recorded here as a JSON Lines
record, independent of (and in addition to) the per-run manifest files used
for immediate undo. The ledger accumulates history across every run, making
it possible to answer questions like "where did this file come from?" or
"what changed on 2024-01-05?" long after any single run's manifest has been
cleaned up.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

LEDGER_HOME = Path.home() / ".ai_rganize"
LEDGER_PATH = LEDGER_HOME / "ledger.jsonl"


@dataclass
class LedgerRecord:
    """A single provenance record for one file move."""

    run_id: str
    source: str
    dest: str
    folder: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    confidence: Optional[float] = None
    file_hash: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LedgerRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def compute_file_hash(path: Path, chunk_size: int = 65536) -> Optional[str]:
    """Compute the sha256 hash of a file, returning None if it can't be read."""
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


def _ledger_path() -> Path:
    LEDGER_HOME.mkdir(parents=True, exist_ok=True)
    return LEDGER_PATH


def append_records(
    records: Iterable[LedgerRecord | dict[str, Any]], path: Optional[Path] = None
) -> int:
    """Append *records* to the ledger, one JSON object per line. Returns count written."""
    ledger_path = Path(path) if path is not None else _ledger_path()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(ledger_path, "a", encoding="utf-8") as fh:
        for record in records:
            data = record.to_dict() if isinstance(record, LedgerRecord) else dict(record)
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_record(record: LedgerRecord | dict[str, Any], path: Optional[Path] = None) -> None:
    append_records([record], path=path)


def _iter_raw_records(path: Optional[Path] = None) -> Iterable[dict[str, Any]]:
    ledger_path = Path(path) if path is not None else LEDGER_PATH
    if not ledger_path.exists():
        return
    with open(ledger_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def query(
    run_id: Optional[str] = None,
    path_contains: Optional[str] = None,
    since: Optional[str | datetime] = None,
    path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Query the ledger, filtering by run id, a path substring, and/or a
    minimum ISO-8601 timestamp (``since``)."""
    if isinstance(since, datetime):
        since_dt = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
    elif isinstance(since, str):
        since_dt = datetime.fromisoformat(since)
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)
    else:
        since_dt = None

    results = []
    for record in _iter_raw_records(path):
        if run_id is not None and record.get("run_id") != run_id:
            continue
        if path_contains is not None:
            haystack = f"{record.get('source', '')} {record.get('dest', '')}"
            if path_contains not in haystack:
                continue
        if since_dt is not None:
            ts = record.get("timestamp")
            try:
                record_dt = datetime.fromisoformat(ts) if ts else None
            except ValueError:
                record_dt = None
            if record_dt is not None and record_dt.tzinfo is None:
                record_dt = record_dt.replace(tzinfo=timezone.utc)
            if record_dt is None or record_dt < since_dt:
                continue
        results.append(record)
    return results


def undo_preview(run_id: str, path: Optional[Path] = None) -> list[dict[str, str]]:
    """Return the list of (dest -> source) moves that undoing *run_id* would perform.

    This does not modify anything on disk; it is purely informational so a
    caller can confirm with the user before acting.
    """
    records = query(run_id=run_id, path=path)
    preview = []
    for record in records:
        preview.append(
            {
                "current_location": record.get("dest", ""),
                "would_restore_to": record.get("source", ""),
                "folder": record.get("folder", ""),
            }
        )
    return preview


def list_run_ids(path: Optional[Path] = None) -> list[str]:
    """Return the distinct run ids present in the ledger, most recent last."""
    seen: dict[str, None] = {}
    for record in _iter_raw_records(path):
        run_id = record.get("run_id")
        if run_id:
            seen[run_id] = None
    return list(seen.keys())
