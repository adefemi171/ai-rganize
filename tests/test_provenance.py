"""Tests for ai_rganize.provenance.ledger."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from ai_rganize.provenance.ledger import (
    LedgerRecord,
    append_record,
    append_records,
    compute_file_hash,
    list_run_ids,
    query,
    undo_preview,
)


def test_append_and_query_by_run_id(isolated_home):
    records = [
        LedgerRecord(run_id="run-1", source="/a/x.txt", dest="/b/Docs/x.txt", folder="Docs"),
        LedgerRecord(run_id="run-2", source="/a/y.txt", dest="/b/Photos/y.txt", folder="Photos"),
    ]
    written = append_records(records)
    assert written == 2

    run1_results = query(run_id="run-1")
    assert len(run1_results) == 1
    assert run1_results[0]["source"] == "/a/x.txt"


def test_ledger_file_is_jsonl(isolated_home):
    from ai_rganize.provenance.ledger import LEDGER_PATH

    append_record(LedgerRecord(run_id="run-1", source="/a", dest="/b", folder="Docs"))
    append_record(LedgerRecord(run_id="run-1", source="/c", dest="/d", folder="Docs"))

    lines = LEDGER_PATH.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "run_id" in parsed


def test_query_by_path_substring(isolated_home):
    append_records(
        [
            LedgerRecord(
                run_id="run-1",
                source="/Users/me/Downloads/report.pdf",
                dest="/Organized/Docs/report.pdf",
                folder="Docs",
            ),
            LedgerRecord(
                run_id="run-1",
                source="/Users/me/Downloads/photo.png",
                dest="/Organized/Photos/photo.png",
                folder="Photos",
            ),
        ]
    )

    results = query(path_contains="report.pdf")
    assert len(results) == 1
    assert results[0]["folder"] == "Docs"


def test_query_since_filters_older_records(isolated_home):
    old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    recent_time = datetime.now(timezone.utc).isoformat()

    append_records(
        [
            LedgerRecord(
                run_id="old", source="/a", dest="/b", folder="Docs", timestamp=old_time
            ),
            LedgerRecord(
                run_id="new", source="/c", dest="/d", folder="Docs", timestamp=recent_time
            ),
        ]
    )

    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    results = query(since=cutoff)
    assert len(results) == 1
    assert results[0]["run_id"] == "new"


def test_undo_preview_maps_dest_back_to_source(isolated_home):
    append_records(
        [
            LedgerRecord(
                run_id="run-1", source="/orig/a.txt", dest="/new/Docs/a.txt", folder="Docs"
            ),
        ]
    )

    preview = undo_preview("run-1")
    assert len(preview) == 1
    assert preview[0]["current_location"] == "/new/Docs/a.txt"
    assert preview[0]["would_restore_to"] == "/orig/a.txt"


def test_undo_preview_empty_for_unknown_run(isolated_home):
    assert undo_preview("does-not-exist") == []


def test_list_run_ids_preserves_order_deduplicated(isolated_home):
    append_records(
        [
            LedgerRecord(run_id="run-1", source="/a", dest="/b", folder="X"),
            LedgerRecord(run_id="run-2", source="/c", dest="/d", folder="Y"),
            LedgerRecord(run_id="run-1", source="/e", dest="/f", folder="X"),
        ]
    )
    assert list_run_ids() == ["run-1", "run-2"]


def test_query_empty_ledger_returns_empty_list(isolated_home):
    assert query() == []


def test_compute_file_hash(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    digest = compute_file_hash(f)
    assert digest is not None
    assert len(digest) == 64  # sha256 hex digest length


def test_compute_file_hash_missing_file_returns_none(tmp_path):
    assert compute_file_hash(tmp_path / "missing.txt") is None


def test_ledger_record_optional_fields_default_none():
    record = LedgerRecord(run_id="r", source="/a", dest="/b", folder="Docs")
    data = record.to_dict()
    assert data["reason"] is None
    assert data["provider"] is None
    assert data["confidence"] is None
    assert data["file_hash"] is None
