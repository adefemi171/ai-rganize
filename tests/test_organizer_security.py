"""Security tests for organizer execute/scan/restore paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_rganize.organizer.base_organizer import BaseOrganizer
from ai_rganize.utils.metadata import (
    capture_metadata,
    create_manifest,
    load_manifest,
    move_preserving_metadata,
    restore_from_manifest,
    save_manifest,
)
from ai_rganize.utils.safety import sanitize_folder_name


def test_library_not_in_default_targets():
    org = BaseOrganizer()
    assert "Library" not in org.target_dirs


def test_scan_skips_symlinks(tmp_path: Path):
    real = tmp_path / "real.txt"
    real.write_text("hi")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    org = BaseOrganizer()
    scanned = org.scan_files(tmp_path)
    names = {f["name"] for f in scanned}
    assert "real.txt" in names
    assert "link.txt" not in names


def test_no_clobber_move(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("aaa")
    b.write_text("bbb")
    with pytest.raises(FileExistsError):
        move_preserving_metadata(a, b)


def test_execute_uses_unique_destination(tmp_path: Path):
    existing = tmp_path / "Docs" / "note.txt"
    existing.parent.mkdir()
    existing.write_text("old")
    source = tmp_path / "note.txt"
    source.write_text("new")
    org = BaseOrganizer()
    plan = {"Docs": [{"path": source, "name": "note.txt", "size": 3}]}
    assert org.execute_organization(plan, tmp_path)
    assert existing.read_text() == "old"
    assert (tmp_path / "Docs" / "note_1.txt").read_text() == "new"


def test_execute_rejects_path_escape(tmp_path: Path):
    cleaned = sanitize_folder_name("../evil")
    assert cleaned == "Unnamed_Folder" or ".." not in cleaned


def test_restore_rejects_outside_root(tmp_path: Path):
    inside = tmp_path / "inside"
    inside.mkdir()
    moved = inside / "file.txt"
    moved.write_text("data")
    outside = tmp_path / "outside_original.txt"

    manifest = create_manifest(inside)
    meta = capture_metadata(moved)
    # Forge a malicious restore destination outside root
    manifest.add_move(outside, moved, "Docs", meta)
    path = save_manifest(manifest, inside)

    loaded = load_manifest(path)
    ok, failed = restore_from_manifest(loaded, verbose=False, allowed_root=inside)
    assert failed >= 1
    assert moved.exists()  # should not have been moved out


def test_display_plan_uses_target_dir(tmp_path: Path, capsys):
    org = BaseOrganizer()
    f = tmp_path / "a.txt"
    f.write_text("x")
    plan = {"Docs": [{"path": f, "name": "a.txt", "size": 1}]}
    org.display_organization_plan(plan, show_details=True, target_dir=tmp_path)
    captured = capsys.readouterr().out.replace("\n", "")
    # Rich may wrap long paths; assert the destination suffix is present
    assert "Docs/a.txt" in captured
