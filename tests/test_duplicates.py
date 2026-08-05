"""Tests for ai_rganize.features.duplicates."""

from __future__ import annotations

from pathlib import Path

from ai_rganize.features.duplicates import find_duplicate_groups, find_exact_duplicates


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_find_exact_duplicates_groups_identical_content(tmp_path):
    a = _write(tmp_path / "a.txt", "same content")
    b = _write(tmp_path / "b.txt", "same content")
    c = _write(tmp_path / "c.txt", "different content")

    duplicates = find_exact_duplicates([a, b, c])

    assert len(duplicates) == 1
    group = next(iter(duplicates.values()))
    assert set(group) == {a, b}


def test_find_exact_duplicates_no_duplicates_returns_empty(tmp_path):
    a = _write(tmp_path / "a.txt", "unique 1")
    b = _write(tmp_path / "b.txt", "unique 2")

    duplicates = find_exact_duplicates([a, b])
    assert duplicates == {}


def test_find_exact_duplicates_size_prefilter_skips_unique_sizes(tmp_path):
    a = _write(tmp_path / "a.txt", "x")
    b = _write(tmp_path / "b.txt", "yy")
    c = _write(tmp_path / "c.txt", "zzz")

    duplicates = find_exact_duplicates([a, b, c])
    assert duplicates == {}


def test_find_exact_duplicates_ignores_missing_files(tmp_path):
    a = _write(tmp_path / "a.txt", "same")
    missing = tmp_path / "does_not_exist.txt"

    duplicates = find_exact_duplicates([a, missing])
    assert duplicates == {}


def test_find_duplicate_groups_sorted_by_wasted_bytes(tmp_path):
    big1 = _write(tmp_path / "big1.txt", "B" * 1000)
    big2 = _write(tmp_path / "big2.txt", "B" * 1000)
    small1 = _write(tmp_path / "small1.txt", "s")
    small2 = _write(tmp_path / "small2.txt", "s")

    groups = find_duplicate_groups([big1, big2, small1, small2])

    assert len(groups) == 2
    assert groups[0]["wasted_bytes"] >= groups[1]["wasted_bytes"]
    assert groups[0]["count"] == 2


def test_find_duplicate_groups_triple_duplicate(tmp_path):
    files = [_write(tmp_path / f"dup_{i}.txt", "triple") for i in range(3)]
    groups = find_duplicate_groups(files)

    assert len(groups) == 1
    assert groups[0]["count"] == 3
    assert groups[0]["wasted_bytes"] == groups[0]["size_bytes"] * 2
