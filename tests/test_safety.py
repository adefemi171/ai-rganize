"""Unit tests for ai_rganize.utils.safety primitives."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_rganize.utils.safety import (
    ensure_destination_safe,
    is_protected_path,
    is_symlink_or_through_symlink,
    is_within_directory,
    normalize_user_path,
    sanitize_folder_name,
    unique_destination,
    validate_restore_path,
)


def test_is_within_directory_true_for_nested_path(tmp_path):
    root = tmp_path / "root"
    nested = root / "a" / "b" / "file.txt"
    assert is_within_directory(nested, root) is True


def test_is_within_directory_false_for_sibling(tmp_path):
    root = tmp_path / "root"
    sibling = tmp_path / "other" / "file.txt"
    assert is_within_directory(sibling, root) is False


def test_is_within_directory_false_for_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    escaping = root / ".." / "escaped.txt"
    assert is_within_directory(escaping, root) is False


def test_ensure_destination_safe_allows_normal_path(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    dest = root / "Documents" / "file.txt"
    result = ensure_destination_safe(dest, root)
    assert str(result).startswith(str(root.resolve()))


def test_ensure_destination_safe_rejects_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    dest = tmp_path / "outside" / "file.txt"
    with pytest.raises(ValueError):
        ensure_destination_safe(dest, root)


def test_unique_destination_no_clobber(tmp_path):
    dest = tmp_path / "file.txt"
    dest.write_text("original")
    unique = unique_destination(dest)
    assert unique != dest
    assert unique.name == "file_1.txt"


def test_unique_destination_increments(tmp_path):
    (tmp_path / "file.txt").write_text("a")
    (tmp_path / "file_1.txt").write_text("b")
    unique = unique_destination(tmp_path / "file.txt")
    assert unique.name == "file_2.txt"


def test_unique_destination_returns_same_if_free(tmp_path):
    dest = tmp_path / "brand_new.txt"
    assert unique_destination(dest) == dest


def test_is_symlink_or_through_symlink_direct(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("hi")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    assert is_symlink_or_through_symlink(link) is True


def test_is_symlink_or_through_symlink_false_for_regular_file(tmp_path):
    regular = tmp_path / "regular.txt"
    regular.write_text("hi")
    assert is_symlink_or_through_symlink(regular) is False


def test_is_symlink_or_through_symlink_parent_symlink(tmp_path):
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked_dir"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    nested_file = linked_dir / "file.txt"
    real_dir.joinpath("file.txt").write_text("hi")
    assert is_symlink_or_through_symlink(nested_file) is True


def test_validate_restore_path_within_root(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    target = root / "subdir" / "file.txt"
    result = validate_restore_path(target, root)
    assert str(result).startswith(str(root.resolve()))


def test_validate_restore_path_rejects_outside_root(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "file.txt"
    with pytest.raises(ValueError):
        validate_restore_path(outside, root)


def test_sanitize_folder_name_strips_invalid_chars():
    assert sanitize_folder_name('bad<>:"/\\|?*name') == "bad_________name"


def test_sanitize_folder_name_blocks_traversal():
    # ".." alone and slash-containing traversal attempts must never survive
    # sanitization intact -- no ".." run and no path separators left behind.
    for dangerous in ("..", "../../etc", "..\\..\\windows"):
        result = sanitize_folder_name(dangerous)
        assert ".." not in result
        assert "/" not in result
        assert "\\" not in result


def test_sanitize_folder_name_empty_returns_default():
    assert sanitize_folder_name("") == "Unnamed_Folder"


def test_sanitize_folder_name_truncates_long_names():
    long_name = "a" * 200
    result = sanitize_folder_name(long_name)
    assert len(result) <= 100


def test_is_protected_path_library():
    library_path = Path.home() / "Library" / "Something"
    assert is_protected_path(library_path) is True


def test_is_protected_path_ssh():
    ssh_path = Path.home() / ".ssh" / "id_rsa"
    assert is_protected_path(ssh_path) is True


def test_is_protected_path_regular_dir_not_protected(tmp_path):
    regular = tmp_path / "Documents" / "file.txt"
    assert is_protected_path(regular) is False


def test_normalize_user_path_allows_under_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "Downloads"
    target.mkdir()
    assert normalize_user_path(str(target), allowed_roots=[tmp_path]) == target.resolve()


def test_normalize_user_path_rejects_outside_allowed_root(tmp_path):
    outside = tmp_path / "allowed"
    outside.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError):
        normalize_user_path(str(other), allowed_roots=[outside])


def test_normalize_user_path_rejects_nul():
    with pytest.raises(ValueError):
        normalize_user_path("/tmp/evil\0name")
