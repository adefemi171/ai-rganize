"""Security-focused path handling tests: traversal, escapes, restore safety.

These complement test_safety.py (which unit-tests each primitive) by
exercising realistic attack scenarios end-to-end.
"""

from __future__ import annotations

import pytest

from ai_rganize.config.exclusions import ExclusionMatcher
from ai_rganize.utils.safety import (
    ensure_destination_safe,
    validate_restore_path,
)


@pytest.mark.parametrize(
    "malicious_relative",
    [
        "../../etc/passwd",
        "../outside.txt",
        "a/../../b/../../etc/hosts",
    ],
)
def test_ensure_destination_safe_blocks_dot_dot_traversal(tmp_path, malicious_relative):
    root = tmp_path / "safe_root"
    root.mkdir()
    malicious_dest = (root / malicious_relative)
    with pytest.raises(ValueError):
        ensure_destination_safe(malicious_dest, root)


def test_ensure_destination_safe_blocks_absolute_escape(tmp_path):
    root = tmp_path / "safe_root"
    root.mkdir()
    absolute_elsewhere = tmp_path / "definitely_outside" / "secret.txt"
    with pytest.raises(ValueError):
        ensure_destination_safe(absolute_elsewhere, root)


def test_validate_restore_path_blocks_traversal_outside_allowed_root(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    traversal_attempt = allowed_root / ".." / ".." / "etc" / "passwd"
    with pytest.raises(ValueError):
        validate_restore_path(traversal_attempt, allowed_root)


def test_validate_restore_path_allows_legitimate_nested_restore(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    legit = allowed_root / "Documents" / "Invoices" / "file.pdf"
    result = validate_restore_path(legit, allowed_root)
    assert str(result).startswith(str(allowed_root.resolve()))


def test_exclusions_protect_git_directory(tmp_path):
    root = tmp_path / "project"
    (root / ".git" / "objects").mkdir(parents=True)
    target = root / ".git" / "objects" / "abcd1234"
    target.write_text("data")

    matcher = ExclusionMatcher([])
    assert matcher.is_excluded(target, root) is True


def test_exclusions_protect_manifest_file(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    manifest = root / ".ai_rganize_manifest.json"
    manifest.write_text("{}")

    matcher = ExclusionMatcher([])
    assert matcher.is_excluded(manifest, root) is True


def test_path_escaping_root_via_symlink_dir_is_flagged_by_is_within_directory(tmp_path):
    from ai_rganize.utils.safety import is_within_directory

    root = tmp_path / "root"
    root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    escape_link = root / "escape"
    escape_link.symlink_to(outside_dir, target_is_directory=True)

    escaped_target = escape_link / "secret.txt"
    # Resolving through the symlink lands outside root -- must not be considered "within".
    assert is_within_directory(escaped_target, root) is False
