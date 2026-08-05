"""Tests for ai_rganize.config.exclusions (gitignore-like matching)."""

from __future__ import annotations

from ai_rganize.config.exclusions import (
    DEFAULT_PROTECTED_PATTERNS,
    ExclusionMatcher,
    build_matcher_for_directory,
    load_ignore_file,
)


def test_default_protected_patterns_present():
    assert "**/.git/**" in DEFAULT_PROTECTED_PATTERNS
    assert "**/node_modules/**" in DEFAULT_PROTECTED_PATTERNS
    assert "**/.venv/**" in DEFAULT_PROTECTED_PATTERNS
    assert "**/Library/**" in DEFAULT_PROTECTED_PATTERNS
    assert "**/.ai_rganize_manifest.json" in DEFAULT_PROTECTED_PATTERNS


def test_node_modules_excluded_by_default(tmp_path):
    root = tmp_path / "project"
    nested = root / "node_modules" / "pkg" / "index.js"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")

    matcher = ExclusionMatcher([])
    assert matcher.is_excluded(nested, root) is True


def test_custom_pattern_excludes_matching_glob(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    tmp_file = root / "cache.tmp"
    tmp_file.write_text("x")

    matcher = ExclusionMatcher(["*.tmp"])
    assert matcher.is_excluded(tmp_file, root) is True


def test_non_matching_file_is_not_excluded(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    normal_file = root / "notes.txt"
    normal_file.write_text("x")

    matcher = ExclusionMatcher([])
    assert matcher.is_excluded(normal_file, root) is False


def test_directory_style_pattern_with_trailing_slash(tmp_path):
    root = tmp_path / "project"
    nested = root / "build" / "output.bin"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")

    matcher = ExclusionMatcher(["build/"])
    assert matcher.is_excluded(nested, root) is True


def test_load_ignore_file_skips_comments_and_blanks(tmp_path):
    ignore_file = tmp_path / ".airganizeignore"
    ignore_file.write_text(
        "\n".join(
            [
                "# comment line",
                "",
                "*.log",
                "  ",
                "secrets/**",
            ]
        )
    )
    patterns = load_ignore_file(ignore_file)
    assert patterns == ["*.log", "secrets/**"]


def test_load_ignore_file_missing_returns_empty(tmp_path):
    assert load_ignore_file(tmp_path / "does_not_exist") == []


def test_build_matcher_for_directory_combines_ignore_file_and_profile_exclusions(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".airganizeignore").write_text("*.log\n")

    matcher = build_matcher_for_directory(root, extra_patterns=["*.bak"])

    log_file = root / "debug.log"
    bak_file = root / "old.bak"
    log_file.write_text("x")
    bak_file.write_text("x")

    assert matcher.is_excluded(log_file, root) is True
    assert matcher.is_excluded(bak_file, root) is True


def test_path_outside_root_is_excluded(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "other" / "file.txt"
    outside.parent.mkdir()
    outside.write_text("x")

    matcher = ExclusionMatcher([])
    assert matcher.is_excluded(outside, root) is True
