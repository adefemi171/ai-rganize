"""Tests for ai_rganize.features.archives, focused heavily on zip-slip and
other archive-based path traversal / resource-exhaustion attacks."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from ai_rganize.features.archives import (
    UnsafeArchiveError,
    is_rar_archive,
    is_supported_archive,
    list_archive_contents,
    organize_archives_in_dir,
    safe_extract,
)


def _make_zip_slip_archive(path: Path) -> Path:
    """A zip whose single member tries to escape via '../../'."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("../../evil.txt", "pwned")
    return path


def _make_absolute_path_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("/etc/evil.txt", "pwned")
    return path


def _make_normal_zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


def _make_tar_slip_archive(path: Path) -> Path:
    with tarfile.open(path, "w") as tf:
        data = b"pwned"
        info = tarfile.TarInfo(name="../../evil.txt")
        info.size = len(data)
        import io

        tf.addfile(info, io.BytesIO(data))
    return path


def test_safe_extract_blocks_zip_slip(tmp_path):
    archive = _make_zip_slip_archive(tmp_path / "evil.zip")
    dest_dir = tmp_path / "extract_here"

    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, dest_dir)

    # Nothing should have leaked outside dest_dir.
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_blocks_absolute_path_member(tmp_path):
    archive = _make_absolute_path_zip(tmp_path / "abs.zip")
    dest_dir = tmp_path / "extract_here"

    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, dest_dir)

    assert not Path("/etc/evil.txt").exists()


def test_safe_extract_blocks_tar_slip(tmp_path):
    archive = _make_tar_slip_archive(tmp_path / "evil.tar")
    dest_dir = tmp_path / "extract_here"

    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, dest_dir)

    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_normal_zip_succeeds(tmp_path):
    archive = _make_normal_zip(tmp_path / "good.zip", {"a.txt": "hello", "sub/b.txt": "world"})
    dest_dir = tmp_path / "extract_here"

    extracted = safe_extract(archive, dest_dir)

    assert len(extracted) == 2
    for path in extracted:
        assert dest_dir.resolve() in path.resolve().parents or path.resolve() == dest_dir.resolve()
    assert (dest_dir / "a.txt").read_text() == "hello"
    assert (dest_dir / "sub" / "b.txt").read_text() == "world"


def test_safe_extract_enforces_max_files(tmp_path):
    archive_path = tmp_path / "many.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        for i in range(5):
            zf.writestr(f"file_{i}.txt", "x")

    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive_path, tmp_path / "out", max_files=2)


def test_safe_extract_enforces_max_total_bytes(tmp_path):
    archive_path = tmp_path / "big.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("big.txt", "x" * 1000)

    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive_path, tmp_path / "out", max_total_bytes=100)


def test_list_archive_contents_zip(tmp_path):
    archive = _make_normal_zip(tmp_path / "list.zip", {"a.txt": "1", "b.txt": "2"})
    contents = list_archive_contents(archive)
    assert set(contents) == {"a.txt", "b.txt"}


def test_list_archive_contents_tar_gz(tmp_path):
    archive_path = tmp_path / "list.tar.gz"
    inner_file = tmp_path / "inner.txt"
    inner_file.write_text("data")
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(inner_file, arcname="inner.txt")

    contents = list_archive_contents(archive_path)
    assert "inner.txt" in contents


def test_rar_archive_rejected(tmp_path):
    fake_rar = tmp_path / "archive.rar"
    fake_rar.write_bytes(b"Rar!\x1a\x07\x00")

    assert is_rar_archive(fake_rar) is True
    assert is_supported_archive(fake_rar) is False

    with pytest.raises(UnsafeArchiveError):
        list_archive_contents(fake_rar)

    with pytest.raises(UnsafeArchiveError):
        safe_extract(fake_rar, tmp_path / "out")


def test_organize_archives_in_dir_inventory_only(tmp_path):
    _make_normal_zip(tmp_path / "docs.zip", {"readme.txt": "hi"})
    (tmp_path / "archive.rar").write_bytes(b"Rar!\x1a\x07\x00")

    result = organize_archives_in_dir(tmp_path, unpack=False)

    assert len(result["archives"]) == 1
    assert result["archives"][0]["unpacked"] is False
    assert "archive.rar" in result["skipped_rar"]
    # No extraction directories should have been created.
    assert not (tmp_path / "docs_extracted").exists()


def test_organize_archives_in_dir_unpack(tmp_path):
    _make_normal_zip(tmp_path / "docs.zip", {"readme.txt": "hi"})

    result = organize_archives_in_dir(tmp_path, unpack=True)

    assert result["archives"][0]["unpacked"] is True
    extracted_dir = tmp_path / "docs_extracted"
    assert extracted_dir.exists()
    assert (extracted_dir / "readme.txt").read_text() == "hi"


def test_safe_extract_blocks_symlink_member_in_tar(tmp_path):
    archive_path = tmp_path / "symlink.tar"
    with tarfile.open(archive_path, "w") as tf:
        info = tarfile.TarInfo(name="link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)

    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive_path, tmp_path / "out")
