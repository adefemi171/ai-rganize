"""Safe archive inspection and extraction.

This module is deliberately conservative: it only uses the Python standard
library (``zipfile`` / ``tarfile``), refuses to extract anything that would
escape the destination directory ("zip-slip"), and enforces hard limits on
file count and total uncompressed size to avoid zip-bomb style denial of
service. RAR archives are not supported (no safe stdlib implementation
exists) and are skipped with a warning.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import Optional

ZIP_EXTENSIONS = {".zip"}
TAR_EXTENSIONS = {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}
RAR_EXTENSIONS = {".rar"}

DEFAULT_MAX_FILES = 1000
DEFAULT_MAX_TOTAL_BYTES = 500_000_000


class UnsafeArchiveError(Exception):
    """Raised when an archive appears to contain a path-traversal attempt or
    otherwise exceeds configured safety limits."""


def _archive_suffix(path: Path) -> Optional[str]:
    name = path.name.lower()
    for suffix in sorted(TAR_EXTENSIONS, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    if path.suffix.lower() in ZIP_EXTENSIONS:
        return path.suffix.lower()
    if path.suffix.lower() in RAR_EXTENSIONS:
        return path.suffix.lower()
    return None


def is_supported_archive(path: Path) -> bool:
    suffix = _archive_suffix(Path(path))
    return suffix is not None and suffix not in RAR_EXTENSIONS


def is_rar_archive(path: Path) -> bool:
    return _archive_suffix(Path(path)) in RAR_EXTENSIONS


def _member_is_safe(member_name: str, dest_dir: Path) -> Optional[Path]:
    """Resolve a member path against *dest_dir*, returning None if unsafe."""
    if not member_name or member_name.strip() == "":
        return None

    # Reject absolute paths and drive-letter paths outright.
    if Path(member_name).is_absolute() or (len(member_name) > 1 and member_name[1] == ":"):
        return None

    # Reject any ".." path component before we even touch the filesystem.
    normalized = member_name.replace("\\", "/")
    if ".." in normalized.split("/"):
        return None

    candidate = (dest_dir / member_name).resolve()
    dest_resolved = dest_dir.resolve()

    try:
        is_inside = candidate.is_relative_to(dest_resolved)
    except AttributeError:  # pragma: no cover - Python < 3.9 fallback
        try:
            candidate.relative_to(dest_resolved)
            is_inside = True
        except ValueError:
            is_inside = False

    if not is_inside:
        return None
    return candidate


def list_archive_contents(archive_path: Path) -> list[str]:
    """List member names inside a zip/tar archive without extracting anything."""
    archive_path = Path(archive_path)
    suffix = _archive_suffix(archive_path)

    if suffix in RAR_EXTENSIONS:
        raise UnsafeArchiveError(
            f"RAR archives are not supported for safety/compatibility reasons: {archive_path.name}"
        )

    if suffix in ZIP_EXTENSIONS:
        with zipfile.ZipFile(archive_path) as zf:
            return zf.namelist()

    if suffix in TAR_EXTENSIONS:
        with tarfile.open(archive_path) as tf:
            return tf.getnames()

    raise ValueError(f"Unsupported or unrecognized archive format: {archive_path.name}")


def safe_extract(
    archive_path: Path,
    dest_dir: Path,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> list[Path]:
    """Safely extract *archive_path* into *dest_dir*.

    Guarantees:
      * Every extracted member resolves to a path strictly inside ``dest_dir``
        (prevents "zip-slip" path traversal).
      * Absolute paths and ``..`` components in member names are rejected.
      * At most ``max_files`` members are extracted.
      * The sum of uncompressed member sizes does not exceed ``max_total_bytes``.

    Raises ``UnsafeArchiveError`` if any check fails; nothing partially unsafe
    is left behind for the offending member (we validate all members before
    extracting any of them).
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = _archive_suffix(archive_path)

    if suffix in RAR_EXTENSIONS:
        raise UnsafeArchiveError(
            f"RAR archives are not supported for safety/compatibility reasons: {archive_path.name}"
        )

    if suffix in ZIP_EXTENSIONS:
        return _safe_extract_zip(archive_path, dest_dir, max_files, max_total_bytes)

    if suffix in TAR_EXTENSIONS:
        return _safe_extract_tar(archive_path, dest_dir, max_files, max_total_bytes)

    raise ValueError(f"Unsupported or unrecognized archive format: {archive_path.name}")


def _safe_extract_zip(
    archive_path: Path, dest_dir: Path, max_files: int, max_total_bytes: int
) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]

        if len(infos) > max_files:
            raise UnsafeArchiveError(
                f"Archive contains {len(infos)} files, exceeding limit of "
                f"{max_files}: {archive_path.name}"
            )

        total_size = sum(i.file_size for i in infos)
        if total_size > max_total_bytes:
            raise UnsafeArchiveError(
                f"Archive uncompressed size {total_size} exceeds limit of "
                f"{max_total_bytes}: {archive_path.name}"
            )

        resolved_targets: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in infos:
            target = _member_is_safe(info.filename, dest_dir)
            if target is None:
                raise UnsafeArchiveError(
                    f"Unsafe path in archive (possible zip-slip): "
                    f"{info.filename!r} in {archive_path.name}"
                )
            resolved_targets.append((info, target))

        for info, target in resolved_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)

    return extracted


def _safe_extract_tar(
    archive_path: Path, dest_dir: Path, max_files: int, max_total_bytes: int
) -> list[Path]:
    extracted: list[Path] = []
    with tarfile.open(archive_path) as tf:
        all_members = tf.getmembers()

        # Reject symlinks/hardlinks/devices/fifos anywhere in the archive --
        # their targets could point outside dest_dir regardless of name,
        # and we never want to materialize device/fifo nodes on disk.
        for member in all_members:
            if member.issym() or member.islnk() or member.isdev():
                raise UnsafeArchiveError(
                    f"Archive contains a symlink/hardlink/device member, "
                    f"which is not allowed: {member.name!r}"
                )

        members = [m for m in all_members if m.isfile()]

        if len(members) > max_files:
            raise UnsafeArchiveError(
                f"Archive contains {len(members)} files, exceeding limit of "
                f"{max_files}: {archive_path.name}"
            )

        total_size = sum(m.size for m in members)
        if total_size > max_total_bytes:
            raise UnsafeArchiveError(
                f"Archive uncompressed size {total_size} exceeds limit of "
                f"{max_total_bytes}: {archive_path.name}"
            )

        resolved_targets: list[tuple[tarfile.TarInfo, Path]] = []
        for member in members:
            target = _member_is_safe(member.name, dest_dir)
            if target is None:
                raise UnsafeArchiveError(
                    f"Unsafe path in archive (possible path traversal): "
                    f"{member.name!r} in {archive_path.name}"
                )
            resolved_targets.append((member, target))

        for member, target in resolved_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)

    return extracted


def organize_archives_in_dir(directory: Path, unpack: bool = False) -> dict[str, object]:
    """Inventory archives in *directory* (non-recursive) and optionally unpack them.

    When ``unpack`` is False (the default), this only returns an inventory of
    discovered archives and their contents -- nothing is written to disk. When
    ``unpack`` is True, each supported archive is safely extracted into a
    sibling directory named ``<archive_stem>_extracted/``.
    """
    directory = Path(directory)
    inventory: list[dict[str, object]] = []
    skipped_rar: list[str] = []

    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        if is_rar_archive(entry):
            skipped_rar.append(entry.name)
            continue
        if not is_supported_archive(entry):
            continue

        record: dict[str, object] = {"archive": entry, "unpacked": False}
        try:
            record["contents"] = list_archive_contents(entry)
        except Exception as exc:  # noqa: BLE001
            record["error"] = str(exc)
            inventory.append(record)
            continue

        if unpack:
            suffix = _archive_suffix(entry) or entry.suffix
            if suffix and entry.name.lower().endswith(suffix):
                stem = entry.name[: -len(suffix)]
            else:
                stem = entry.stem
            extract_dir = directory / f"{stem}_extracted"
            try:
                record["extracted_files"] = safe_extract(entry, extract_dir)
                record["extracted_to"] = extract_dir
                record["unpacked"] = True
            except UnsafeArchiveError as exc:
                record["error"] = str(exc)

        inventory.append(record)

    return {"archives": inventory, "skipped_rar": skipped_rar}
