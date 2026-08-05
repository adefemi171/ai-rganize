"""Optional file-organization features: duplicates, archives, cloud sync, triage."""

from .archives import (
    UnsafeArchiveError,
    is_rar_archive,
    is_supported_archive,
    list_archive_contents,
    organize_archives_in_dir,
    safe_extract,
)
from .cloud_sync import discover_cloud_roots, get_cloud_root, known_providers
from .duplicates import find_duplicate_groups, find_exact_duplicates
from .triage import (
    cluster_by_category,
    cluster_by_extension,
    explain_file,
    suggest_archives,
)

__all__ = [
    "find_duplicate_groups",
    "find_exact_duplicates",
    "UnsafeArchiveError",
    "is_rar_archive",
    "is_supported_archive",
    "list_archive_contents",
    "organize_archives_in_dir",
    "safe_extract",
    "discover_cloud_roots",
    "get_cloud_root",
    "known_providers",
    "cluster_by_category",
    "cluster_by_extension",
    "explain_file",
    "suggest_archives",
]
