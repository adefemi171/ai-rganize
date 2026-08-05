"""Utility functions."""

from .file_utils import extract_person_name, get_file_size_kb, get_file_size_mb, is_system_file
from .metadata import (
    FileMetadata,
    MoveRecord,
    OrganizationManifest,
    capture_metadata,
    cleanup_empty_folders,
    create_manifest,
    find_manifest,
    load_manifest,
    move_preserving_metadata,
    restore_from_manifest,
    restore_metadata,
    save_manifest,
)
from .safety import (
    ensure_destination_safe,
    is_protected_path,
    is_symlink_or_through_symlink,
    is_within_directory,
    sanitize_folder_name,
    unique_destination,
    validate_restore_path,
)
from .utils import Console, Panel

__all__ = [
    'Console',
    'Panel',
    'extract_person_name',
    'is_system_file',
    'get_file_size_mb',
    'get_file_size_kb',
    # Safety exports
    'ensure_destination_safe',
    'is_protected_path',
    'is_symlink_or_through_symlink',
    'is_within_directory',
    'sanitize_folder_name',
    'unique_destination',
    'validate_restore_path',
    # Metadata exports
    'FileMetadata',
    'capture_metadata',
    'restore_metadata',
    'move_preserving_metadata',
    'OrganizationManifest',
    'MoveRecord',
    'create_manifest',
    'save_manifest',
    'load_manifest',
    'find_manifest',
    'restore_from_manifest',
    'cleanup_empty_folders'
]
