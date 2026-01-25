"""
Metadata preservation for file organization.
Handles capturing and restoring timestamps, extended attributes, and manifest for undo.
"""

import os
import sys
import json
import base64
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

try:
    import xattr
    HAS_XATTR = True
except ImportError:
    HAS_XATTR = False

_XATTR_WARNING_SHOWN = False


@dataclass
class FileMetadata:
    """Stores complete metadata for a file."""
    original_path: str
    filename: str
    size: int
    created_time: float      # birthtime / st_birthtime
    modified_time: float     # mtime / st_mtime
    accessed_time: float     # atime / st_atime
    permissions: int         # file mode
    extended_attrs: Dict[str, str] = field(default_factory=dict)  # xattr (base64 encoded)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_path': self.original_path,
            'filename': self.filename,
            'size': self.size,
            'created_time': self.created_time,
            'modified_time': self.modified_time,
            'accessed_time': self.accessed_time,
            'permissions': self.permissions,
            'extended_attrs': self.extended_attrs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        """Create FileMetadata from dictionary."""
        return cls(
            original_path=data['original_path'],
            filename=data['filename'],
            size=data['size'],
            created_time=data['created_time'],
            modified_time=data['modified_time'],
            accessed_time=data['accessed_time'],
            permissions=data['permissions'],
            extended_attrs=data.get('extended_attrs', {})
        )


def _get_creation_time(path: Path) -> float:
    """Get file creation time (birthtime) cross-platform."""
    stat_info = path.stat()
    
    if sys.platform == 'darwin':
        # macOS: st_birthtime is available
        return stat_info.st_birthtime
    elif sys.platform == 'win32':
        # Windows: st_ctime is creation time
        return stat_info.st_ctime
    else:
        # Linux: st_ctime is metadata change time, not creation
        # Try to get birthtime if available (some filesystems support it)
        try:
            return stat_info.st_birthtime
        except AttributeError:
            # Fall back to mtime as best approximation
            return stat_info.st_mtime


def _get_extended_attrs(path: Path) -> Dict[str, str]:
    """Get extended attributes from file."""
    global _XATTR_WARNING_SHOWN
    
    if not HAS_XATTR:
        if not _XATTR_WARNING_SHOWN:
            _XATTR_WARNING_SHOWN = True
        return {}
    
    attrs = {}
    try:
        xattr_obj = xattr.xattr(str(path))
        for name in xattr_obj.list():
            try:
                value = xattr_obj.get(name)
                attrs[name] = base64.b64encode(value).decode('ascii')
            except Exception:
                pass
    except Exception:
        pass  # Skip if xattr fails entirely
    
    return attrs


def _set_extended_attrs(path: Path, attrs: Dict[str, str]) -> None:
    """Restore extended attributes to file."""
    if not HAS_XATTR or not attrs:
        return
    
    try:
        xattr_obj = xattr.xattr(str(path))
        for name, value_b64 in attrs.items():
            try:
                value = base64.b64decode(value_b64.encode('ascii'))
                xattr_obj.set(name, value)
            except Exception:
                pass
    except Exception:
        pass  # Skip if xattr fails entirely


def _restore_creation_time_macos(path: Path, creation_time: float) -> bool:
    """Restore creation time on macOS using SetFile or touch."""
    if sys.platform != 'darwin':
        return False
    
    try:
        # Convert timestamp to the format SetFile expects: "MM/DD/YYYY HH:MM:SS"
        dt = datetime.fromtimestamp(creation_time)
        date_str = dt.strftime("%m/%d/%Y %H:%M:%S")
        
        # Try SetFile first (from Xcode Command Line Tools)
        try:
            subprocess.run(
                ['SetFile', '-d', date_str, str(path)],
                check=True,
                capture_output=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fallback: use touch with -t flag for modification time at least
        # Note: touch cannot set creation time, only access/modification
        return False
        
    except Exception:
        return False


def capture_metadata(path: Path) -> FileMetadata:
    """
    Capture complete metadata from a file.
    
    Args:
        path: Path to the file
        
    Returns:
        FileMetadata object containing all captured metadata
    """
    stat_info = path.stat()
    
    return FileMetadata(
        original_path=str(path.resolve()),
        filename=path.name,
        size=stat_info.st_size,
        created_time=_get_creation_time(path),
        modified_time=stat_info.st_mtime,
        accessed_time=stat_info.st_atime,
        permissions=stat_info.st_mode,
        extended_attrs=_get_extended_attrs(path)
    )


def restore_metadata(path: Path, metadata: FileMetadata) -> None:
    """
    Restore metadata to a file after move.
    
    Args:
        path: Path to the file (new location)
        metadata: FileMetadata to restore
    """
    # Restore access and modification times
    os.utime(path, (metadata.accessed_time, metadata.modified_time))
    
    # Restore permissions
    try:
        os.chmod(path, metadata.permissions)
    except Exception:
        pass  # Permission changes may fail on some systems
    
    # Restore extended attributes
    _set_extended_attrs(path, metadata.extended_attrs)
    
    # Restore creation time on macOS
    if sys.platform == 'darwin':
        _restore_creation_time_macos(path, metadata.created_time)


def move_preserving_metadata(source: Path, dest: Path) -> FileMetadata:
    """
    Move file while preserving all metadata including creation time.
    
    Args:
        source: Source file path
        dest: Destination file path
        
    Returns:
        FileMetadata captured before the move (for manifest)
    """
    # 1. Capture metadata before move
    metadata = capture_metadata(source)
    
    # 2. Perform the move
    shutil.move(str(source), str(dest))
    
    # 3. Restore all metadata
    restore_metadata(dest, metadata)
    
    return metadata


# ============================================================================
# Manifest Management
# ============================================================================

MANIFEST_FILENAME = '.ai_rganize_manifest.json'
MANIFEST_VERSION = '1.0'


@dataclass
class MoveRecord:
    """Record of a single file move operation."""
    original: str
    destination: str
    category: str
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original': self.original,
            'destination': self.destination,
            'category': self.category,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MoveRecord':
        return cls(
            original=data['original'],
            destination=data['destination'],
            category=data['category'],
            timestamp=data['timestamp'],
            metadata=data['metadata']
        )


@dataclass
class OrganizationManifest:
    """Complete manifest of an organization operation."""
    version: str
    created: str
    source_directory: str
    ai_provider: Optional[str]
    model: Optional[str]
    moves: List[MoveRecord] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'created': self.created,
            'source_directory': self.source_directory,
            'ai_provider': self.ai_provider,
            'model': self.model,
            'moves': [m.to_dict() for m in self.moves]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrganizationManifest':
        return cls(
            version=data['version'],
            created=data['created'],
            source_directory=data['source_directory'],
            ai_provider=data.get('ai_provider'),
            model=data.get('model'),
            moves=[MoveRecord.from_dict(m) for m in data.get('moves', [])]
        )
    
    def add_move(self, original: Path, destination: Path, category: str, 
                 metadata: FileMetadata) -> None:
        """Add a move record to the manifest."""
        self.moves.append(MoveRecord(
            original=str(original),
            destination=str(destination),
            category=category,
            timestamp=datetime.now().isoformat(),
            metadata=metadata.to_dict()
        ))


def create_manifest(source_directory: Path, ai_provider: Optional[str] = None,
                   model: Optional[str] = None) -> OrganizationManifest:
    """
    Create a new organization manifest.
    
    Args:
        source_directory: The directory being organized
        ai_provider: Name of the AI provider used (optional)
        model: Model name used (optional)
        
    Returns:
        New OrganizationManifest instance
    """
    return OrganizationManifest(
        version=MANIFEST_VERSION,
        created=datetime.now().isoformat(),
        source_directory=str(source_directory),
        ai_provider=ai_provider,
        model=model,
        moves=[]
    )


def save_manifest(manifest: OrganizationManifest, target_dir: Path) -> Path:
    """
    Save organization manifest to the target directory.
    
    Args:
        manifest: The manifest to save
        target_dir: Directory to save the manifest in
        
    Returns:
        Path to the saved manifest file
    """
    manifest_path = target_dir / MANIFEST_FILENAME
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
    
    return manifest_path


def load_manifest(manifest_path: Path) -> OrganizationManifest:
    """
    Load an organization manifest from file.
    
    Args:
        manifest_path: Path to the manifest file
        
    Returns:
        OrganizationManifest instance
        
    Raises:
        FileNotFoundError: If manifest doesn't exist
        json.JSONDecodeError: If manifest is invalid JSON
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return OrganizationManifest.from_dict(data)


def find_manifest(directory: Path) -> Optional[Path]:
    """
    Find a manifest file in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        Path to manifest if found, None otherwise
    """
    manifest_path = directory / MANIFEST_FILENAME
    if manifest_path.exists():
        return manifest_path
    return None


def restore_from_manifest(manifest: OrganizationManifest, 
                         verbose: bool = False) -> tuple[int, int]:
    """
    Undo organization by restoring files to original locations.
    
    Args:
        manifest: The manifest to restore from
        verbose: Whether to print progress
        
    Returns:
        Tuple of (successful_restores, failed_restores)
    """
    successful = 0
    failed = 0
    
    # Process in reverse order to handle any dependencies
    for move in reversed(manifest.moves):
        source = Path(move.destination)
        dest = Path(move.original)
        
        try:
            if not source.exists():
                if verbose:
                    print(f"⚠️  File not found: {source}")
                failed += 1
                continue
            
            # Create parent directory if needed
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file back
            shutil.move(str(source), str(dest))
            
            # Restore original metadata
            if move.metadata:
                metadata = FileMetadata.from_dict(move.metadata)
                restore_metadata(dest, metadata)
            
            if verbose:
                print(f"✅ Restored: {source.name} → {dest.parent}/")
            successful += 1
            
        except Exception as e:
            if verbose:
                print(f"❌ Failed to restore {source.name}: {e}")
            failed += 1
    
    return successful, failed


def cleanup_empty_folders(directory: Path, verbose: bool = False) -> int:
    """
    Remove empty folders after restoration.
    
    Args:
        directory: Root directory to clean
        verbose: Whether to print progress
        
    Returns:
        Number of folders removed
    """
    removed = 0
    
    # Walk bottom-up to handle nested empty directories
    for dirpath in sorted(directory.rglob('*'), reverse=True):
        if dirpath.is_dir():
            try:
                # Check if directory is empty (ignoring hidden files like .DS_Store)
                contents = [f for f in dirpath.iterdir() if not f.name.startswith('.')]
                if not contents:
                    # Remove any hidden files first
                    for hidden in dirpath.iterdir():
                        try:
                            hidden.unlink()
                        except Exception:
                            pass
                    dirpath.rmdir()
                    removed += 1
                    if verbose:
                        print(f"🗑️  Removed empty folder: {dirpath.name}")
            except Exception:
                pass  # Skip if we can't remove
    
    return removed
