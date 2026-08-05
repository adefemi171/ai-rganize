"""Base organizer functionality."""

import platform
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from ..file_analysis import FileAnalyzer
from ..permissions import PermissionHandler
from ..utils.metadata import (
    create_manifest,
    move_preserving_metadata,
    save_manifest,
)
from ..utils.safety import (
    ensure_destination_safe,
    is_protected_path,
    is_symlink_or_through_symlink,
    normalize_user_path,
    sanitize_folder_name,
    unique_destination,
)


class BaseOrganizer:
    def __init__(self, max_file_size_mb: int = 10):
        self.home_dir = Path.home()
        self.target_dirs = self._get_common_directories()
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.file_analyzer = FileAnalyzer(max_file_size_mb)

        # Organization categories for rule-based categorization
        self.categories = {
            'documents': ['pdf', 'doc', 'docx', 'txt', 'rtf', 'pages'],
            'images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'heic', 'webp'],
            'videos': ['mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv', 'webm'],
            'audio': ['mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg'],
            'archives': ['zip', 'rar', '7z', 'tar', 'gz'],
            'code': ['py', 'js', 'html', 'css', 'java', 'cpp', 'c', 'php', 'rb', 'go'],
            'spreadsheets': ['xls', 'xlsx', 'csv', 'numbers'],
            'presentations': ['ppt', 'pptx', 'key'],
            'other': []
        }

    def _get_common_directories(self) -> Dict[str, Path]:
        home = Path.home()
        # Intentionally omit ~/Library and other sensitive roots
        common = {
            'Documents': home / 'Documents',
            'Desktop': home / 'Desktop',
            'Downloads': home / 'Downloads',
            'Pictures': home / 'Pictures',
        }
        if platform.system() != "Darwin":
            common['Videos'] = home / 'Videos'
            common['Music'] = home / 'Music'
        return {name: path for name, path in common.items() if path is not None}

    def check_permissions(self) -> bool:
        return PermissionHandler().check_permissions(self.target_dirs)

    def scan_files(self, directory: Path, allow_protected: bool = False) -> List[Dict]:
        files = []

        # Sanitize caller-supplied paths before any filesystem ops (CodeQL path-injection).
        try:
            directory = normalize_user_path(directory)
        except ValueError:
            return files

        if not directory.is_dir():
            return files
        if not allow_protected and is_protected_path(directory):
            print(f"⚠️  Skipping protected directory: {directory}")
            return files

        try:
            # Iterate after confinement; still skip symlinks / protected leaves.
            for file_path in directory.rglob('*'):
                try:
                    # Skip symlinks (and anything reached through a symlinked parent)
                    if is_symlink_or_through_symlink(file_path):
                        continue
                    if not file_path.is_file():
                        continue
                    if self.file_analyzer.is_system_file(file_path):
                        continue
                    if not allow_protected and is_protected_path(file_path):
                        continue
                    files.append({
                        'path': file_path,
                        'name': file_path.name,
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                    })
                except (PermissionError, OSError):
                    continue
        except PermissionError:
            pass

        return files

    def create_backup(self, files: List[Dict]) -> bool:
        try:
            backup_dir = Path.home() / 'ai_rganize_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)

            for file_info in files:
                source = file_info['path']
                try:
                    relative_path = source.relative_to(Path.home())
                except ValueError:
                    relative_path = Path(source.name)
                backup_path = backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, backup_path)

            print(f"✅ Backup created at: {backup_dir}")
            return True

        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

    def execute_organization(self, plan: Dict, target_dir: Path,
                             save_manifest_file: bool = True,
                             ai_provider: Optional[str] = None,
                             model: Optional[str] = None) -> bool:
        """
        Execute the organization plan with metadata preservation.
        
        Destinations are confined under *target_dir*. Existing files are never
        overwritten; colliding names get a numeric suffix.
        """
        try:
            files_moved = 0
            target_root = target_dir.expanduser().resolve()
            manifest = create_manifest(target_root, ai_provider, model)

            for folder_name, files in plan.items():
                if folder_name == 'summary' or not files:
                    continue

                clean_name = self._clean_folder_name(folder_name)
                dest_folder = ensure_destination_safe(target_root / clean_name, target_root)
                dest_folder.mkdir(exist_ok=True)

                for file_info in files:
                    source = Path(file_info['path'])
                    if not source.exists():
                        print(f"⚠️  File not found: {source}")
                        continue
                    if is_symlink_or_through_symlink(source):
                        print(f"⚠️  Skipping symlink: {source}")
                        continue

                    dest = ensure_destination_safe(dest_folder / source.name, target_root)
                    dest = unique_destination(dest)

                    try:
                        if source.resolve() == dest.resolve():
                            continue
                    except OSError:
                        pass

                    try:
                        metadata = move_preserving_metadata(source, dest)
                        manifest.add_move(source, dest, clean_name, metadata)
                        files_moved += 1
                        print(f"✅ Moved: {source.name} → {clean_name}/{dest.name}")
                    except Exception as e:
                        print(f"❌ Failed to move {source.name}: {e}")

            if files_moved > 0 and save_manifest_file:
                manifest_path = save_manifest(manifest, target_root)
                print(f"📋 Manifest saved: {manifest_path}")

            if files_moved == 0:
                print("⚠️  No files were moved. The organization plan may be empty or files are already in the correct location.")

            return files_moved > 0

        except Exception as e:
            print(f"❌ Organization failed: {e}")
            traceback.print_exc()
            return False

    def _clean_folder_name(self, name: str) -> str:
        return sanitize_folder_name(name)

    def display_organization_plan(self, plan: Dict, show_details: bool = True,
                                  target_dir: Optional[Path] = None):
        console = Console()

        if not show_details:
            summary = plan.get('summary', {})
            method = summary.get('method', 'unknown')
            ai_files = summary.get('ai_files_processed', 0)
            cost = summary.get('cost_estimate', 0)

            console.print(
                f"\n📊 [bold]Summary:[/bold] {summary.get('total_files', 0)} files will be "
                f"organized into {summary.get('total_folders', 0)} folders"
            )
            if method == 'ai-powered' and ai_files > 0:
                console.print(
                    f"🤖 [bold]AI Processing:[/bold] {ai_files} files processed "
                    f"with AI (${cost:.4f} estimated cost)"
                )
            return

        for folder_name, files in plan.items():
            if folder_name == 'summary' or not files:
                continue

            total_size = sum(f['size'] for f in files)
            size_mb = total_size / (1024 * 1024)
            console.print(f"\n📁 [bold]{folder_name}[/bold] ({len(files)} files, {size_mb:.1f} MB)")

            for file_info in files:
                source_path = Path(file_info['path'])
                file_size = file_info['size'] / 1024
                if target_dir is not None:
                    destination_path = Path(target_dir) / folder_name / source_path.name
                else:
                    # Per-file root: organize into the file's immediate scanned root parent
                    destination_path = source_path.parent / folder_name / source_path.name

                console.print(f"  📄 {file_info['name']} ({file_size:.1f} KB)")
                console.print(f"     From: {source_path.parent}")
                console.print(f"     To:   {destination_path}")
                console.print()
