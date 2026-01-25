"""Base organizer functionality."""

import os
import shutil
import platform
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from ..permissions import PermissionHandler
from ..file_analysis import FileAnalyzer
from ..utils.metadata import (
    move_preserving_metadata,
    create_manifest,
    save_manifest,
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
        
        if platform.system() == "Darwin":  # macOS
            return {
                'Documents': home / 'Documents',
                'Desktop': home / 'Desktop',
                'Downloads': home / 'Downloads',
                'Pictures': home / 'Pictures' if (home / 'Pictures').exists() else None,
                'Library': home / 'Library'
            }
        elif platform.system() == "Windows":
            return {
                'Documents': home / 'Documents',
                'Desktop': home / 'Desktop',
                'Downloads': home / 'Downloads',
                'Pictures': home / 'Pictures',
                'Videos': home / 'Videos',
                'Music': home / 'Music'
            }
        else:  # Linux and others
            return {
                'Documents': home / 'Documents',
                'Desktop': home / 'Desktop',
                'Downloads': home / 'Downloads',
                'Pictures': home / 'Pictures',
                'Videos': home / 'Videos',
                'Music': home / 'Music'
            }
    
    def check_permissions(self) -> bool:
        return PermissionHandler().check_permissions(self.target_dirs)
    
    def scan_files(self, directory: Path) -> List[Dict]:
        files = []
        
        if not directory.exists():
            return files
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and not self.file_analyzer.is_system_file(file_path):
                    # Include all files regardless of size
                    files.append({
                        'path': file_path,
                        'name': file_path.name,
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                    })
        except PermissionError:
            pass  # Skip directories we can't access
        
        return files
    
    def create_backup(self, files: List[Dict]) -> bool:
        try:
            backup_dir = Path.home() / 'ai_rganize_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            for file_info in files:
                source = file_info['path']
                relative_path = source.relative_to(source.parents[1])  # Get relative path
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
        
        Args:
            plan: Organization plan mapping folder names to files
            target_dir: Target directory for organization
            save_manifest_file: Whether to save manifest for undo capability
            ai_provider: Name of AI provider used (for manifest)
            model: Model name used (for manifest)
            
        Returns:
            True if any files were moved successfully
        """
        try:
            files_moved = 0
            folders_created = 0
            
            # Create manifest for tracking moves
            manifest = create_manifest(target_dir, ai_provider, model)
            
            for folder_name, files in plan.items():
                if folder_name == 'summary':
                    continue
                
                if not files:  # Skip empty folders
                    continue
                
                # Create destination folder
                dest_folder = target_dir / folder_name
                dest_folder.mkdir(exist_ok=True)
                folders_created += 1
                
                # Move files
                for file_info in files:
                    source = file_info['path']
                    dest = dest_folder / source.name
                    
                    # Check if source exists
                    if not source.exists():
                        print(f"⚠️  File not found: {source}")
                        continue
                    
                    # Don't move if already in correct location
                    if source.resolve() == dest.resolve():
                        continue
                    
                    # Move the file with metadata preservation
                    try:
                        metadata = move_preserving_metadata(source, dest)
                        
                        # Add to manifest
                        manifest.add_move(source, dest, folder_name, metadata)
                        
                        files_moved += 1
                        print(f"✅ Moved: {source.name} → {folder_name}/")
                    except Exception as e:
                        print(f"❌ Failed to move {source.name}: {e}")
            
            # Save manifest if any files were moved
            if files_moved > 0 and save_manifest_file:
                manifest_path = save_manifest(manifest, target_dir)
                print(f"📋 Manifest saved: {manifest_path}")
            
            if files_moved == 0:
                print("⚠️  No files were moved. The organization plan may be empty or files are already in the correct location.")
            
            return files_moved > 0
        
        except Exception as e:
            print(f"❌ Organization failed: {e}")
            traceback.print_exc()
            return False
    
    def _clean_folder_name(self, name: str) -> str:
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        
        # Limit length
        if len(name) > 100:
            name = name[:100]
        
        # Remove leading/trailing spaces and dots
        name = name.strip(' .')
        
        return name or 'Unnamed_Folder'
    
    def display_organization_plan(self, plan: Dict, show_details: bool = True):
        console = Console()
        
        if not show_details:
            # Show only summary
            summary = plan.get('summary', {})
            method = summary.get('method', 'unknown')
            ai_files = summary.get('ai_files_processed', 0)
            cost = summary.get('cost_estimate', 0)
            
            if method == 'ai-powered' and ai_files > 0:
                console.print(f"\n📊 [bold]Summary:[/bold] {summary.get('total_files', 0)} files will be organized into {summary.get('total_folders', 0)} folders")
                console.print(f"🤖 [bold]AI Processing:[/bold] {ai_files} files processed with AI (${cost:.4f} estimated cost)")
            else:
                console.print(f"\n📊 [bold]Summary:[/bold] {summary.get('total_files', 0)} files will be organized into {summary.get('total_folders', 0)} folders")
            return
        
        # Show detailed plan
        for folder_name, files in plan.items():
            if folder_name == 'summary':
                continue
            
            if not files:
                continue
            
            # Calculate total size
            total_size = sum(f['size'] for f in files)
            size_mb = total_size / (1024 * 1024)
            
            # Display folder info
            console.print(f"\n📁 [bold]{folder_name}[/bold] ({len(files)} files, {size_mb:.1f} MB)")
            
            for file_info in files:
                source_path = file_info['path']
                file_size = file_info['size'] / 1024  # KB
                destination_path = source_path.parent / folder_name / source_path.name
                
                console.print(f"  📄 {file_info['name']} ({file_size:.1f} KB)")
                console.print(f"     From: {source_path.parent}")
                console.print(f"     To:   {destination_path}")
                console.print()  # Add spacing
