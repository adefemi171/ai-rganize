"""
Core AIrganizer functionality
"""

import os
import sys
import json
import shutil
import hashlib
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import openai
from PIL import Image
import mimetypes

from .utils import Console, Progress, SpinnerColumn, TextColumn, Table, Panel
from .permissions import PermissionHandler


class AI_rganize:
    """Main AI-rganize class for intelligent file organization."""
    
    def __init__(self, api_key: Optional[str] = None, max_file_size_mb: int = 10):
        """Initialize AI-rganize."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.home_dir = Path.home()
        self.target_dirs = self._get_common_directories()
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        
        # Organization categories
        self.categories = {
            'documents': ['pdf', 'doc', 'docx', 'txt', 'rtf', 'pages'],
            'images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'heic', 'webp'],
            'videos': ['mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv', 'webm'],
            'audio': ['mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg'],
            'archives': ['zip', 'rar', '7z', 'tar', 'gz'],
            'code': ['py', 'js', 'html', 'css', 'java', 'cpp', 'c', 'php', 'rb', 'go'],
            'spreadsheets': ['xls', 'xlsx', 'csv', 'numbers'],
            'presentations': ['ppt', 'pptx', 'key']
        }
        
        self.backup_dir = self.home_dir / '.airganizer_backup'
        self.log_file = self.home_dir / '.airganizer_log.json'
        self.permission_handler = PermissionHandler()
    
    def _get_common_directories(self) -> Dict[str, Path]:
        """Get common directories based on the operating system."""
        system = platform.system().lower()
        home = self.home_dir
        
        if system == "darwin":  # macOS
            return {
                'Documents': home / 'Documents',
                'Downloads': home / 'Downloads',
                'Desktop': home / 'Desktop',
                'Library': home / 'Library'  # Note: Library folder may require special permissions
            }
        elif system == "linux":  # Linux (Ubuntu, etc.)
            return {
                'Documents': home / 'Documents',
                'Desktop': home / 'Desktop',
                'Pictures': home / 'Pictures',
                'Downloads': home / 'Downloads',
                'Music': home / 'Music',
                'Videos': home / 'Videos',
                'Public': home / 'Public'
            }
        elif system == "windows":  # Windows
            return {
                'Documents': home / 'Documents',
                'Desktop': home / 'Desktop',
                'Pictures': home / 'Pictures',
                'Downloads': home / 'Downloads',
                'Music': home / 'Music',
                'Videos': home / 'Videos',
                'Public': home / 'Public'
            }
        else:  # Fallback for other systems
            return {
                'Documents': home / 'Documents',
                'Desktop': home / 'Desktop',
                'Downloads': home / 'Downloads'
            }
    
    def get_mime_type(self, file_path: Path) -> str:
        """Get MIME type using Python's mimetypes module as fallback."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'
    
    def check_permissions(self) -> bool:
        """Check if we have permission to access the target directories."""
        return self.permission_handler.check_permissions(self.target_dirs)
    
    def scan_files(self, directory: Path) -> List[Dict]:
        """Scan a directory for files and return file information."""
        files = []
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        file_info = {
                            'path': str(file_path),
                            'name': file_path.name,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime),
                            'extension': file_path.suffix.lower().lstrip('.'),
                            'mime_type': self.get_mime_type(file_path),
                            'relative_path': str(file_path.relative_to(directory))
                        }
                        files.append(file_info)
                    except (OSError, PermissionError) as e:
                        print(f"Warning: Could not access {file_path}: {e}")
        except PermissionError as e:
            print(f"Permission denied accessing {directory}: {e}")
            
        return files
    
    def get_file_content_preview(self, file_path: str, max_size: int = 1000) -> str:
        """Get a preview of file content for AI analysis."""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > max_size:
                return f"[File too large: {file_size} bytes]"
            
            # Try to read text files
            if file_path.lower().endswith(('.txt', '.md', '.py', '.js', '.html', '.css')):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()[:max_size]
            
            # For images, get basic info
            elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                try:
                    with Image.open(file_path) as img:
                        return f"Image: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}"
                except:
                    return "[Image file]"
            
            return "[Binary file]"
        except Exception as e:
            return f"[Error reading file: {e}]"
    
    def categorize_with_ai(self, file_info: Dict) -> str:
        """Use AI to categorize a file based on its content and metadata."""
        import time
        
        try:
            content_preview = self.get_file_content_preview(file_info['path'])
            
            prompt = f"""
            Analyze this file and suggest the best category for organization:
            
            File: {file_info['name']}
            Extension: {file_info['extension']}
            MIME Type: {file_info['mime_type']}
            Size: {file_info['size']} bytes
            Modified: {file_info['modified']}
            
            Content Preview:
            {content_preview}
            
            Based on this information, suggest ONE of these categories:
            - documents (text files, PDFs, office docs)
            - images (photos, graphics, screenshots)
            - videos (movie files, video recordings)
            - audio (music, recordings, podcasts)
            - archives (compressed files)
            - code (programming files)
            - spreadsheets (data files, CSV, Excel)
            - presentations (slides, Keynote)
            - other (doesn't fit other categories)
            
            Respond with ONLY the category name, nothing else.
            """
            
            # Add retry logic for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=50,
                        temperature=0.1
                    )
                    
                    category = response.choices[0].message.content.strip().lower()
                    
                    # Validate category
                    if category in self.categories or category == 'other':
                        return category
                    else:
                        return 'other'
                        
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 20  # Exponential backoff
                            print(f"Rate limit hit, waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"Rate limit exceeded after {max_retries} attempts")
                            return 'other'
                    else:
                        raise e
                
        except Exception as e:
            print(f"AI categorization failed for {file_info['name']}: {e}")
            return 'other'
    
    def rule_based_categorization(self, file_info: Dict) -> str:
        """Use simple rules to categorize files based on extension and MIME type."""
        extension = file_info['extension']
        mime_type = file_info['mime_type']
        
        for category, extensions in self.categories.items():
            if extension in extensions:
                return category
        
        # MIME type fallback
        if mime_type.startswith('image/'):
            return 'images'
        elif mime_type.startswith('video/'):
            return 'videos'
        elif mime_type.startswith('audio/'):
            return 'audio'
        elif mime_type.startswith('text/'):
            return 'documents'
        elif mime_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return 'documents'
        
        return 'other'
    
    def create_organization_plan(self, files: List[Dict], use_ai_limit: int = 50) -> Dict[str, List[Dict]]:
        """Create an organization plan using AI categorization."""
        print("Analyzing files with AI...")
        
        organization_plan = {category: [] for category in self.categories.keys()}
        organization_plan['other'] = []
        
        # Limit AI usage to avoid rate limits
        ai_files_processed = 0
        
        for file_info in files:
            # First try rule-based categorization
            category = self.rule_based_categorization(file_info)
            
            # If rule-based fails, use AI (with limit)
            if (category == 'other' and 
                file_info['size'] < self.max_file_size_bytes and 
                ai_files_processed < use_ai_limit):
                category = self.categorize_with_ai(file_info)
                ai_files_processed += 1
            elif category == 'other' and ai_files_processed >= use_ai_limit:
                print(f"AI limit reached ({use_ai_limit} files), using rule-based for remaining files")
            
            organization_plan[category].append(file_info)
        
        return organization_plan
    
    def create_backup(self, files: List[Dict]) -> bool:
        """Create a backup of files before organization."""
        try:
            self.backup_dir.mkdir(exist_ok=True)
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{backup_timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            print(f"Creating backup at {backup_path}...")
            
            for file_info in files:
                src = Path(file_info['path'])
                dst = backup_path / file_info['relative_path']
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            
            print("Backup created successfully!")
            return True
            
        except Exception as e:
            print(f"Backup failed: {e}")
            return False
    
    def display_organization_plan(self, plan: Dict[str, List[Dict]]) -> None:
        """Display the organization plan to the user."""
        print("\nFile Organization Plan:")
        print("=" * 50)
        
        total_files = 0
        total_size = 0
        
        for category, files in plan.items():
            if files:
                size_mb = sum(f['size'] for f in files) / (1024 * 1024)
                print(f"{category.title()}: {len(files)} files ({size_mb:.1f} MB)")
                total_files += len(files)
                total_size += size_mb
        
        print(f"\nTOTAL: {total_files} files ({total_size:.1f} MB)")
        print("=" * 50)
    
    def execute_organization(self, plan: Dict[str, List[Dict]], target_dir: Path) -> bool:
        """Execute the file organization plan."""
        try:
            # Create category directories
            for category in plan.keys():
                if plan[category]:  # Only create if there are files
                    category_dir = target_dir / category
                    category_dir.mkdir(exist_ok=True)
            
            # Move files
            moved_files = []
            for category, files in plan.items():
                if not files:
                    continue
                    
                category_dir = target_dir / category
                
                for file_info in files:
                    src = Path(file_info['path'])
                    dst = category_dir / file_info['name']
                    
                    # Handle duplicate names
                    counter = 1
                    original_dst = dst
                    while dst.exists():
                        name_parts = original_dst.stem, counter, original_dst.suffix
                        dst = original_dst.parent / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                        counter += 1
                    
                    try:
                        shutil.move(str(src), str(dst))
                        moved_files.append({
                            'original': str(src),
                            'new': str(dst),
                            'category': category
                        })
                        print(f"Moved: {src.name} → {category}/{dst.name}")
                    except Exception as e:
                        print(f"Failed to move {src.name}: {e}")
            
            # Log the organization
            self._log_organization(moved_files)
            return True
            
        except Exception as e:
            print(f"Organization failed: {e}")
            return False
    
    def _log_organization(self, moved_files: List[Dict]) -> None:
        """Log the organization results."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'moved_files': moved_files
        }
        
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'organizations': []}
            
            log_data['organizations'].append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"Could not log organization: {e}")
