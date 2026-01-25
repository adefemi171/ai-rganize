"""
Rule-based file organizer using MIME types and file extensions.
"""

from typing import Dict, List
from pathlib import Path
import mimetypes

from .base_organizer import BaseOrganizer


class RuleBasedOrganizer(BaseOrganizer):
    """Rule-based file organization using MIME types and extensions."""
    
    def create_organization_plan(self, files: List[Dict], verbose: bool = False) -> Dict:
        """Create organization plan using rule-based categorization."""
        plan = {}
        
        for file_info in files:
            file_path = file_info['path']
            category = self._categorize_file(file_path)
            
            if category not in plan:
                plan[category] = []
            
            plan[category].append(file_info)
        
        # Add summary
        plan['summary'] = {
            'total_files': len(files),
            'total_folders': len([k for k in plan.keys() if k != 'summary']),
            'method': 'rule-based'
        }
        
        return plan
    
    def _categorize_file(self, file_path: Path) -> str:
        # Get file extension
        extension = file_path.suffix.lower().lstrip('.')
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Categorize based on extension and MIME type
        for category, extensions in self.categories.items():
            if extension in extensions:
                return category.title()
        
        # Fallback to MIME type
        if mime_type:
            if mime_type.startswith('image/'):
                return 'Images'
            elif mime_type.startswith('video/'):
                return 'Videos'
            elif mime_type.startswith('audio/'):
                return 'Audio'
            elif mime_type.startswith('text/'):
                return 'Documents'
            elif mime_type in ['application/pdf']:
                return 'Documents'
            elif mime_type in ['application/zip', 'application/x-rar-compressed']:
                return 'Archives'
        
        # Default category
        return 'Other'

