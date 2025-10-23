"""
File analysis utilities for content extraction and metadata.
"""

import os
import mimetypes
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import PyPDF2


class FileAnalyzer:
    """Analyzes files to extract content and metadata for AI categorization."""
    
    def __init__(self, max_file_size_mb: int = 10):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def get_file_content_preview(self, file_path: Path) -> str:
        """Get a content preview of the file for AI analysis."""
        try:
            # Check file size
            if file_path.stat().st_size > self.max_file_size_bytes:
                return f"Large file ({file_path.stat().st_size / (1024*1024):.1f}MB) - content analysis skipped"
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Analyze based on file type
            if mime_type and mime_type.startswith('image/'):
                return self._analyze_image(file_path)
            elif mime_type and mime_type.startswith('video/'):
                return self._analyze_video(file_path)
            elif mime_type and mime_type.startswith('audio/'):
                return self._analyze_audio(file_path)
            elif file_path.suffix.lower() == '.pdf':
                return self._analyze_pdf(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md', '.log']:
                return self._analyze_text(file_path)
            else:
                return f"File type: {mime_type or 'unknown'}, Size: {file_path.stat().st_size / 1024:.1f}KB"
        
        except Exception as e:
            return f"Error analyzing file: {str(e)[:100]}"
    
    def _analyze_image(self, file_path: Path) -> str:
        """Analyze image files for content hints."""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
                
                # Basic image analysis
                analysis = f"Image: {width}x{height}, {mode} mode, {format_name} format"
                
                # Add size info
                file_size = file_path.stat().st_size / 1024
                analysis += f", {file_size:.1f}KB"
                
                # Try to detect if it's a screenshot
                if width >= 1920 and height >= 1080:
                    analysis += " (likely screenshot or high-res image)"
                elif width < 500 and height < 500:
                    analysis += " (likely thumbnail or small image)"
                
                return analysis
        
        except Exception as e:
            return f"Image analysis error: {str(e)[:50]}"
    
    def _analyze_video(self, file_path: Path) -> str:
        """Analyze video files."""
        try:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            return f"Video file: {file_size:.1f}MB"
        except Exception:
            return "Video file (analysis unavailable)"
    
    def _analyze_audio(self, file_path: Path) -> str:
        """Analyze audio files."""
        try:
            file_size = file_path.stat().st_size / 1024  # KB
            return f"Audio file: {file_size:.1f}KB"
        except Exception:
            return "Audio file (analysis unavailable)"
    
    def _analyze_pdf(self, file_path: Path) -> str:
        """Extract text content from PDF files."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from first few pages
                text_content = ""
                max_pages = min(3, len(pdf_reader.pages))  # First 3 pages max
                
                for page_num in range(max_pages):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + " "
                
                # Clean and truncate text
                text_content = text_content.strip()[:500]  # First 500 chars
                
                if text_content:
                    # Look for common document patterns
                    if any(keyword in text_content.lower() for keyword in ['resume', 'cv', 'curriculum vitae']):
                        return f"PDF Resume/CV: {text_content[:100]}..."
                    elif any(keyword in text_content.lower() for keyword in ['cover letter', 'application']):
                        return f"PDF Cover Letter: {text_content[:100]}..."
                    elif any(keyword in text_content.lower() for keyword in ['invoice', 'receipt', 'bill']):
                        return f"PDF Financial Document: {text_content[:100]}..."
                    else:
                        return f"PDF Document: {text_content[:100]}..."
                else:
                    return "PDF Document (text extraction failed)"
        
        except Exception as e:
            return f"PDF analysis error: {str(e)[:50]}"
    
    def _analyze_text(self, file_path: Path) -> str:
        """Analyze text files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read(500)  # First 500 characters
                
                if content:
                    return f"Text file: {content[:100]}..."
                else:
                    return "Empty text file"
        
        except Exception as e:
            return f"Text analysis error: {str(e)[:50]}"
    
    def extract_person_name(self, filename: str) -> Optional[str]:
        """Extract potential person name from filename."""
        # Remove common extensions
        name = Path(filename).stem
        
        # Look for patterns like "FirstName_LastName" or "LastName_FirstName"
        parts = name.replace('-', '_').replace('.', '_').split('_')
        
        # Filter out common non-name parts
        filtered_parts = []
        for part in parts:
            if (len(part) > 2 and 
                part.lower() not in ['cv', 'resume', 'profile', 'application', 'cover', 'letter'] and
                not part.isdigit()):
                filtered_parts.append(part)
        
        if len(filtered_parts) >= 2:
            # Return the first two parts as potential name
            return f"{filtered_parts[0]} {filtered_parts[1]}"
        
        return None
    
    def is_system_file(self, file_path: Path) -> bool:
        """Check if file is a system file that should be ignored."""
        system_patterns = [
            '.DS_Store',
            'Thumbs.db',
            '.Spotlight-V100',
            '.Trashes',
            '.fseventsd',
            '.TemporaryItems',
            'desktop.ini'
        ]
        
        return file_path.name in system_patterns or file_path.name.startswith('.')

