"""
File analysis utilities for content extraction and metadata.
"""

import mimetypes
from pathlib import Path
from typing import Optional

from .analyzers import VideoAnalyzer, AudioAnalyzer, DocumentAnalyzer, ImageAnalyzer, TextAnalyzer
from .utils import extract_person_name, is_system_file


class FileAnalyzer:
    """Analyzes files to extract content and metadata for AI categorization."""
    
    def __init__(self, max_file_size_mb: int = 10):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Initialize specialized analyzers
        self.video_analyzer = VideoAnalyzer(self.max_file_size_bytes)
        self.audio_analyzer = AudioAnalyzer(self.max_file_size_bytes)
        self.document_analyzer = DocumentAnalyzer(self.max_file_size_bytes)
        self.image_analyzer = ImageAnalyzer(self.max_file_size_bytes)
        self.text_analyzer = TextAnalyzer()
    
    def get_file_content_preview(self, file_path: Path) -> str:
        try:
            if is_system_file(file_path):
                return "System file (skipped)"
            
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Route to appropriate analyzer based on file type
            if mime_type:
                if mime_type.startswith('video/'):
                    return self.video_analyzer.analyze_video(file_path)
                elif mime_type.startswith('audio/'):
                    return self.audio_analyzer.analyze_audio(file_path)
                elif mime_type == 'application/pdf':
                    return self.document_analyzer.analyze_pdf(file_path)
                elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                                  'application/msword']:
                    return self.document_analyzer.analyze_word_document(file_path)
                elif mime_type.startswith('image/'):
                    return self.image_analyzer.analyze_image(file_path)
                elif mime_type.startswith('text/'):
                    return self.text_analyzer.analyze_text(file_path)
            
            # Fallback based on file extension
            extension = file_path.suffix.lower()
            if extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
                return self.video_analyzer.analyze_video(file_path)
            elif extension in ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']:
                return self.audio_analyzer.analyze_audio(file_path)
            elif extension == '.pdf':
                return self.document_analyzer.analyze_pdf(file_path)
            elif extension in ['.docx', '.doc']:
                return self.document_analyzer.analyze_word_document(file_path)
            elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                return self.image_analyzer.analyze_image(file_path)
            elif extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml']:
                return self.text_analyzer.analyze_text(file_path)
            
            # Default fallback
            return f"File: {file_path.name} ({file_path.stat().st_size / 1024:.1f}KB)"
            
        except Exception as e:
            return f"Analysis error: {str(e)[:50]}"
    
    def get_image_base64(self, file_path: Path) -> Optional[str]:
        return self.image_analyzer._get_image_base64(file_path)
    
    def extract_person_name(self, filename: str) -> Optional[str]:
        return extract_person_name(filename)
    
    def is_system_file(self, file_path: Path) -> bool:
        return is_system_file(file_path)