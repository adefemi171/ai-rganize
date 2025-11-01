"""
Image analysis utilities for content extraction and categorization.
"""

import base64
from pathlib import Path
from typing import Optional
from PIL import Image


class ImageAnalyzer:
    """Analyzes image files for content and categorization."""
    
    def __init__(self, max_file_size_bytes: int):
        self.max_file_size_bytes = max_file_size_bytes
    
    def analyze_image(self, file_path: Path) -> str:
        """Analyze image files using OpenAI Vision API."""
        try:
            # Get image dimensions
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format or "Unknown"
            
            file_size = file_path.stat().st_size / 1024  # KB
            analysis = f"Image: {width}x{height}, {mode} mode, {format_name} format, {file_size:.1f}KB"
            
            # Try to analyze with Vision API
            try:
                vision_analysis = self._analyze_image_with_vision_api(file_path)
                if vision_analysis:
                    analysis += f" | Content: {vision_analysis}"
            except Exception as e:
                analysis += f" | Vision analysis failed: {str(e)[:30]}"
            
            return analysis
        except Exception:
            return "Image file (analysis unavailable)"
    
    def _analyze_image_with_vision_api(self, file_path: Path) -> str:
        """Analyze image content using OpenAI Vision API."""
        try:
            # Get base64 encoded image
            base64_image = self._get_image_base64(file_path)
            if not base64_image:
                return "Image too large for analysis"
            
            from openai import OpenAI
            client = OpenAI()
            
            # Analyze image with Vision API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and describe what you see. Focus on: 1) What type of content it is (screenshot, photo, document, etc.), 2) Any text visible, 3) Any people or objects, 4) The purpose or context. Keep it concise (max 100 words)."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Vision API error: {str(e)[:50]}"
    
    def _get_image_base64(self, file_path: Path) -> Optional[str]:
        """Get base64 encoded image for OpenAI Vision API."""
        try:
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception:
            return None
