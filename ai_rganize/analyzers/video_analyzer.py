"""
Video analysis utilities for content extraction and categorization.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class VideoAnalyzer:
    """Analyzes video files for content and categorization."""
    
    def __init__(self, max_file_size_bytes: int):
        self.max_file_size_bytes = max_file_size_bytes
    
    def analyze_video(self, file_path: Path) -> str:
        """Analyze video files using OpenAI Vision API."""
        try:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            analysis = f"Video file: {file_size:.1f}MB"
            
            # Try to extract a frame and analyze with Vision API
            try:
                video_analysis = self._analyze_video_with_vision_api(file_path)
                if video_analysis:
                    analysis += f" | Content: {video_analysis}"
            except Exception as e:
                analysis += f" | Video analysis failed: {str(e)[:30]}"
            
            return analysis
        except Exception:
            return "Video file (analysis unavailable)"
    
    def _analyze_video_with_vision_api(self, file_path: Path) -> str:
        """Analyze video content by extracting a frame and using Vision API."""
        try:
            # Extract a frame from the video using ffmpeg
            frame_path = self._extract_video_frame(file_path)
            if not frame_path or not frame_path.exists():
                return "Could not extract video frame for analysis"
            
            # Get base64 encoded frame
            base64_frame = self._get_image_base64(frame_path)
            if not base64_frame:
                return "Video frame too large for analysis"
            
            from openai import OpenAI
            client = OpenAI()
            
            # Analyze video frame with Vision API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this video frame and describe: 1) What type of content the video contains, 2) Any visible text or UI elements, 3) The purpose or context, 4) If it appears to be a screen recording, tutorial, meeting, or other content type. Keep it concise (max 100 words)."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_frame}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150
            )
            
            # Clean up extracted frame
            try:
                frame_path.unlink()
            except:
                pass
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Video analysis error: {str(e)[:50]}"
    
    def _extract_video_frame(self, file_path: Path) -> Optional[Path]:
        """Extract a frame from video using ffmpeg."""
        try:
            # Create temporary file for frame
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                frame_path = Path(temp_file.name)
            
            # Extract frame at 1 second mark using ffmpeg
            cmd = [
                'ffmpeg', '-i', str(file_path), 
                '-ss', '1', '-vframes', '1', 
                '-q:v', '2', str(frame_path), '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and frame_path.exists():
                return frame_path
            else:
                return None
                
        except Exception as e:
            return None
    
    def _get_image_base64(self, file_path: Path) -> Optional[str]:
        """Get base64 encoded image for OpenAI Vision API."""
        try:
            import base64
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception:
            return None
