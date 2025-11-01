"""
Audio analysis utilities for content extraction and categorization.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class AudioAnalyzer:
    """Analyzes audio files for content and categorization."""
    
    def __init__(self, max_file_size_bytes: int):
        self.max_file_size_bytes = max_file_size_bytes
    
    def analyze_audio(self, file_path: Path) -> str:
        """Analyze audio files using OpenAI Whisper API."""
        try:
            file_size = file_path.stat().st_size / 1024  # KB
            analysis = f"Audio file: {file_size:.1f}KB"
            
            # Try to transcribe and analyze with Whisper API
            try:
                audio_analysis = self._analyze_audio_with_whisper_api(file_path)
                if audio_analysis:
                    analysis += f" | Content: {audio_analysis}"
            except Exception as e:
                analysis += f" | Audio analysis failed: {str(e)[:30]}"
            
            return analysis
        except Exception:
            return "Audio file (analysis unavailable)"
    
    def _analyze_audio_with_whisper_api(self, file_path: Path) -> str:
        """Analyze audio content using OpenAI Whisper API with audio preprocessing."""
        try:
            # Preprocess audio for better analysis (extract first 2 minutes)
            # Note: Large files will be preprocessed (first 2 minutes extracted) to manage processing time
            processed_audio_path = self._preprocess_audio_for_analysis(file_path)
            if not processed_audio_path or not processed_audio_path.exists():
                return "Could not preprocess audio for analysis"
            
            from openai import OpenAI
            client = OpenAI()
            
            # Transcribe audio with Whisper API
            with open(processed_audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            # Clean up processed audio file
            try:
                processed_audio_path.unlink()
            except:
                pass
            
            # Analyze the transcript
            if transcript and len(transcript.strip()) > 10:
                # Analyze transcript content
                analysis_prompt = f"Analyze this audio transcript and describe: 1) What type of content it is (meeting, music, speech, etc.), 2) Key topics or themes, 3) The purpose or context. Keep it concise (max 100 words).\n\nTranscript: {transcript[:500]}"
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    max_tokens=150
                )
                
                return response.choices[0].message.content.strip()
            else:
                return "Audio transcript too short or unclear"
            
        except Exception as e:
            return f"Whisper API error: {str(e)[:50]}"
    
    def _preprocess_audio_for_analysis(self, file_path: Path) -> Optional[Path]:
        """Preprocess audio file for better analysis using ffmpeg."""
        try:
            # Create temporary file for processed audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                processed_path = Path(temp_file.name)
            
            # Extract first 2 minutes and convert to WAV for better Whisper analysis
            cmd = [
                'ffmpeg', '-i', str(file_path), 
                '-t', '120',  # First 2 minutes (120 seconds)
                '-ar', '16000',  # 16kHz sample rate (optimal for Whisper)
                '-ac', '1',  # Mono
                '-y', str(processed_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and processed_path.exists():
                return processed_path
            else:
                return None
                
        except Exception as e:
            return None
