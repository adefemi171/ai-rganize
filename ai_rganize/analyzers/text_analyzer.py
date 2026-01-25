"""Text file analysis."""

from pathlib import Path


class TextAnalyzer:
    def analyze_text(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read(1000)  # First 1000 characters
            
            if not content.strip():
                return "Empty text file"
            
            # Look for common patterns
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in ['resume', 'cv', 'curriculum vitae']):
                return f"Text Resume/CV: {content[:100]}..."
            elif any(keyword in content_lower for keyword in ['cover letter', 'application']):
                return f"Text Cover Letter: {content[:100]}..."
            elif any(keyword in content_lower for keyword in ['invoice', 'receipt', 'bill']):
                return f"Text Financial Document: {content[:100]}..."
            elif any(keyword in content_lower for keyword in ['meeting', 'minutes', 'agenda']):
                return f"Text Meeting Notes: {content[:100]}..."
            elif any(keyword in content_lower for keyword in ['code', 'function', 'class', 'import']):
                return f"Text Code/Programming: {content[:100]}..."
            else:
                return f"Text Document: {content[:100]}..."
        
        except Exception as e:
            return f"Text analysis error: {str(e)[:50]}"
