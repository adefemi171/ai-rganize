"""
AI client abstraction for different LLM providers.
"""

import os
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import openai

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

from .rate_limiting import RateLimiter


class BaseAIClient(ABC):
    """Base class for AI clients."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.api_key = api_key
        self.model = model
        self.rate_limiter = RateLimiter()
    
    @abstractmethod
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False) -> List[str]:
        """Categorize a batch of files using AI."""
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> float:
        """Get cost per token for the current model."""
        pass


class OpenAIClient(BaseAIClient):
    """OpenAI client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'gpt-4o'):
        super().__init__(api_key, model)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.rate_limiter = RateLimiter(cost_per_token=self.get_cost_per_token())
    
    def get_cost_per_token(self) -> float:
        """Get cost per token based on the model."""
        model_costs = {
            'gpt-4o': 0.01,  # $0.005 input + $0.015 output average
            'gpt-4o-mini': 0.00015,  # $0.00015 input + $0.0006 output average
            'gpt-4': 0.03,  # $0.03 input + $0.06 output average
            'gpt-3.5-turbo': 0.0005,  # $0.0005 input + $0.0015 output average
        }
        return model_costs.get(self.model, 0.01)  # Default to GPT-4o pricing
    
    def get_document_analysis_cost(self, file_size_mb: float) -> float:
        """Calculate cost for document analysis using OpenAI File Search API."""
        # OpenAI File Search API costs:
        # - File upload: $0.10 per file
        # - Assistant usage: $0.01 per 1K tokens (GPT-4o)
        # - File storage: $0.10 per GB per day (minimal for our use)
        
        base_cost = 0.10  # File upload cost
        processing_cost = (file_size_mb * 0.01)  # Processing cost based on file size
        return base_cost + processing_cost
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False) -> List[str]:
        """Categorize files using OpenAI API."""
        from .file_analysis import FileAnalyzer
        
        analyzer = FileAnalyzer()
        batch_content = []
        
        for file_info in file_batch:
            content_preview = analyzer.get_file_content_preview(file_info['path'])
            batch_content.append(f"File: {file_info['name']} (Location: {file_info['path'].parent.name}, Content: {content_preview})\n")
        
        prompt = self._create_categorization_prompt(batch_content, len(file_batch))
        
        # Enhanced retry logic with exponential backoff
        for attempt in range(self.rate_limiter.max_retries):
            try:
                # Wait if needed to respect rate limits
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request for {len(file_batch)} files...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1
                )
                
                # Update usage stats from response
                if hasattr(response, 'usage'):
                    tokens_used = response.usage.total_tokens
                    self.rate_limiter.usage_stats['tokens_used'] += tokens_used
                    # Calculate cost based on model pricing
                    cost_per_token = self.get_cost_per_token()
                    cost = (tokens_used / 1000) * cost_per_token
                    self.rate_limiter.usage_stats['cost_estimate'] += cost
                
                # Monitor usage headers for additional rate limiting info
                if hasattr(response, 'response_headers'):
                    self.rate_limiter.update_usage_from_headers(response.response_headers)
                
                # Update request count
                self.rate_limiter.usage_stats['requests_made'] += 1
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                folder_names = [line.strip() for line in response_text.split('\n') if line.strip()]
                
                if len(folder_names) != len(file_batch):
                    raise ValueError(f"Expected {len(file_batch)} folder names, got {len(folder_names)}")
                
                if verbose:
                    print(f"    📥 Received batch response: {folder_names}")
                
                return folder_names
                
            except Exception as e:
                if not self.rate_limiter.should_retry(attempt, e):
                    raise e
                
                delay = self.rate_limiter.get_retry_delay(attempt)
                if delay > 0:
                    if verbose:
                        print(f"    ⚠️  Retry {attempt + 1}/{self.rate_limiter.max_retries} in {delay:.1f}s...")
                    import time
                    time.sleep(delay)
        
        raise Exception("Max retries exceeded")
    
    def _create_categorization_prompt(self, batch_content: str, file_count: int) -> str:
        """Create the AI prompt for file categorization."""
        return f"""
                Analyze these {file_count} files and their ACTUAL CONTENT to create meaningful, intelligent folder categories based on their PURPOSE, CONTENT, and SIMILARITIES.
                
                FILES TO ANALYZE:
                {''.join(batch_content)}
                
                CRITICAL INTELLIGENCE REQUIREMENTS:
                1. READ ACTUAL CONTENT: Analyze the real content of files (PDF text, document content, image descriptions, video content, audio transcripts)
                2. DETECT SIMILARITIES: Look for files that belong together (same person, family, project, theme, purpose)
                3. CREATE CONSISTENT NAMES: Use the SAME folder name for similar files across different subdirectories
                4. GROUP INTELLIGENTLY: Don't just categorize by file type - group by actual purpose and relationships
                5. CONSIDER CONTEXT: Look at file locations, names, content, and metadata to understand relationships
                6. FOLDER STRUCTURE ANALYSIS: Analyze existing folder structure to understand relationships and hierarchies
                7. CONTENT-BASED GROUPING: Group files by their actual content and purpose, not just file extensions
                8. VISUAL SIMILARITY DETECTION: For images, detect similar faces, objects, or scenes and group them together
                9. VIDEO CONTENT ANALYSIS: For videos, analyze content type (tutorials, meetings, recordings) and group accordingly
                10. AUDIO CONTENT ANALYSIS: For audio, analyze transcript content (meetings, music, speech) and group by purpose
                11. SCREENSHOT CONTENT ANALYSIS: For screenshots, detect programming content, UI elements, or similar applications and group accordingly
                12. FOLDER MERGING INTELLIGENCE: Detect when entire folders should be merged (look for folders containing related content that should be consolidated)
                13. INTELLIGENT REORGANIZATION: Create new folder structures that merge related subdirectories and files intelligently
                14. FAMILY NAME DETECTION: Look for shared surnames (e.g., files with common family names should be grouped together)
                15. SURNAME PATTERN MATCHING: If files share the same surname or family name, use the SAME folder name regardless of their current subdirectory
                
                SIMILARITY DETECTION PATTERNS:
                - Same person's documents (resumes, profiles, applications) → Use consistent person-based folder names
                - FAMILY RELATIONSHIPS: Same surname or family name → Group by family relationship
                - Related projects or work (same company, role, field) → Group by project/company
                - Similar content themes (travel photos, family events, work screenshots) → Group by theme
                - Files from different subdirectories but same purpose → Use same folder name
                - Related file types with common purpose → Group by purpose, not just type
                - COMMON PATTERNS: Look for shared surnames, prefixes, or naming patterns that indicate relationships
                - VISUAL SIMILARITIES: Same faces in photos, similar objects, or scenes → Group by visual content
                - SCREENSHOT CONTENT: Programming code, UI elements, similar applications → Group by technical content
                - FOLDER MERGING: Detect when folders contain related content that should be consolidated → Merge into single folder
                - INTELLIGENT REORGANIZATION: Create new folder structures that consolidate related subdirectories
                
                EXAMPLES OF INTELLIGENT GROUPING:
                - All resumes from same person → "Professional_Resumes" or "Job_Applications"
                - FAMILY GROUPING: Multiple family members → "Family_Professional_Documents" or "Family_Profiles"
                - FAMILY SURNAME DETECTION: Files with shared surnames → "Family_Documents" (group by family relationship)
                - VISUAL GROUPING: Photos with same person's face → "Personal_Photos" or "Family_Member_Photos"
                - SCREENSHOT GROUPING: Programming code screenshots → "Programming_Screenshots" or "Code_Development_Screenshots"
                - VIDEO GROUPING: Tutorial videos → "Tutorial_Videos" or "Educational_Content"
                - AUDIO GROUPING: Meeting recordings → "Meeting_Recordings" or "Work_Audio"
                - FOLDER MERGING: Related folders containing similar content → "International_Professional_Documents" or "Multi_Country_Applications"
                - INTELLIGENT REORGANIZATION: Related subdirectories → Merge into single organized folder structure
                - All Microsoft project files → "Microsoft_Project_Files" 
                - All family vacation photos → "Family_Vacation_Photos"
                - All work meeting screenshots → "Work_Meeting_Screenshots"
                - All financial documents from 2024 → "Financial_Records_2024"
                - All creative design work → "Creative_Design_Projects"
                - All tutorial videos → "Educational_Videos"
                - All meeting recordings → "Meeting_Recordings"
                - All music files → "Music_Collection"
                - All technical documentation → "Technical_Documentation"
                - All wedding event photos → "Wedding_Event_Photos"
                - All software engineering job applications → "Software_Engineering_Applications"
                - All system files → "System_Files"
                
                INTELLIGENCE RULES:
                1. Use IDENTICAL folder names for similar files (even from different subdirectories)
                2. Group by PERSON, FAMILY, PROJECT, THEME, or PURPOSE - not just file type
                3. Look for patterns in filenames, content, and location context
                4. Create meaningful, descriptive folder names that humans would understand
                5. Consider the actual relationship between files, not just their current location
                6. DETECT FAMILY RELATIONSHIPS: If files share the same surname or family name, group them together
                7. MERGE SIMILAR PATTERNS: Look for common prefixes, suffixes, or naming patterns that indicate relationships
                8. ANALYZE VISUAL CONTENT: For images, detect similar faces, objects, or scenes and group accordingly
                9. ANALYZE SCREENSHOT CONTENT: For screenshots, detect programming content, UI elements, or similar applications
                10. USE ALIASES: Never use actual people's names in folder names - use roles, purposes, or generic identifiers instead
                11. FOLDER MERGING INTELLIGENCE: Detect when entire subdirectories should be merged (look for related content across different folders)
                12. INTELLIGENT REORGANIZATION: Create new folder structures that consolidate related subdirectories and files
                
                RESPOND WITH EXACTLY {file_count} FOLDER NAMES, one per line, in the same order as the files.
                Use the SAME folder name for files that belong together, regardless of their current subdirectory.
                NEVER use actual people's names - use roles, purposes, or generic identifiers instead.
                CONSIDER FOLDER MERGING: If files from different subdirectories belong together, use the SAME folder name to merge them.
                Example:
                Professional_Resumes
                Professional_Resumes
                International_Professional_Documents
                International_Professional_Documents
                Microsoft_Project_Files
                Family_Vacation_Photos
                """


class ClaudeClient(BaseAIClient):
    """Claude client implementation (future)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'claude-3-sonnet'):
        raise ValueError("Claude provider not yet implemented. Use --llm-provider openai for now.")
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False) -> List[str]:
        raise NotImplementedError("Claude client not implemented yet")
    
    def get_cost_per_token(self) -> float:
        raise NotImplementedError("Claude client not implemented yet")


class GeminiClient(BaseAIClient):
    """Gemini client implementation (future)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'gemini-pro'):
        raise ValueError("Gemini provider not yet implemented. Use --llm-provider openai for now.")
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False) -> List[str]:
        raise NotImplementedError("Gemini client not implemented yet")
    
    def get_cost_per_token(self) -> float:
        raise NotImplementedError("Gemini client not implemented yet")


def create_ai_client(provider: str, api_key: Optional[str] = None, model: str = None) -> BaseAIClient:
    """Factory function to create AI clients."""
    if provider == 'openai':
        return OpenAIClient(api_key, model)
    elif provider == 'claude':
        return ClaudeClient(api_key, model)
    elif provider == 'gemini':
        return GeminiClient(api_key, model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
