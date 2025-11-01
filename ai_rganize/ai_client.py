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
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False, 
                        max_folders: Optional[int] = None, existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        """Categorize a batch of files using AI.
        
        Args:
            file_batch: List of file dictionaries to categorize
            verbose: Whether to print verbose output
            max_folders: Maximum number of folders to create (None = no limit)
            existing_folders: List of folder names already created in previous batches
            remaining_folder_slots: Number of new folder slots remaining (if max_folders is set)
        
        Returns:
            List of folder names, one per file in the batch
        """
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> float:
        """Get cost per token for the current model."""
        pass
    
    def _create_categorization_prompt(self, batch_content: List[str], file_count: int, 
                                     max_folders: Optional[int] = None, 
                                     existing_folders: Optional[List[str]] = None,
                                     remaining_folder_slots: Optional[int] = None) -> str:
        """Create the AI prompt for file categorization (shared across all providers)."""
        
        # Build folder limit instructions
        folder_limit_instructions = ""
        if max_folders is not None:
            folder_limit_instructions = f"""
                CRITICAL FOLDER LIMIT CONSTRAINT:
                - You MUST create AT MOST {max_folders} unique folder names across ALL batches.
                - You have already created {len(existing_folders) if existing_folders else 0} folder(s): {', '.join(existing_folders) if existing_folders else 'none'}
                - You can create AT MOST {remaining_folder_slots} new unique folder name(s) in this batch.
                - IMPORTANT: Reuse existing folder names when files are similar to already-categorized files.
                - If you need a new category but have no slots remaining, assign files to the most similar existing folder.
                - Distribute files intelligently across the available folder names (existing + new).
                """
        
        return f"""
                Analyze these {file_count} files and their ACTUAL CONTENT to create meaningful, intelligent folder categories based on their PURPOSE, CONTENT, and SIMILARITIES.
                {folder_limit_instructions}
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
                CRITICAL: You MUST return EXACTLY {file_count} lines, one folder name per file. Each file gets its own line.
                Use the SAME folder name for files that belong together, regardless of their current subdirectory.
                NEVER use actual people's names - use roles, purposes, or generic identifiers instead.
                CONSIDER FOLDER MERGING: If files from different subdirectories belong together, use the SAME folder name to merge them.
                {f'FOLDER LIMIT ENFORCEMENT: You MUST use at most {remaining_folder_slots} new unique folder name(s). Reuse existing folder names when appropriate: {", ".join(existing_folders) if existing_folders else "none"}' if max_folders is not None and remaining_folder_slots is not None else ''}
                
                EXAMPLE FORMAT (exactly {file_count} lines):
                Professional_Resumes
                Professional_Resumes
                International_Professional_Documents
                International_Professional_Documents
                Microsoft_Project_Files
                Family_Vacation_Photos
                
                IMPORTANT: Write EXACTLY {file_count} folder names, one per line, nothing else.
                """


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
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None, existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        """Categorize files using OpenAI API."""
        from .file_analysis import FileAnalyzer
        
        analyzer = FileAnalyzer()
        batch_content = []
        
        for file_info in file_batch:
            content_preview = analyzer.get_file_content_preview(file_info['path'])
            batch_content.append(f"File: {file_info['name']} (Location: {file_info['path'].parent.name}, Content: {content_preview})\n")
        
        prompt = self._create_categorization_prompt(batch_content, len(file_batch), 
                                                   max_folders, existing_folders, remaining_folder_slots)
        
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
                    max_tokens=1024,  # Increased to ensure full response
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
                
                if verbose:
                    print(f"    📝 Raw AI response preview: {response_text[:200]}...")
                
                # Try to parse folder names - handle various formats
                folder_names = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # Remove numbering/bullets if present (e.g., "1. Folder_Name" or "- Folder_Name")
                    line = line.lstrip('0123456789.-) ').strip()
                    # Remove quotes if present
                    line = line.strip('"\'')
                    if line:
                        folder_names.append(line)
                
                # If we got fewer folder names than files, there might be an issue
                if len(folder_names) < len(file_batch):
                    if verbose:
                        print(f"    ⚠️  Warning: AI returned {len(folder_names)} folder names for {len(file_batch)} files")
                        print(f"    📝 Full response: {response_text}")
                    raise ValueError(f"Expected {len(file_batch)} folder names (one per file), but got {len(folder_names)}. Response: {response_text[:500]}")
                
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
    


class ClaudeClient(BaseAIClient):
    """Anthropic Claude client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'claude-sonnet-4-20250514'):  # Latest Claude Sonnet 4.5
        super().__init__(api_key, model)
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.rate_limiter = RateLimiter(cost_per_token=self.get_cost_per_token())
    
    def get_cost_per_token(self) -> float:
        """Get cost per token based on the model."""
        model_costs = {
            # Claude 4.5 Series (Latest)
            'claude-sonnet-4-20250514': 0.009,  # $3 input + $15 output per MTok = $9/MTok avg = $0.009/1K
            'claude-sonnet-4': 0.009,
            'claude-haiku-4-20250514': 0.003,  # $1 input + $5 output per MTok = $3/MTok avg = $0.003/1K
            'claude-haiku-4': 0.003,
            'claude-opus-4-20250514': 0.045,  # $15 input + $75 output per MTok = $45/MTok avg = $0.045/1K
            'claude-opus-4': 0.045,
            # Claude 3.5 Series
            'claude-3-5-sonnet-20241022': 0.003,  # $0.003 input + $0.015 output average
            'claude-3-5-sonnet': 0.003,
            # Claude 3 Series (Legacy)
            'claude-3-opus-20240229': 0.015,  # $0.015 input + $0.075 output average
            'claude-3-opus': 0.015,
            'claude-3-sonnet-20240229': 0.003,  # $0.003 input + $0.015 output average
            'claude-3-sonnet': 0.003,
            'claude-3-haiku-20240307': 0.00025,  # $0.00025 input + $0.00125 output average
            'claude-3-haiku': 0.00025,
        }
        return model_costs.get(self.model, 0.009)  # Default to Claude Sonnet 4.5 pricing
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None, existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        """Categorize files using Anthropic Claude API."""
        from .file_analysis import FileAnalyzer
        
        analyzer = FileAnalyzer()
        batch_content = []
        
        for file_info in file_batch:
            content_preview = analyzer.get_file_content_preview(file_info['path'])
            batch_content.append(f"File: {file_info['name']} (Location: {file_info['path'].parent.name}, Content: {content_preview})\n")
        
        prompt = self._create_categorization_prompt(batch_content, len(file_batch),
                                                   max_folders, existing_folders, remaining_folder_slots)
        
        for attempt in range(self.rate_limiter.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request to Claude for {len(file_batch)} files...")
                
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Update usage stats
                if hasattr(message, 'usage'):
                    input_tokens = message.usage.input_tokens
                    output_tokens = message.usage.output_tokens
                    total_tokens = input_tokens + output_tokens
                    self.rate_limiter.usage_stats['tokens_used'] += total_tokens
                    cost_per_token = self.get_cost_per_token()
                    cost = (total_tokens / 1000) * cost_per_token
                    self.rate_limiter.usage_stats['cost_estimate'] += cost
                
                self.rate_limiter.usage_stats['requests_made'] += 1
                
                # Parse response
                response_text = message.content[0].text.strip()
                
                if verbose:
                    print(f"    📝 Raw AI response preview: {response_text[:200]}...")
                
                # Try to parse folder names - handle various formats
                folder_names = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # Remove numbering/bullets if present (e.g., "1. Folder_Name" or "- Folder_Name")
                    line = line.lstrip('0123456789.-) ').strip()
                    # Remove quotes if present
                    line = line.strip('"\'')
                    if line:
                        folder_names.append(line)
                
                # If we got fewer folder names than files, there might be an issue
                if len(folder_names) < len(file_batch):
                    if verbose:
                        print(f"    ⚠️  Warning: AI returned {len(folder_names)} folder names for {len(file_batch)} files")
                        print(f"    📝 Full response: {response_text}")
                    raise ValueError(f"Expected {len(file_batch)} folder names (one per file), but got {len(folder_names)}. Response: {response_text[:500]}")
                
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


class GeminiClient(BaseAIClient):
    """Google Gemini client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'gemini-2.5-pro-exp'):  # Latest Gemini 2.5 Pro
        super().__init__(api_key, model)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
        self.rate_limiter = RateLimiter(cost_per_token=self.get_cost_per_token())
    
    def get_cost_per_token(self) -> float:
        """Get cost per token based on the model."""
        model_costs = {
            # Gemini 2.5 Series (Latest)
            'gemini-2.5-pro-exp': 0.00125,  # Pricing TBD - using 1.5 Pro as estimate
            'gemini-2.5-pro': 0.00125,
            # Gemini 2.0 Series
            'gemini-2.0-flash-exp': 0.000075,  # $0.000075 input + $0.0003 output average
            'gemini-2.0-flash': 0.000075,
            # Gemini 1.5 Series
            'gemini-1.5-pro': 0.00125,  # $0.00125 input + $0.005 output average
            'gemini-1.5-flash': 0.000075,  # $0.000075 input + $0.0003 output average
            # Gemini 1.0 Series (Legacy)
            'gemini-pro': 0.0005,  # $0.0005 input + $0.0015 output average
        }
        return model_costs.get(self.model, 0.00125)  # Default to Gemini 2.5 Pro pricing
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None, existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        """Categorize files using Google Gemini API."""
        from .file_analysis import FileAnalyzer
        
        analyzer = FileAnalyzer()
        batch_content = []
        
        for file_info in file_batch:
            content_preview = analyzer.get_file_content_preview(file_info['path'])
            batch_content.append(f"File: {file_info['name']} (Location: {file_info['path'].parent.name}, Content: {content_preview})\n")
        
        prompt = self._create_categorization_prompt(batch_content, len(file_batch),
                                                   max_folders, existing_folders, remaining_folder_slots)
        
        for attempt in range(self.rate_limiter.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request to Gemini for {len(file_batch)} files...")
                
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': 1024,
                    }
                )
                
                # Update usage stats (Gemini doesn't always provide detailed usage)
                self.rate_limiter.usage_stats['requests_made'] += 1
                # Estimate tokens (rough approximation)
                estimated_tokens = len(prompt.split()) * 1.3 + len(response.text.split()) * 1.3
                self.rate_limiter.usage_stats['tokens_used'] += int(estimated_tokens)
                cost_per_token = self.get_cost_per_token()
                cost = (estimated_tokens / 1000) * cost_per_token
                self.rate_limiter.usage_stats['cost_estimate'] += cost
                
                # Parse response
                response_text = response.text.strip()
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


class OllamaClient(BaseAIClient):
    """Ollama client implementation for local Llama and other models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'llama3.1', base_url: Optional[str] = None):
        super().__init__(api_key, model)
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package is required. Install with: pip install ollama")
        
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        # Ollama doesn't require API keys for local use, but supports them for remote instances
        self.client = ollama.Client(host=self.base_url)
        self.rate_limiter = RateLimiter(cost_per_token=0.0)  # Local models are free
    
    def get_cost_per_token(self) -> float:
        """Get cost per token - Ollama is free for local use."""
        return 0.0
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None, existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        """Categorize files using Ollama API."""
        from .file_analysis import FileAnalyzer
        
        analyzer = FileAnalyzer()
        batch_content = []
        
        for file_info in file_batch:
            content_preview = analyzer.get_file_content_preview(file_info['path'])
            batch_content.append(f"File: {file_info['name']} (Location: {file_info['path'].parent.name}, Content: {content_preview})\n")
        
        prompt = self._create_categorization_prompt(batch_content, len(file_batch),
                                                   max_folders, existing_folders, remaining_folder_slots)
        
        for attempt in range(self.rate_limiter.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request to Ollama ({self.model}) for {len(file_batch)} files...")
                
                # Ollama client API
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': 0.1,
                        'num_predict': 1024,
                    }
                )
                
                # Update usage stats
                self.rate_limiter.usage_stats['requests_made'] += 1
                # Ollama is free, so cost is 0
                
                # Parse response (Ollama returns response directly or in 'response' field)
                if isinstance(response, dict):
                    response_text = response.get('response', str(response)).strip()
                else:
                    response_text = str(response).strip()
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


class MistralClient(BaseAIClient):
    """Mistral AI client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'mistral-large-latest'):
        super().__init__(api_key, model)
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("mistralai package is required. Install with: pip install mistralai")
        
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")
        
        self.client = Mistral(api_key=self.api_key)
        self.rate_limiter = RateLimiter(cost_per_token=self.get_cost_per_token())
    
    def get_cost_per_token(self) -> float:
        """Get cost per token based on the model."""
        model_costs = {
            'mistral-large-latest': 0.002,  # $0.002 input + $0.006 output average
            'mistral-medium-latest': 0.00027,  # $0.00027 input + $0.0008 output average
            'mistral-small-latest': 0.0002,  # $0.0002 input + $0.0006 output average
        }
        return model_costs.get(self.model, 0.002)  # Default to Mistral Large pricing
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None, existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        """Categorize files using Mistral AI API."""
        from .file_analysis import FileAnalyzer
        
        analyzer = FileAnalyzer()
        batch_content = []
        
        for file_info in file_batch:
            content_preview = analyzer.get_file_content_preview(file_info['path'])
            batch_content.append(f"File: {file_info['name']} (Location: {file_info['path'].parent.name}, Content: {content_preview})\n")
        
        prompt = self._create_categorization_prompt(batch_content, len(file_batch),
                                                   max_folders, existing_folders, remaining_folder_slots)
        
        for attempt in range(self.rate_limiter.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request to Mistral for {len(file_batch)} files...")
                
                # Mistral AI uses chat.completions.create (similar to OpenAI)
                chat_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024
                )
                
                # Update usage stats
                if hasattr(chat_response, 'usage'):
                    total_tokens = chat_response.usage.total_tokens
                    self.rate_limiter.usage_stats['tokens_used'] += total_tokens
                    cost_per_token = self.get_cost_per_token()
                    cost = (total_tokens / 1000) * cost_per_token
                    self.rate_limiter.usage_stats['cost_estimate'] += cost
                
                self.rate_limiter.usage_stats['requests_made'] += 1
                
                # Parse response (Mistral uses similar structure to OpenAI)
                response_text = chat_response.choices[0].message.content.strip()
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


class OpenRouterClient(BaseAIClient):
    """OpenRouter client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'anthropic/claude-3.5-sonnet'):
        super().__init__(api_key, model)
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        # OpenRouter uses OpenAI-compatible API with custom base URL
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.rate_limiter = RateLimiter(cost_per_token=self.get_cost_per_token())
    
    def get_cost_per_token(self) -> float:
        """Get cost per token based on the model."""
        model_costs = {
            # Anthropic models via OpenRouter
            'anthropic/claude-3.5-sonnet': 0.003,  # $0.003 per 1K tokens (avg)
            'anthropic/claude-3-opus': 0.015,  # $0.015 per 1K tokens (avg)
            'anthropic/claude-3-haiku': 0.00025,  # $0.00025 per 1K tokens (avg)
            # OpenAI models via OpenRouter
            'openai/gpt-4o': 0.01,  # $0.01 per 1K tokens (avg)
            'openai/gpt-4o-mini': 0.00015,  # $0.00015 per 1K tokens (avg)
            # Google models via OpenRouter
            'google/gemini-2.0-flash-exp': 0.000075,  # $0.000075 per 1K tokens (avg)
            'google/gemini-pro-1.5': 0.00125,  # $0.00125 per 1K tokens (avg)
            # Meta models via OpenRouter
            'meta-llama/llama-3.1-70b-instruct': 0.00035,  # $0.00035 per 1K tokens (avg)
        }
        return model_costs.get(self.model, 0.003)  # Default to Claude 3.5 Sonnet pricing
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None, existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        """Categorize files using OpenRouter API."""
        from .file_analysis import FileAnalyzer
        
        analyzer = FileAnalyzer()
        batch_content = []
        
        for file_info in file_batch:
            content_preview = analyzer.get_file_content_preview(file_info['path'])
            batch_content.append(f"File: {file_info['name']} (Location: {file_info['path'].parent.name}, Content: {content_preview})\n")
        
        prompt = self._create_categorization_prompt(batch_content, len(file_batch),
                                                   max_folders, existing_folders, remaining_folder_slots)
        
        # Enhanced retry logic with exponential backoff
        for attempt in range(self.rate_limiter.max_retries):
            try:
                # Wait if needed to respect rate limits
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request to OpenRouter ({self.model}) for {len(file_batch)} files...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.1,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/yourusername/ai-rganize",
                        "X-Title": "AI-rganize"
                    }
                )
                
                # Update usage stats from response
                if hasattr(response, 'usage'):
                    tokens_used = response.usage.total_tokens
                    self.rate_limiter.usage_stats['tokens_used'] += tokens_used
                    # Calculate cost based on model pricing
                    cost_per_token = self.get_cost_per_token()
                    cost = (tokens_used / 1000) * cost_per_token
                    self.rate_limiter.usage_stats['cost_estimate'] += cost
                
                # Update request count
                self.rate_limiter.usage_stats['requests_made'] += 1
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                
                if verbose:
                    print(f"    📝 Raw AI response preview: {response_text[:200]}...")
                
                # Try to parse folder names - handle various formats
                folder_names = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # Remove numbering/bullets if present (e.g., "1. Folder_Name" or "- Folder_Name")
                    line = line.lstrip('0123456789.-) ').strip()
                    # Remove quotes if present
                    line = line.strip('"\'')
                    if line:
                        folder_names.append(line)
                
                # If we got fewer folder names than files, there might be an issue
                if len(folder_names) < len(file_batch):
                    if verbose:
                        print(f"    ⚠️  Warning: AI returned {len(folder_names)} folder names for {len(file_batch)} files")
                        print(f"    📝 Full response: {response_text}")
                    raise ValueError(f"Expected {len(file_batch)} folder names (one per file), but got {len(folder_names)}. Response: {response_text[:500]}")
                
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


def create_ai_client(provider: str, api_key: Optional[str] = None, model: str = None) -> BaseAIClient:
    """Factory function to create AI clients."""
    provider_lower = provider.lower()
    
    if provider_lower == 'openai':
        return OpenAIClient(api_key, model)
    elif provider_lower == 'claude':
        return ClaudeClient(api_key, model)
    elif provider_lower == 'gemini':
        return GeminiClient(api_key, model)
    elif provider_lower == 'ollama':
        return OllamaClient(api_key, model)
    elif provider_lower == 'mistral':
        return MistralClient(api_key, model)
    elif provider_lower == 'openrouter':
        return OpenRouterClient(api_key, model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: openai, claude, gemini, ollama, mistral, openrouter")
