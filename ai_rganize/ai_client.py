"""AI client implementations for different LLM providers."""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .rate_limiting import RateLimiter

# Load prompt template
_PROMPT_FILE = Path(__file__).parent / 'prompts' / 'categorization.txt'
_PROMPT_TEMPLATE = _PROMPT_FILE.read_text() if _PROMPT_FILE.exists() else None


class BaseAIClient(ABC):
    """Base class for AI clients."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.api_key = api_key
        self.model = model
        self.rate_limiter = RateLimiter()
    
    @abstractmethod
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False, 
                        max_folders: Optional[int] = None, 
                        existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        """Categorize files and return folder names."""
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> float:
        """Get cost per 1K tokens for the current model."""
        pass
    
    def _build_prompt(self, batch_content: List[str], file_count: int,
                     max_folders: Optional[int] = None,
                     existing_folders: Optional[List[str]] = None,
                     remaining_folder_slots: Optional[int] = None) -> str:
        """Build the categorization prompt."""
        
        constraint = ""
        if max_folders:
            existing = existing_folders or []
            constraint = f"""
FOLDER LIMIT: Create at most {max_folders} total folders.
Already created ({len(existing)}): {', '.join(existing) if existing else 'none'}
New folders allowed: {remaining_folder_slots or 0}
Reuse existing folders when appropriate."""

        files_list = ''.join(batch_content)
        
        if _PROMPT_TEMPLATE:
            return _PROMPT_TEMPLATE.format(
                file_count=file_count,
                constraint=constraint,
                files_list=files_list
            )
        
        # Fallback if prompt file not found
        return f"""Categorize these {file_count} files into folders.
{constraint}

FILES:
{files_list}

Return {file_count} folder names, one per line:"""

    def _prepare_batch(self, file_batch: List[Dict], verbose: bool = False) -> List[str]:
        """Prepare file descriptions for the prompt."""
        from .file_analysis import FileAnalyzer
        
        analyzer = FileAnalyzer()
        batch_content = []
        
        for f in file_batch:
            path = f['path']
            if isinstance(path, str):
                path = Path(path)
            
            if f.get('content'):
                preview = f['content'][:500]
            else:
                preview = analyzer.get_file_content_preview(path)
            
            batch_content.append(f"- {f['name']} | {preview[:200]}\n")
        
        return batch_content

    def _parse_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse folder names from response."""
        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
        
        cleaned = []
        for line in lines:
            line = line.lstrip('0123456789.-) ').strip().strip('"\'')
            if line:
                cleaned.append(line)
        
        while len(cleaned) < expected_count:
            cleaned.append("Miscellaneous")
        
        return cleaned[:expected_count]


class OpenAIClient(BaseAIClient):
    """OpenAI GPT client."""
    
    MODEL_COSTS = {
        'gpt-4o': 0.01,
        'gpt-4o-mini': 0.00015,
        'gpt-4-turbo': 0.01,
        'gpt-4': 0.03,
        'gpt-3.5-turbo': 0.0005,
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'gpt-4o'):
        super().__init__(api_key, model or 'gpt-4o')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        import openai
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def get_cost_per_token(self) -> float:
        return self.MODEL_COSTS.get(self.model, 0.01)
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None,
                        existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        
        batch_content = self._prepare_batch(file_batch, verbose)
        prompt = self._build_prompt(batch_content, len(file_batch), 
                                   max_folders, existing_folders, remaining_folder_slots)
        
        for attempt in range(self.rate_limiter.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request for {len(file_batch)} files...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.1
                )
                
                if hasattr(response, 'usage'):
                    tokens = response.usage.total_tokens
                    self.rate_limiter.usage_stats['tokens_used'] += tokens
                    self.rate_limiter.usage_stats['cost_estimate'] += (tokens / 1000) * self.get_cost_per_token()
                self.rate_limiter.usage_stats['requests_made'] += 1
                
                result = response.choices[0].message.content.strip()
                if verbose:
                    print(f"    📝 Raw AI response preview: {result[:200]}...")
                
                folders = self._parse_response(result, len(file_batch))
                if verbose:
                    print(f"    📥 Received batch response: {folders}")
                
                return folders
                
            except Exception as e:
                if not self.rate_limiter.should_retry(attempt, e):
                    raise
                delay = self.rate_limiter.get_retry_delay(attempt)
                if verbose:
                    print(f"    ⚠️  Retry {attempt + 1} in {delay:.1f}s...")
                time.sleep(delay)
        
        raise Exception("Max retries exceeded")


class ClaudeClient(BaseAIClient):
    """Anthropic Claude client."""
    
    MODEL_COSTS = {
        'claude-sonnet-4-20250514': 0.009,
        'claude-sonnet-4': 0.009,
        'claude-haiku-4-20250514': 0.003,
        'claude-opus-4-20250514': 0.045,
        'claude-3-5-sonnet-20241022': 0.003,
        'claude-3-opus-20240229': 0.015,
        'claude-3-sonnet-20240229': 0.003,
        'claude-3-haiku-20240307': 0.00025,
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'claude-sonnet-4-20250514'):
        super().__init__(api_key, model or 'claude-sonnet-4-20250514')
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key)
    
    def get_cost_per_token(self) -> float:
        return self.MODEL_COSTS.get(self.model, 0.009)
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None,
                        existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        
        batch_content = self._prepare_batch(file_batch, verbose)
        prompt = self._build_prompt(batch_content, len(file_batch),
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
                
                if hasattr(message, 'usage'):
                    tokens = message.usage.input_tokens + message.usage.output_tokens
                    self.rate_limiter.usage_stats['tokens_used'] += tokens
                    self.rate_limiter.usage_stats['cost_estimate'] += (tokens / 1000) * self.get_cost_per_token()
                self.rate_limiter.usage_stats['requests_made'] += 1
                
                result = message.content[0].text.strip()
                if verbose:
                    print(f"    📝 Raw AI response preview: {result[:200]}...")
                
                folders = self._parse_response(result, len(file_batch))
                if verbose:
                    print(f"    📥 Received batch response: {folders}")
                
                return folders
                
            except Exception as e:
                if not self.rate_limiter.should_retry(attempt, e):
                    raise
                delay = self.rate_limiter.get_retry_delay(attempt)
                if verbose:
                    print(f"    ⚠️  Retry {attempt + 1} in {delay:.1f}s...")
                time.sleep(delay)
        
        raise Exception("Max retries exceeded")


class GeminiClient(BaseAIClient):
    """Google Gemini client."""
    
    MODEL_COSTS = {
        'gemini-2.5-pro-exp': 0.00125,
        'gemini-2.0-flash': 0.0001,
        'gemini-1.5-pro': 0.00125,
        'gemini-1.5-flash': 0.0001,
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'gemini-2.0-flash'):
        super().__init__(api_key, model or 'gemini-2.0-flash')
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
    
    def get_cost_per_token(self) -> float:
        return self.MODEL_COSTS.get(self.model, 0.00125)
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None,
                        existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        
        batch_content = self._prepare_batch(file_batch, verbose)
        prompt = self._build_prompt(batch_content, len(file_batch),
                                   max_folders, existing_folders, remaining_folder_slots)
        
        for attempt in range(self.rate_limiter.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request to Gemini for {len(file_batch)} files...")
                
                response = self.client.generate_content(
                    prompt,
                    generation_config={'temperature': 0.1, 'max_output_tokens': 1024}
                )
                
                self.rate_limiter.usage_stats['requests_made'] += 1
                
                result = response.text.strip()
                if verbose:
                    print(f"    📝 Raw AI response preview: {result[:200]}...")
                
                folders = self._parse_response(result, len(file_batch))
                if verbose:
                    print(f"    📥 Received batch response: {folders}")
                
                return folders
                
            except Exception as e:
                if not self.rate_limiter.should_retry(attempt, e):
                    raise
                delay = self.rate_limiter.get_retry_delay(attempt)
                if verbose:
                    print(f"    ⚠️  Retry {attempt + 1} in {delay:.1f}s...")
                time.sleep(delay)
        
        raise Exception("Max retries exceeded")


class OllamaClient(BaseAIClient):
    """Ollama local model client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'llama3.1'):
        super().__init__(api_key, model or 'llama3.1')
        self.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        import ollama
        self.client = ollama.Client(host=self.base_url)
    
    def get_cost_per_token(self) -> float:
        return 0.0
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None,
                        existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        
        batch_content = self._prepare_batch(file_batch, verbose)
        prompt = self._build_prompt(batch_content, len(file_batch),
                                   max_folders, existing_folders, remaining_folder_slots)
        
        for attempt in range(self.rate_limiter.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request to Ollama ({self.model}) for {len(file_batch)} files...")
                
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={'temperature': 0.1, 'num_predict': 1024}
                )
                
                self.rate_limiter.usage_stats['requests_made'] += 1
                
                result = response['response'].strip()
                if verbose:
                    print(f"    📝 Raw AI response preview: {result[:200]}...")
                
                folders = self._parse_response(result, len(file_batch))
                if verbose:
                    print(f"    📥 Received batch response: {folders}")
                
                return folders
                
            except Exception as e:
                if not self.rate_limiter.should_retry(attempt, e):
                    raise
                delay = self.rate_limiter.get_retry_delay(attempt)
                if verbose:
                    print(f"    ⚠️  Retry {attempt + 1} in {delay:.1f}s...")
                time.sleep(delay)
        
        raise Exception("Max retries exceeded")


class MistralClient(BaseAIClient):
    """Mistral AI client."""
    
    MODEL_COSTS = {
        'mistral-small-latest': 0.001,
        'mistral-medium-latest': 0.0027,
        'mistral-large-latest': 0.004,
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'mistral-small-latest'):
        super().__init__(api_key, model or 'mistral-small-latest')
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not set")
        
        from mistralai import Mistral
        self.client = Mistral(api_key=self.api_key)
    
    def get_cost_per_token(self) -> float:
        return self.MODEL_COSTS.get(self.model, 0.002)
    
    def categorize_files(self, file_batch: List[Dict], verbose: bool = False,
                        max_folders: Optional[int] = None,
                        existing_folders: Optional[List[str]] = None,
                        remaining_folder_slots: Optional[int] = None) -> List[str]:
        
        batch_content = self._prepare_batch(file_batch, verbose)
        prompt = self._build_prompt(batch_content, len(file_batch),
                                   max_folders, existing_folders, remaining_folder_slots)
        
        for attempt in range(self.rate_limiter.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                if verbose:
                    print(f"    📡 Sending batch request to Mistral for {len(file_batch)} files...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024
                )
                
                if hasattr(response, 'usage'):
                    tokens = response.usage.total_tokens
                    self.rate_limiter.usage_stats['tokens_used'] += tokens
                    self.rate_limiter.usage_stats['cost_estimate'] += (tokens / 1000) * self.get_cost_per_token()
                self.rate_limiter.usage_stats['requests_made'] += 1
                
                result = response.choices[0].message.content.strip()
                if verbose:
                    print(f"    📝 Raw AI response preview: {result[:200]}...")
                
                folders = self._parse_response(result, len(file_batch))
                if verbose:
                    print(f"    📥 Received batch response: {folders}")
                
                return folders
                
            except Exception as e:
                if not self.rate_limiter.should_retry(attempt, e):
                    raise
                delay = self.rate_limiter.get_retry_delay(attempt)
                if verbose:
                    print(f"    ⚠️  Retry {attempt + 1} in {delay:.1f}s...")
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
    """Create an AI client for the specified provider."""
    clients = {
        'openai': OpenAIClient,
        'claude': ClaudeClient,
        'gemini': GeminiClient,
        'ollama': OllamaClient,
        'mistral': MistralClient,
    }
    
    provider = provider.lower()
    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}. Options: {list(clients.keys())}")
    
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
