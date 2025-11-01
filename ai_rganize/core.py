"""
Core AI-rganize functionality - Main entry point and high-level logic.
"""

from typing import Dict, List, Optional
from pathlib import Path

# Import all the modular components
from .organizers import BaseOrganizer, RuleBasedOrganizer, AIOrganizer
from .rate_limiting import RateLimiter
from .ai_client import create_ai_client, BaseAIClient
from .file_analysis import FileAnalyzer
from .permissions import PermissionHandler


class AI_rganize:
    """
    Main AI-rganize class - High-level interface for file organization.
    
    This class provides a simple interface that delegates to the appropriate
    organizer (rule-based or AI-powered) based on the configuration.
    """
    
    def __init__(self, api_key: Optional[str] = None, max_file_size_mb: int = 10, 
                 use_ai: bool = True, batch_size: int = 5, max_cost: float = 1.0, 
                 model: str = 'gpt-4o', llm_provider: str = 'openai', max_folders: Optional[int] = None):
        """
        Initialize AI-rganize with configuration.
        
        Args:
            api_key: API key for AI services
            max_file_size_mb: Maximum file size for analysis
            use_ai: Whether to use AI-powered categorization
            batch_size: Number of files to process in each AI batch
            max_cost: Maximum cost for AI processing
            model: AI model to use
            llm_provider: LLM provider (openai, claude, gemini, etc.)
            max_folders: Maximum number of folders to create (None = no limit)
        """
        self.api_key = api_key
        self.max_file_size_mb = max_file_size_mb
        self.use_ai = use_ai
        self.batch_size = batch_size
        self.max_cost = max_cost
        self.model = model
        self.llm_provider = llm_provider
        self.max_folders = max_folders
        
        # Initialize the appropriate organizer
        if use_ai:
            self.organizer = AIOrganizer(
                api_key=api_key,
                max_file_size_mb=max_file_size_mb,
                batch_size=batch_size,
                max_cost=max_cost,
                model=model,
                llm_provider=llm_provider,
                max_folders=max_folders
            )
        else:
            self.organizer = RuleBasedOrganizer(max_file_size_mb=max_file_size_mb)
    
    def check_permissions(self) -> bool:
        """Check if we have permission to access target directories."""
        return self.organizer.check_permissions()
    
    def scan_files(self, directory: Path) -> List[Dict]:
        """Scan directory for files to organize."""
        return self.organizer.scan_files(directory)
    
    def create_organization_plan(self, files: List[Dict], ai_limit: int = 50, verbose: bool = False) -> Dict:
        """Create organization plan."""
        if self.use_ai:
            return self.organizer.create_organization_plan(files, ai_limit, verbose)
        else:
            return self.organizer.create_organization_plan(files, verbose)
    
    def display_organization_plan(self, plan: Dict, show_details: bool = True):
        """Display the organization plan."""
        self.organizer.display_organization_plan(plan, show_details)
    
    def create_backup(self, files: List[Dict]) -> bool:
        """Create backup of files before organization."""
        return self.organizer.create_backup(files)
    
    def execute_organization(self, plan: Dict, target_dir: Path) -> bool:
        """Execute the organization plan."""
        return self.organizer.execute_organization(plan, target_dir)
    
    @property
    def target_dirs(self) -> Dict[str, Path]:
        """Get target directories."""
        return self.organizer.target_dirs


# Convenience functions for common operations
def create_rule_based_organizer(max_file_size_mb: int = 10) -> RuleBasedOrganizer:
    """Create a rule-based organizer."""
    return RuleBasedOrganizer(max_file_size_mb=max_file_size_mb)


def create_ai_organizer(api_key: Optional[str] = None, max_file_size_mb: int = 10,
                       batch_size: int = 5, max_cost: float = 1.0, 
                       model: str = 'gpt-4o', llm_provider: str = 'openai',
                       max_folders: Optional[int] = None) -> AIOrganizer:
    """Create an AI-powered organizer."""
    return AIOrganizer(
        api_key=api_key,
        max_file_size_mb=max_file_size_mb,
        batch_size=batch_size,
        max_cost=max_cost,
        model=model,
        llm_provider=llm_provider,
        max_folders=max_folders
    )


def create_organizer(use_ai: bool = True, **kwargs) -> BaseOrganizer:
    """Create an organizer based on the specified type."""
    if use_ai:
        return create_ai_organizer(**kwargs)
    else:
        return create_rule_based_organizer(**kwargs)


# Export the main classes and functions
__all__ = [
    'AI_rganize',
    'BaseOrganizer',
    'RuleBasedOrganizer', 
    'AIOrganizer',
    'RateLimiter',
    'BaseAIClient',
    'FileAnalyzer',
    'PermissionHandler',
    'create_ai_client',
    'create_rule_based_organizer',
    'create_ai_organizer',
    'create_organizer'
]