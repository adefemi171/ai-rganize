"""High-level AI-rganize interface."""

from typing import Dict, List, Optional
from pathlib import Path

from .organizers import BaseOrganizer, RuleBasedOrganizer, AIOrganizer
from .rate_limiting import RateLimiter
from .ai_client import create_ai_client, BaseAIClient
from .file_analysis import FileAnalyzer
from .permissions import PermissionHandler


class AI_rganize:
    """Main interface that delegates to rule-based or AI organizers."""
    
    def __init__(self, api_key: Optional[str] = None, max_file_size_mb: int = 10, 
                 use_ai: bool = True, batch_size: int = 5, max_cost: float = 1.0, 
                 model: str = 'gpt-4o', llm_provider: str = 'openai', max_folders: Optional[int] = None):
        self.use_ai = use_ai
        
        if use_ai:
            self.organizer = AIOrganizer(
                api_key=api_key, max_file_size_mb=max_file_size_mb,
                batch_size=batch_size, max_cost=max_cost,
                model=model, llm_provider=llm_provider, max_folders=max_folders
            )
        else:
            self.organizer = RuleBasedOrganizer(max_file_size_mb=max_file_size_mb)
    
    def check_permissions(self) -> bool:
        return self.organizer.check_permissions()
    
    def scan_files(self, directory: Path) -> List[Dict]:
        return self.organizer.scan_files(directory)
    
    def create_organization_plan(self, files: List[Dict], ai_limit: int = 50, verbose: bool = False) -> Dict:
        if self.use_ai:
            return self.organizer.create_organization_plan(files, ai_limit, verbose)
        return self.organizer.create_organization_plan(files, verbose)
    
    def display_organization_plan(self, plan: Dict, show_details: bool = True):
        self.organizer.display_organization_plan(plan, show_details)
    
    def create_backup(self, files: List[Dict]) -> bool:
        return self.organizer.create_backup(files)
    
    def execute_organization(self, plan: Dict, target_dir: Path) -> bool:
        return self.organizer.execute_organization(plan, target_dir)
    
    @property
    def target_dirs(self) -> Dict[str, Path]:
        return self.organizer.target_dirs


def create_rule_based_organizer(max_file_size_mb: int = 10) -> RuleBasedOrganizer:
    return RuleBasedOrganizer(max_file_size_mb=max_file_size_mb)


def create_ai_organizer(api_key: Optional[str] = None, max_file_size_mb: int = 10,
                       batch_size: int = 5, max_cost: float = 1.0, 
                       model: str = 'gpt-4o', llm_provider: str = 'openai',
                       max_folders: Optional[int] = None) -> AIOrganizer:
    return AIOrganizer(
        api_key=api_key, max_file_size_mb=max_file_size_mb,
        batch_size=batch_size, max_cost=max_cost,
        model=model, llm_provider=llm_provider, max_folders=max_folders
    )


def create_organizer(use_ai: bool = True, **kwargs) -> BaseOrganizer:
    return create_ai_organizer(**kwargs) if use_ai else create_rule_based_organizer(**kwargs)


__all__ = [
    'AI_rganize', 'BaseOrganizer', 'RuleBasedOrganizer', 'AIOrganizer',
    'RateLimiter', 'BaseAIClient', 'FileAnalyzer', 'PermissionHandler',
    'create_ai_client', 'create_rule_based_organizer', 'create_ai_organizer', 'create_organizer'
]