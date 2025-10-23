"""
File organization classes - Rule-based and AI-powered
"""

# Import the new modular classes
from .base_organizer import BaseOrganizer
from .rule_based_organizer import RuleBasedOrganizer
from .ai_organizer import AIOrganizer
from .rate_limiting import RateLimiter
from .ai_client import create_ai_client, BaseAIClient

# Re-export for backward compatibility
__all__ = ['BaseOrganizer', 'RuleBasedOrganizer', 'AIOrganizer', 'RateLimiter', 'BaseAIClient', 'create_ai_client']