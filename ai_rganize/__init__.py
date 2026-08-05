"""AI-rganize - Intelligent file organization using AI."""

__version__ = "1.0.0"

from .cli import main
from .organizers import AIOrganizer, RuleBasedOrganizer

__all__ = ["RuleBasedOrganizer", "AIOrganizer", "main"]
