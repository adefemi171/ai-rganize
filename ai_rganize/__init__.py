"""AI-rganize - Intelligent file organization using AI."""

__version__ = "1.0.0"

from .organizers import RuleBasedOrganizer, AIOrganizer
from .cli import main

__all__ = ["RuleBasedOrganizer", "AIOrganizer", "main"]
