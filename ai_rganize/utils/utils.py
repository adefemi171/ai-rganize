"""
Utility classes and functions for AIrganizer
"""

from rich.console import Console as RichConsole
from rich.panel import Panel as RichPanel
from rich.prompt import Confirm, Prompt


class Console:
    """Console wrapper for consistent output."""
    
    def __init__(self):
        self.console = RichConsole()
    
    def print(self, *args, **kwargs):
        """Print with rich formatting."""
        self.console.print(*args, **kwargs)
    
    def input(self, prompt: str = "") -> str:
        """Get user input."""
        return Prompt.ask(prompt)
    
    def confirm(self, message: str) -> bool:
        """Get user confirmation."""
        return Confirm.ask(message)


class Panel:
    """Panel wrapper."""
    
    @staticmethod
    def fit(content: str, **kwargs):
        """Create a fitted panel."""
        return RichPanel.fit(content, **kwargs)
