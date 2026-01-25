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
        self.console.print(*args, **kwargs)
    
    def input(self, prompt: str = "") -> str:
        return Prompt.ask(prompt)
    
    def confirm(self, message: str) -> bool:
        return Confirm.ask(message)


class Panel:
    @staticmethod
    def fit(content: str, **kwargs):
        return RichPanel.fit(content, **kwargs)
