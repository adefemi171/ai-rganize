"""
Utility classes and functions for AIrganizer
"""

import platform
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm, Prompt


class Console:
    """Console wrapper for consistent output."""
    
    def __init__(self):
        self.console = Console()
    
    def print(self, *args, **kwargs):
        """Print with rich formatting."""
        self.console.print(*args, **kwargs)
    
    def input(self, prompt: str = "") -> str:
        """Get user input."""
        return Prompt.ask(prompt)
    
    def confirm(self, message: str) -> bool:
        """Get user confirmation."""
        return Confirm.ask(message)


class Progress:
    """Progress bar wrapper."""
    
    def __init__(self, console=None):
        self.console = console or Console().console
    
    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()
    
    def add_task(self, description: str, total: int = None):
        """Add a progress task."""
        return self.progress.add_task(description, total=total)
    
    def advance(self, task_id: int, advance: int = 1):
        """Advance a progress task."""
        self.progress.advance(task_id, advance)


class SpinnerColumn:
    """Spinner column for progress bars."""
    pass


class TextColumn:
    """Text column for progress bars."""
    pass


class Table:
    """Table wrapper."""
    
    def __init__(self, title: str = None):
        self.table = Table(title=title)
    
    def add_column(self, name: str, **kwargs):
        """Add a column to the table."""
        self.table.add_column(name, **kwargs)
    
    def add_row(self, *args, **kwargs):
        """Add a row to the table."""
        self.table.add_row(*args, **kwargs)
    
    def print(self, console=None):
        """Print the table."""
        if console is None:
            console = Console().console
        console.print(self.table)


class Panel:
    """Panel wrapper."""
    
    @staticmethod
    def fit(content: str, **kwargs):
        """Create a fitted panel."""
        return Panel(content, **kwargs)
