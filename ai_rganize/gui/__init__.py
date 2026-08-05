"""Local-only Flask dashboard for AI-rganize (127.0.0.1:8765)."""

from .app import HOST, PORT, create_app, main

__all__ = ["HOST", "PORT", "create_app", "main"]
