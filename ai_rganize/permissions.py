"""
Permission handling for AIrganizer
"""

import os
import platform
from pathlib import Path
from typing import Dict


class PermissionHandler:
    """Handle file permissions across different operating systems."""

    def __init__(self):
        self.system = platform.system().lower()

    def check_permissions(self, target_dirs: Dict[str, Path]) -> bool:
        """Check read+write access to target directories (writes required to move files)."""
        print(f"Checking directory permissions on {platform.system()}...")

        accessible_dirs = []
        for name, path in target_dirs.items():
            if path is None:
                continue
            if path.exists() and os.access(path, os.R_OK | os.W_OK):
                accessible_dirs.append(name)
                print(f"✓ {name}: {path}")
            else:
                print(f"✗ {name}: {path} (not readable/writable or doesn't exist)")

        if not accessible_dirs:
            print("No accessible directories found!")
            print("Try organizing a specific directory: --directory /path/to/dir")
            return False

        print(f"Found {len(accessible_dirs)} accessible directories")
        return True


    def get_permission_guide(self) -> str:
        if self.system == "darwin":
            return """
🔒 macOS Permission Setup:
1. Open System Settings > Privacy & Security > Files and Folders (or Full Disk Access)
2. Add Terminal, Python, or your IDE to the list
3. Restart your terminal/IDE
4. Never run this tool with elevated privileges against system directories
"""
        elif self.system == "linux":
            return """
🔒 Linux Permission Setup:
1. Ensure your user has read/write access to the target directories
2. If using snap/flatpak, check app permissions
3. Do NOT use sudo with this tool — it moves personal files
4. Check ownership: ls -la ~/Documents
"""
        elif self.system == "windows":
            return """
🔒 Windows Permission Setup:
1. Ensure your user account has modify rights on the target folders
2. Check Windows Defender exclusions if scans are blocked
3. Prefer running as your normal user, not Administrator
4. Check UAC settings if prompts block access
"""
        else:
            return """
🔒 General Permission Setup:
1. Ensure you have read/write access to target directories
2. Check file ownership and permissions
3. Do not elevate privileges unless you fully understand the risk
"""
