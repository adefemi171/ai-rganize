"""
Permission handling for AIrganizer
"""

import os
import platform
from pathlib import Path
from typing import Dict, List


class PermissionHandler:
    """Handle file permissions across different operating systems."""
    
    def __init__(self):
        self.system = platform.system().lower()
    
    def check_permissions(self, target_dirs: Dict[str, Path]) -> bool:
        """Check if we have permission to access the target directories."""
        print(f"Checking directory permissions on {platform.system()}...")
        
        accessible_dirs = []
        for name, path in target_dirs.items():
            if path.exists() and os.access(path, os.R_OK):
                accessible_dirs.append(name)
                print(f"✓ {name}: {path}")
            else:
                # Special handling for macOS Library folder
                if name == 'Library' and self.system == 'darwin':
                    print(f"⚠ {name}: {path} (may require special permissions on macOS)")
                else:
                    print(f"✗ {name}: {path} (not accessible or doesn't exist)")
        
        if not accessible_dirs:
            print("No accessible directories found!")
            print("Try organizing a specific directory: --directory /path/to/dir")
            return False
        
        print(f"Found {len(accessible_dirs)} accessible directories")
        return True
    
    
    def get_permission_guide(self) -> str:
        """Get OS-specific permission setup guide."""
        if self.system == "darwin":  # macOS
            return """
🔒 macOS Permission Setup:
1. Open System Preferences > Security & Privacy > Privacy
2. Select 'Full Disk Access' or 'Files and Folders'
3. Add Terminal, Python, or your IDE to the list
4. Restart your terminal/IDE
"""
        elif self.system == "linux":  # Linux
            return """
🔒 Linux Permission Setup:
1. Make sure you have read/write access to your home directory
2. If using snap/flatpak, check app permissions
3. For restricted directories, you may need to run with sudo
4. Check file ownership: ls -la ~/Documents
"""
        elif self.system == "windows":  # Windows
            return """
🔒 Windows Permission Setup:
1. Run as Administrator if needed
2. Check Windows Defender exclusions
3. Ensure your user account has full control
4. Check UAC (User Account Control) settings
"""
        else:
            return """
🔒 General Permission Setup:
1. Ensure you have read/write access to target directories
2. Check file ownership and permissions
3. Run with appropriate privileges if needed
"""
