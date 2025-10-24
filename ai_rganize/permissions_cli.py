"""
Permission setup CLI for AIrganizer
"""

import click
from pathlib import Path
from .organizer import BaseOrganizer
from .permissions import PermissionHandler


@click.command()
def main():
    """AIrganizer Permission Setup - Check and configure file access permissions."""
    
    print("🔒 AIrganizer - Permission Setup")
    print("=" * 50)
    
    # Create a temporary organizer to get directories
    try:
        organizer = BaseOrganizer()  # Use base organizer to avoid API key requirement
        target_dirs = organizer.target_dirs
    except Exception as e:
        print(f"Error: {e}")
        return
    
    handler = PermissionHandler()
    
    # Check permissions
    if handler.check_permissions(target_dirs):
        print("\n✅ Permission setup complete!")
        print("You can now run: airganizer --help")
    else:
        print("\n❌ Permission setup incomplete")
        print("Please follow the guide above and try again.")
        print(handler.get_permission_guide())


if __name__ == '__main__':
    main()
