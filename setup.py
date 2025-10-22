#!/usr/bin/env python3
"""
Setup script for AIrganizer using uv
"""

import subprocess
import sys
import os
from pathlib import Path

def check_uv_installed():
    """Check if uv is installed."""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ uv is not installed")
    return False

def install_uv():
    """Install uv if not present."""
    print("📦 Installing uv...")
    try:
        # Install uv using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
        print("✅ uv installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install uv: {e}")
        print("\n💡 Alternative installation methods:")
        print("1. macOS: brew install uv")
        print("2. Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("3. Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        return False

def setup_project():
    """Set up the project using uv."""
    print("🚀 Setting up AIrganizer with uv...")
    
    try:
        # Sync dependencies (creates .venv and installs packages)
        subprocess.check_call(['uv', 'sync'])
        print("✅ Dependencies synced and virtual environment created")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set up project: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions to the user."""
    print("📝 Usage instructions:")
    print("  uv run airganizer --help")
    print("  uv run airganizer --dry-run")
    print("  uv run airganizer --directory ~/Documents")
    print("  uv sync                    # Install/update dependencies")
    print("  uv lock --upgrade         # Update lock file")

def main():
    """Main installation function."""
    print("🤖 AIrganizer - uv Setup")
    print("=" * 40)
    
    # Check if uv is installed
    if not check_uv_installed():
        if not install_uv():
            print("❌ Cannot proceed without uv")
            return False
    
    # Set up project
    if not setup_project():
        print("❌ Project setup failed")
        return False
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n✅ AIrganizer setup complete with uv!")
    print("\n📋 Next steps:")
    print("1. Set your OpenAI API key: export OPENAI_API_KEY='your_key_here'")
    print("2. Test with dry run: uv run airganizer --dry-run")
    print("3. Run AIrganizer: uv run airganizer")
    
    return True

if __name__ == "__main__":
    main()