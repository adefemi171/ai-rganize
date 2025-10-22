# AI-rganize 🧠📁

An intelligent file organization tool that uses AI to automatically categorize and organize your files. Works on **macOS**, **Linux (Ubuntu, etc.)**, and **Windows**.

## Features

- 🤖 **AI-Powered Categorization**: Uses OpenAI's GPT to intelligently categorize files based on content and metadata
- 📂 **Multiple Directory Support**: Organizes common directories (Documents, Desktop, Downloads, etc.)
- 🔒 **Cross-Platform**: Works on macOS, Linux (Ubuntu, etc.), and Windows
- 🔒 **Permission Handling**: Properly handles file access permissions across platforms
- 💾 **Backup System**: Creates automatic backups before moving files
- 🎯 **Smart Organization**: Groups files into logical categories (documents, images, videos, audio, etc.)
- 🖥️ **Terminal Interface**: Clean, user-friendly command-line interface
- 🔍 **Dry Run Mode**: Preview organization plan before making changes

## Installation

### **Quick Start with uv (Recommended)**

**Why uv?** uv provides fast, reliable Python project management with automatic virtual environment creation, dependency resolution, and lockfile management. It's significantly faster than pip and handles all the complexity of Python environments for you.

**First, install uv:**

**macOS:**
```bash
brew install uv
```

**Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (PyPI):**

```bash
pip install uv
```

For more installation options, see the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

**Then set up ai-rganize:**

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-rganize.git
cd ai-rganize

# 2. Create virtual environment with uv (automatically activates)
uv venv

# 3. Install dependencies and package
uv pip install -r requirements.txt
uv pip install -e .

# 4. Set up API key
export OPENAI_API_KEY="your_api_key_here"

# 5. Test with dry run
ai-rganize --dry-run
```

**What uv does for you:**

- 🚀 **Automatic virtual environment creation** (`.venv/` directory)
- 📦 **Dependency resolution and installation** from `pyproject.toml`
- 🔒 **Lockfile management** (`uv.lock`) for reproducible builds
- ⚡ **Fast package installation** (10-100x faster than pip)
- 🛠️ **Project management** with `uv sync`, `uv add`, `uv remove`
- 🐍 **Python version management** with `uv python install`

### **Manual Installation**

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Get OpenAI API Key**:
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a new API key
   - Add it to your environment or `.env` file

3. **Set up permissions**:
   - **macOS**: System Preferences > Security & Privacy > Privacy
   - **Linux**: Check file ownership and permissions
   - **Windows**: Run as Administrator if needed

## Usage

### Basic Usage

**Basic commands:**

```bash
ai-rganize                                    # Organize all common directories
ai-rganize --dry-run                         # Dry run (preview without moving files)
ai-rganize --directory ~/Documents           # Organize specific directory
ai-rganize --max-file-size 5                 # Limit AI analysis to files smaller than 5MB
ai-rganize --ai-limit 20 --max-file-size 15  # Limit AI analysis to 20 files and files smaller than 15MB
```

**Alternative: Using uv run (no activation needed):**

```bash
uv run ai-rganize --dry-run
uv run ai-rganize --directory ~/Documents
```

**Project management with uv:**

```bash
uv sync                    # Install/update dependencies
uv lock --upgrade         # Update lock file
uv add package-name       # Add new dependency
uv remove package-name    # Remove dependency
uv tree                   # View dependency tree
```

### Command Line Options

- `--api-key`: OpenAI API key (or set OPENAI_API_KEY env var)
- `--directory, -d`: Specific directory to organize
- `--dry-run`: Show organization plan without moving files
- `--backup/--no-backup`: Create backup before organizing (default: backup)
- `--ai-limit`: Maximum number of files to analyze with AI (default: 50)
- `--max-file-size`: Maximum file size in MB for AI analysis (default: 10)
- `--no-ai`: Disable AI categorization, use only rule-based

### Environment Variables

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Or create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

## How It Works

1. **File Scanning**: Scans target directories for files
2. **AI Analysis**: Uses OpenAI GPT to analyze file content and metadata
3. **Categorization**: Groups files into logical categories:
   - 📄 Documents (PDFs, Word docs, text files)
   - 🖼️ Images (photos, graphics, screenshots)
   - 🎬 Videos (movie files, recordings)
   - 🎵 Audio (music, podcasts, recordings)
   - 📦 Archives (zip files, compressed files)
   - 💻 Code (programming files)
   - 📊 Spreadsheets (Excel, CSV files)
   - 📽️ Presentations (PowerPoint, Keynote)
   - 📁 Other (miscellaneous files)

4. **Organization**: Moves files to appropriate category folders
5. **Logging**: Records all changes for reference

## Safety Features

- ✅ **Automatic Backups**: Creates timestamped backups before any changes
- ✅ **Dry Run Mode**: Preview changes before executing
- ✅ **Permission Checks**: Verifies access to directories
- ✅ **Duplicate Handling**: Prevents overwriting existing files
- ✅ **Error Handling**: Graceful handling of permission errors
- ✅ **Logging**: Complete audit trail of all changes

## File Categories

The tool organizes files into these categories:

| Category | File Types | Examples |
|----------|------------|----------|
| Documents | PDF, DOC, TXT, RTF | Reports, letters, manuals |
| Images | JPG, PNG, GIF, HEIC | Photos, screenshots, graphics |
| Videos | MP4, MOV, AVI, MKV | Movies, recordings, clips |
| Audio | MP3, WAV, FLAC, M4A | Music, podcasts, recordings |
| Archives | ZIP, RAR, 7Z, TAR | Compressed files, backups |
| Code | PY, JS, HTML, CSS | Programming files, scripts |
| Spreadsheets | XLS, CSV, NUMBERS | Data files, reports |
| Presentations | PPT, KEY | Slides, presentations |

## Development

### **Project Structure**

```text
ai-rganize/
├── __init__.py          # Package initialization
├── core.py              # Main ai-rganize class
├── cli.py               # Command-line interface
├── permissions.py       # Permission handling
├── permissions_cli.py   # Permission setup CLI
└── utils.py             # Utility functions
```

### **Development & Contributing**

**Setup for development:**

```bash
# Clone and setup
git clone https://github.com/yourusername/ai-rganize.git
cd ai-rganize
uv venv
uv pip install -r requirements.txt
uv pip install -e .
```

**Development commands:**

```bash
ai-rganize --help               # Test the tool
ai-rganize --dry-run           # Test with dry run
uv sync                          # Install/update dependencies
uv lock --upgrade                # Update lock file
```

**Contributing:**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test: `ai-rganize --dry-run`
5. Submit a pull request

## Troubleshooting

### Permission Issues

```bash
# Check permissions
ai-rganize-permissions

# If you get permission errors, try organizing specific directories:
ai-rganize --directory ~/Documents
```

### API Key Issues

```bash
# Make sure your API key is set:
export OPENAI_API_KEY="your_key_here"
# Or check your .env file
```

### Large Files

The tool automatically skips files larger than 10MB for AI analysis to avoid high API costs. These files are categorized using rule-based methods.

## Backup and Recovery

- Backups are stored in `~/.ai-rganize_backup/`
- Each backup is timestamped
- Organization logs are stored in `~/.ai-rganize_log.json`

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection (for AI analysis)

## Dependencies

- `openai`: AI categorization
- `click`: Command-line interface
- `rich`: Beautiful terminal output
- `Pillow`: Image file analysis

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## Disclaimer

This tool moves and organizes your files. Always:

- Test with `--dry-run` first
- Keep backups enabled
- Review the organization plan before proceeding
- Use at your own risk

The AI categorization is based on file content analysis and may not be 100% accurate. Always review the results before proceeding with large-scale organization.