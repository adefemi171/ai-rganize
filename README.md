# AIrganizer 🧠📁

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
```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Clone the repository
git clone https://github.com/yourusername/airganizer.git
cd airganizer

# 3. Set up with uv
python setup.py

# 4. Set up API key
export OPENAI_API_KEY="your_api_key_here"

# 5. Test with dry run
uv run airganizer --dry-run
```

### **Alternative: pip installation**
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/airganizer.git
cd airganizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install in development mode
pip install -e .

# 4. Set up API key
export OPENAI_API_KEY="your_api_key_here"

# 5. Test with dry run
airganizer --dry-run
```

### **From PyPI (when published)**
```bash
pip install airganizer
```

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

**With uv (recommended):**
```bash
# Organize all common directories (Documents, Desktop, Downloads, etc.)
uv run airganizer

# Dry run (preview without moving files)
uv run airganizer --dry-run

# Organize specific directory
uv run airganizer --directory ~/Documents

# Skip backup (not recommended)
uv run airganizer --no-backup
```

**Direct uv commands:**
```bash
# Run AIrganizer with any options
uv run airganizer --help
uv run airganizer --dry-run
uv run airganizer --directory ~/Documents

# Install/update dependencies
uv sync
uv lock --upgrade
```

**With pip:**
```bash
# Organize all common directories (Documents, Desktop, Downloads, etc.)
airganizer

# Dry run (preview without moving files)
airganizer --dry-run

# Organize specific directory
airganizer --directory ~/Documents

# Skip backup (not recommended)
airganizer --no-backup
```

### Command Line Options

- `--api-key`: OpenAI API key (or set OPENAI_API_KEY env var)
- `--directory, -d`: Specific directory to organize
- `--dry-run`: Show organization plan without moving files
- `--backup/--no-backup`: Create backup before organizing (default: backup)
- `--ai-limit`: Maximum number of files to analyze with AI (default: 50)
- `--no-ai`: Disable AI categorization, use only rule-based

### Environment Variables

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Or create a `.env` file:
```
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
```
airganizer/
├── __init__.py          # Package initialization
├── core.py              # Main AIrganizer class
├── cli.py               # Command-line interface
├── permissions.py       # Permission handling
├── permissions_cli.py   # Permission setup CLI
└── utils.py             # Utility functions
```

### **Development Setup with uv**
```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Clone and setup
git clone https://github.com/yourusername/airganizer.git
cd airganizer
python setup.py

# 3. Development commands
uv run airganizer --help         # Run AIrganizer
uv sync                          # Install dependencies
uv lock --upgrade                # Update dependencies
```

### **Development Setup with pip**
```bash
# 1. Install in development mode
pip install -e .

# 2. Run AIrganizer
airganizer --help
```

### **Contributing**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes: `uv run airganizer --dry-run`
5. Submit a pull request

### **Available Commands**

**With uv:**
- `uv run airganizer` - Run AIrganizer
- `uv sync` - Install/update dependencies
- `uv lock --upgrade` - Update lock file

**With pip:**
- `airganizer` - Run AIrganizer

## Troubleshooting

### Permission Issues
```bash
# Check permissions
airganizer-permissions

# If you get permission errors, try organizing specific directories:
airganizer --directory ~/Documents
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

- Backups are stored in `~/.airganizer_backup/`
- Each backup is timestamped
- Organization logs are stored in `~/.airganizer_log.json`

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