# AI-rganize 🧠📁

An intelligent file organization tool that uses AI to automatically categorize and organize your files. Works on **macOS**, **Linux (Ubuntu, etc.)**, and **Windows**.

Name was inspired by the music I was listening to when this idea popped in my head, the music is [Organize](https://open.spotify.com/track/2wgvxtggKVzPkl0smF2UzI) by Asake

## Features

- 🤖 **AI-Powered Categorization**: Uses AI (OpenAI, Claude, Gemini, Ollama, Mistral) to intelligently categorize files based on content and metadata
- 📂 **Multiple Directory Support**: Organizes common directories (Documents, Desktop, Downloads, etc.)
- 🎯 **Content Analysis**: Analyzes PDFs, Word docs, images, videos, and audio files for intelligent categorization
- 📁 **Folder Limit Control**: Specify maximum number of folders to create (e.g., organize 200 files into 6 folders)
- 🔒 **Cross-Platform**: Works on macOS, Linux (Ubuntu, etc.), and Windows
- 🔒 **Permission Handling**: Properly handles file access permissions across platforms
- 💾 **Backup System**: Optional automatic backups before moving files
- 🖥️ **Terminal Interface**: Clean, user-friendly command-line interface
- 🔍 **Dry Run Mode**: Preview organization plan before making changes

## Installation

### **Quick Start with uv (Recommended)**

**Why uv?** uv provides fast, reliable Python project management with automatic virtual environment creation, dependency resolution, and lockfile management. It's significantly faster than pip and handles all the complexity of Python environments for you.

**First, install uv and ffmpeg:**

**macOS:**

```bash
brew install uv ffmpeg
```

**Linux:**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg
sudo apt update && sudo apt install ffmpeg  # Ubuntu/Debian
# or
sudo yum install ffmpeg  # CentOS/RHEL
```

**Windows:**

```powershell
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install ffmpeg
choco install ffmpeg
# or download from https://ffmpeg.org/download.html
```

**Alternative (PyPI):**

```bash
pip install uv
# Then install ffmpeg separately for your platform
```

For more installation options, see the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

**Then set up ai-rganize:**

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-rganize.git
cd ai-rganize

# 2. Create virtual environment with uv
uv venv

# 3. Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# 4. Install dependencies and package
uv pip install -r requirements.txt
uv pip install -e .

# 5. Set up API key (choose your provider)
export OPENAI_API_KEY="your_api_key_here"        # For OpenAI
# export ANTHROPIC_API_KEY="your_api_key_here"   # For Claude
# export GEMINI_API_KEY="your_api_key_here"      # For Gemini
# export MISTRAL_API_KEY="your_api_key_here"     # For Mistral

# 6. Test with dry run
ai-rganize --dry-run
```

**What uv does for you:**

- 🚀 **Automatic virtual environment creation** (`.venv/` directory)
- 📦 **Dependency resolution and installation** from `pyproject.toml`
- 🔒 **Lockfile management** (`uv.lock`) for reproducible builds
- ⚡ **Fast package installation** (10-100x faster than pip)
- 🛠️ **Project management** with `uv sync`, `uv add`, `uv remove`
- 🐍 **Python version management** with `uv python install`

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

### Advanced Usage

**LLM Provider Selection:**

```bash
# OpenAI (default)
ai-rganize --llm-provider openai --model gpt-4o --dry-run

# Claude (Latest: Claude Sonnet 4.5)
ai-rganize --llm-provider claude --dry-run

# Gemini (Latest: Gemini 2.5 Pro)
ai-rganize --llm-provider gemini --dry-run

# Ollama (Local models)
ai-rganize --llm-provider ollama --model llama3.1 --dry-run

# Mistral
ai-rganize --llm-provider mistral --dry-run
```

**Folder Limits and Batch Control:**

```bash
# Organize 200 files into exactly 6 folders
ai-rganize --directory ~/Desktop/files --max-folders 6 --dry-run

# Control batch processing and costs
ai-rganize --batch-size 3 --max-cost 0.5 --dry-run
ai-rganize --ai-limit 100 --batch-size 10 --dry-run
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

- `--api-key`: API key for selected LLM provider (or set corresponding env var)
- `--directory, -d`: Specific directory to organize
- `--dry-run`: Show organization plan without moving files
- `--backup/--no-backup`: Create backup before organizing (default: no-backup)
- `--llm-provider`: LLM provider to use - openai, claude, gemini, ollama, mistral (default: openai)
- `--model`: Model name (defaults vary by provider)
- `--max-folders`: Maximum number of folders to create (auto-calculates batch size)
- `--batch-size`: Files per batch (auto-calculated when --max-folders is set)
- `--ai-limit`: Maximum files to analyze with AI (default: 50)
- `--max-file-size`: Maximum file size in MB for AI analysis (default: 10)
- `--max-cost`: Maximum cost in USD for AI processing (default: 1.0)
- `--no-ai`: Disable AI categorization, use only rule-based
- `--verbose, -v`: Show detailed processing information

### Environment Variables

Set the API key for your chosen provider:

```bash
export OPENAI_API_KEY="your_api_key_here"        # For OpenAI
export ANTHROPIC_API_KEY="your_api_key_here"     # For Claude
export GEMINI_API_KEY="your_api_key_here"        # For Gemini
export MISTRAL_API_KEY="your_api_key_here"       # For Mistral
export OLLAMA_BASE_URL="http://localhost:11434"  # For Ollama (optional, default shown)
```

Or create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
```

## How It Works

1. **File Scanning**: Scans target directories for files
2. **Content Analysis**: Analyzes file content using AI:
   - 📄 **Documents**: Extracts text from PDFs and Word docs
   - 🖼️ **Images**: Uses Vision API to analyze image content
   - 🎬 **Videos**: Extracts frames and analyzes video content
   - 🎵 **Audio**: Transcribes and analyzes audio content
3. **AI Categorization**: Uses your chosen LLM provider to intelligently categorize files
4. **Organization**: Moves files to appropriate category folders based on content and purpose
5. **Smart Grouping**: Detects relationships (family, projects, themes) and groups accordingly

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
├── __init__.py
├── cli.py               # Command-line interface
├── core.py              # Main entry point
├── ai_client.py         # LLM provider clients
├── file_analysis.py     # File content analysis
├── organizer/           # Organization strategies
│   ├── base_organizer.py
│   ├── ai_organizer.py
│   └── rule_based_organizer.py
├── analyzers/           # Content analyzers
│   ├── document_analyzer.py
│   ├── image_analyzer.py
│   ├── video_analyzer.py
│   ├── audio_analyzer.py
│   └── text_analyzer.py
├── permissions/         # Permission handling
├── utils/               # Utilities
└── rate_limiting.py     # Rate limiting and cost tracking
```

### **Development & Contributing**

**Setup for development:**

```bash
# Clone and setup
git clone https://github.com/adefemi171/ai-rganize.git
cd ai-rganize
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
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
# Make sure your API key is set for your chosen provider:
export OPENAI_API_KEY="your_key_here"        # For OpenAI
export ANTHROPIC_API_KEY="your_key_here"     # For Claude
export GEMINI_API_KEY="your_key_here"        # For Gemini
# Or check your .env file
```

### Video/Audio Analysis Issues

```bash
# If video/audio analysis fails, check ffmpeg:
ffmpeg -version

# Test with a small file first:
ai-rganize --directory ~/Desktop --ai-limit 1 --dry-run --verbose
```

### Large Files

All files are included for organization and AI analysis regardless of size. There are no file size restrictions - all files will receive full AI content analysis. The `--max-file-size` flag is deprecated and no longer restricts file analysis.

## Backup and Recovery

- Backups are stored in `~/.ai-rganize_backup/`
- Each backup is timestamped
- Organization logs are stored in `~/.ai-rganize_log.json`

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
