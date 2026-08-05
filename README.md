# AI-rganize 🧠📁

An intelligent file organization tool that uses AI to automatically categorize and organize your files. Works on **macOS**, **Linux (Ubuntu, etc.)**, and **Windows**.

Name was inspired by the music I was listening to when this idea popped in my head, the music is [Organize](https://open.spotify.com/track/2wgvxtggKVzPkl0smF2UzI) by Asake

## Features (v1.0.0)

- 🤖 **AI categorization** via OpenAI, Claude, Gemini, Ollama, Mistral, OpenRouter
- 🗳️ **LLM Council** multi-provider majority vote (`council-organize` / `--council`)
- 📂 **Profiles, exclusions, review plans**, and a provenance ledger for undo/audit
- 🛡️ **Hardened moves**: no-clobber, symlink skip, path confinement, protected dirs blocked
- 📦 **Safe archive unpack** (zip/tar, zip-slip resistant) and exact-hash **duplicate** detection
- ☁️ **Cloud folder discovery** for local Dropbox / iCloud / Google Drive mounts
- 👀 **Watch mode**, feedback memory, and a localhost **GUI dashboard** (`ai-rganize-gui`)
- 🔍 **Dry-run** + optional backup + manifest restore
- See [RELEASE_NOTES.md](RELEASE_NOTES.md) and [CONTRIBUTING.md](CONTRIBUTING.md)

## Installation

You need **ffmpeg** for video/audio analysis (`brew install ffmpeg` / `apt install ffmpeg`).

### 1. PyPI (best for most users)

> Available after the first `v*` GitHub Release + PyPI publish (see [docs/PUBLISHING.md](docs/PUBLISHING.md)).

```bash
# Recommended: isolated CLI tool install
uv tool install ai_rganize

# Or classic pip / pipx
pip install ai_rganize
pipx install ai_rganize
```

Upgrade later: `uv tool upgrade ai_rganize`

### 2. GitHub + uv / pipx (works today, before PyPI)

```bash
# Latest main
uv tool install "git+https://github.com/adefemi171/airganizer.git"

# Pin to a release tag
uv tool install "ai_rganize @ git+https://github.com/adefemi171/airganizer.git@v1.0.0"

# pipx alternative
pipx install "git+https://github.com/adefemi171/airganizer.git@v1.0.0"
```

### 3. Homebrew (macOS)

```bash
brew tap adefemi171/airganize
brew install ai-rganize
```

Tap repo: [`homebrew-airganize`](https://github.com/adefemi171/homebrew-airganize) (create once; formula source lives in this repo at [`homebrew/ai-rganize.rb`](homebrew/ai-rganize.rb)).

### Development install (contributors)

```bash
git clone https://github.com/adefemi171/airganizer.git
cd airganizer
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,gui]"
```

### API keys

```bash
export OPENAI_API_KEY="your_api_key_here"
# export ANTHROPIC_API_KEY="..."
# export GEMINI_API_KEY="..."
# export MISTRAL_API_KEY="..."

ai-rganize organize --dry-run -d ~/Downloads
```

Publishing maintainers: see [docs/PUBLISHING.md](docs/PUBLISHING.md).

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
