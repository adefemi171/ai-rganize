

# AI-rganize 🧠📁

Una herramienta inteligente de organización de archivos que utiliza IA para categorizar y organizar tus archivos automáticamente. Funciona en **macOS**, **Linux (Ubuntu, etc.)** y **Windows**.

El nombre se inspiró en la música que estaba escuchando cuando se me ocurrió esta idea: la canción es [Organize](https://open.spotify.com/track/2wgvxtggKVzPkl0smF2UzI) de Asake

## Características

- 🤖 **Categorización con IA**: Utiliza IA (OpenAI, Claude, Gemini, Ollama, Mistral) para categorizar archivos de manera inteligente basada en el contenido y los metadatos
- 📂 **Soporte para múltiples directorios**: Organiza directorios comunes (Documentos, Escritorio, Descargas, etc.)
- 🎯 **Análisis de contenido**: Analiza PDFs, documentos de Word, imágenes, videos y archivos de audio para una categorización inteligente
- 📁 **Control de límite de carpetas**: Especifica el número máximo de carpetas a crear (p. ej., organizar 200 archivos en 6 carpetas)
- 🔒 **Multiplataforma**: Funciona en macOS, Linux (Ubuntu, etc.) y Windows
- 🔒 **Gestión de permisos**: Maneja correctamente los permisos de acceso a archivos en todas las plataformas
- 💾 **Sistema de respaldo**: Respaldos automáticos opcionales antes de mover archivos
- 🖥️ **Interfaz de terminal**: Interfaz de línea de comandos limpia y fácil de usar
- 🔍 **Modo de simulación (Dry Run)**: Vista previa del plan de organización antes de realizar cambios

## Instalación

### **Inicio rápido con uv (Recomendado)**

**¿Por qué uv?** uv ofrece una gestión de proyectos en Python rápida y confiable con creación automática de entornos virtuales, resolución de dependencias y gestión de archivos de bloqueo (lockfiles). Es significativamente más rápido que pip y se encarga de toda la complejidad de los entornos de Python por ti.

**Primero, instala uv y ffmpeg:**

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

**Alternativa (PyPI):**

```bash
pip install uv
# Then install ffmpeg separately for your platform
```

Para más opciones de instalación, consulta la [guía oficial de instalación de uv](https://docs.astral.sh/uv/getting-started/installation/).

**Luego, configura ai-rganize:**

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

**Qué hace uv por ti:**

- 🚀 **Creación automática de entornos virtuales** (directorio `.venv/`)
- 📦 **Resolución e instalación de dependencias** desde `pyproject.toml`
- 🔒 **Gestión de archivos de bloqueo** (`uv.lock`) para compilaciones reproducibles
- ⚡ **Instalación rápida de paquetes** (10-100 veces más rápido que pip)
- 🛠️ **Gestión de proyectos** con `uv sync`, `uv add`, `uv remove`
- 🐍 **Gestión de versiones de Python** con `uv python install`

## Uso

### Uso básico

**Comandos básicos:**

```bash
ai-rganize                                    # Organize all common directories
ai-rganize --dry-run                         # Dry run (preview without moving files)
ai-rganize --directory ~/Documents           # Organize specific directory
ai-rganize --max-file-size 5                 # Limit AI analysis to files smaller than 5MB
ai-rganize --ai-limit 20 --max-file-size 15  # Limit AI analysis to 20 files and files smaller than 15MB
```

**Alternativa: Usar uv run (no requiere activación):**

```bash
uv run ai-rganize --dry-run
uv run ai-rganize --directory ~/Documents
```

### Uso avanzado

**Selección de proveedor de LLM:**

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

**Límites de carpetas y control por lotes:**

```bash
# Organize 200 files into exactly 6 folders
ai-rganize --directory ~/Desktop/files --max-folders 6 --dry-run

# Control batch processing and costs
ai-rganize --batch-size 3 --max-cost 0.5 --dry-run
ai-rganize --ai-limit 100 --batch-size 10 --dry-run
```

**Gestión de proyectos con uv:**

```bash
uv sync                    # Install/update dependencies
uv lock --upgrade         # Update lock file
uv add package-name       # Add new dependency
uv remove package-name    # Remove dependency
uv tree                   # View dependency tree
```

### Opciones de línea de comandos

- `--api-key`: Clave API para el proveedor de LLM seleccionado (o establece la variable de entorno correspondiente)
- `--directory, -d`: Directorio específico a organizar
- `--dry-run`: Muestra el plan de organización sin mover archivos
- `--backup/--no-backup`: Crea un respaldo antes de organizar (predeterminado: no-backup)
- `--llm-provider`: Proveedor de LLM a usar - openai, claude, gemini, ollama, mistral (predeterminado: openai)
- `--model`: Nombre del modelo (los predeterminados varían según el proveedor)
- `--max-folders`: Número máximo de carpetas a crear (calcula automáticamente el tamaño del lote)
- `--batch-size`: Archivos por lote (se calcula automáticamente cuando se establece --max-folders)
- `--ai-limit`: Número máximo de archivos para analizar con IA (predeterminado: 50)
- `--max-file-size`: Tamaño máximo de archivo en MB para el análisis con IA (predeterminado: 10)
- `--max-cost`: Costo máximo en USD para el procesamiento con IA (predeterminado: 1.0)
- `--no-ai`: Desactiva la categorización con IA, usa solo basado en reglas
- `--verbose, -v`: Muestra información detallada del procesamiento

### Variables de entorno

Establece la clave API para tu proveedor elegido:

```bash
export OPENAI_API_KEY="your_api_key_here"        # For OpenAI
export ANTHROPIC_API_KEY="your_api_key_here"     # For Claude
export GEMINI_API_KEY="your_api_key_here"        # For Gemini
export MISTRAL_API_KEY="your_api_key_here"       # For Mistral
export OLLAMA_BASE_URL="http://localhost:11434"  # For Ollama (optional, default shown)
```

O crea un archivo `.env`:

```env
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
```

## Cómo funciona

1. **Escaneo de archivos**: Escanea los directorios objetivo en busca de archivos
2. **Análisis de contenido**: Analiza el contenido de los archivos utilizando IA:
   - 📄 **Documentos**: Extrae texto de PDFs y documentos de Word
   - 🖼️ **Imágenes**: Usa la API de Visión para analizar el contenido de las imágenes
   - 🎬 **Videos**: Extrae fotogramas y analiza el contenido del video
   - 🎵 **Audio**: Transcribe y analiza el contenido de audio
3. **Categorización con IA**: Utiliza el proveedor de LLM seleccionado para categorizar archivos de manera inteligente
4. **Organización**: Mueve los archivos a carpetas de categorías apropiadas basándose en el contenido y propósito
5. **Agrupación inteligente**: Detecta relaciones (familia, proyectos, temas) y los agrupa en consecuencia

## Características de seguridad

- ✅ **Respaldos automáticos**: Crea respaldos con marca de tiempo antes de cualquier cambio
- ✅ **Modo de simulación**: Vista previa de cambios antes de ejecutar
- ✅ **Verificación de permisos**: Verifica el acceso a los directorios
- ✅ **Manejo de duplicados**: Evita sobrescribir archivos existentes
- ✅ **Manejo de errores**: Manejo adecuado de errores de permisos
- ✅ **Registro (Logging)**: Registro completo de auditoría de todos los cambios

## Categorías de archivos

La herramienta organiza los archivos en estas categorías:

| Categoría | Tipos de archivo | Ejemplos |
|----------|------------|----------|
| Documentos | PDF, DOC, TXT, RTF | Informes, cartas, manuales |
| Imágenes | JPG, PNG, GIF, HEIC | Fotos, capturas de pantalla, gráficos |
| Videos | MP4, MOV, AVI, MKV | Películas, grabaciones, clips |
| Audio | MP3, WAV, FLAC, M4A | Música, podcasts, grabaciones |
| Archivos/Comprimidos | ZIP, RAR, 7Z, TAR | Archivos comprimidos, respaldos |
| Código | PY, JS, HTML, CSS | Archivos de programación, scripts |
| Hojas de cálculo | XLS, CSV, NUMBERS | Archivos de datos, informes |
| Presentaciones | PPT, KEY | Diapositivas, presentaciones |

## Desarrollo

### **Estructura del proyecto**

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

### **Desarrollo y contribuciones**

**Configuración para desarrollo:**

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

**Comandos de desarrollo:**

```bash
ai-rganize --help               # Test the tool
ai-rganize --dry-run           # Test with dry run
uv sync                          # Install/update dependencies
uv lock --upgrade                # Update lock file
```

**Cómo contribuir:**

1. Haz un fork del repositorio
2. Crea una rama para la característica (feature branch)
3. Realiza tus cambios
4. Prueba: `ai-rganize --dry-run`
5. Envía un pull request

## Solución de problemas

### Problemas de permisos

```bash
# Check permissions
ai-rganize-permissions

# If you get permission errors, try organizing specific directories:
ai-rganize --directory ~/Documents
```

### Problemas con la clave API

```bash
# Make sure your API key is set for your chosen provider:
export OPENAI_API_KEY="your_key_here"        # For OpenAI
export ANTHROPIC_API_KEY="your_key_here"     # For Claude
export GEMINI_API_KEY="your_key_here"        # For Gemini
# Or check your .env file
```

### Problemas con el análisis de video/audio

```bash
# If video/audio analysis fails, check ffmpeg:
ffmpeg -version

# Test with a small file first:
ai-rganize --directory ~/Desktop --ai-limit 1 --dry-run --verbose
```

### Archivos grandes

Todos los archivos se incluyen para la organización y el análisis con IA sin importar su tamaño. No hay restricciones de tamaño de archivo: todos recibirán un análisis completo de contenido con IA. La bandera `--max-file-size` está obsoleta y ya no restringe el análisis de archivos.

## Respaldo y recuperación

- Los respaldos se almacenan en `~/.ai-rganize_backup/`
- Cada respaldo tiene marca de tiempo
- Los registros de organización se almacenan en `~/.ai-rganize_log.json`

## Licencia

Licencia MIT - ¡siéntete libre de usarla y modificarla!

## Contribuciones

¡Agradecemos las contribuciones! No dudes en enviar issues y pull requests.

## Descargo de responsabilidad

Esta herramienta mueve y organiza tus archivos. Siempre:

- Prueba primero con `--dry-run`
- Mantén los respaldos habilitados
- Revisa el plan de organización antes de continuar
- Úsalo bajo tu propio riesgo

La categorización con IA se basa en el análisis del contenido de los archivos y puede no ser 100% precisa. Revisa siempre los resultados antes de proceder con una organización a gran escala.
