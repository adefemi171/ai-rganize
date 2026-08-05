# Contributing to AI-rganize

Thanks for helping improve a tool that moves real user files. Please treat filesystem safety as the top priority.

## Before you open a PR

1. Create a branch from `main`
2. Install: `uv venv && uv pip install -e ".[dev,gui]"`
3. Run locally: `ruff check ai_rganize tests && pytest -q`
4. Prefer `--dry-run` in any manual testing of organize flows

## First-time contributors

- A maintainer must review and approve before merge
- PRs that touch move/restore/archive/subprocess code get extra scrutiny
- Do not request merge until CI is green

## High-risk areas (require careful review)

- `ai_rganize/cli.py`
- `ai_rganize/utils/metadata.py`
- `ai_rganize/utils/safety.py`
- `ai_rganize/organizer/`
- `ai_rganize/features/archives.py`
- `ai_rganize/analyzers/` (subprocess / media parsers)

## Maintainer setup (GitHub)

Enable branch protection on `main`:

- Require status checks: CI test job (all Python versions)
- Require 1 approving review / CODEOWNERS
- Disallow force pushes
- Optionally require conversation resolution

## Commit style

Keep commits focused. Prefer small PRs that separate safety fixes from feature work when practical.

## Releases / install channels

Maintainers: see [docs/PUBLISHING.md](docs/PUBLISHING.md) for PyPI Trusted Publishing, GitHub Releases, and Homebrew tap updates.
