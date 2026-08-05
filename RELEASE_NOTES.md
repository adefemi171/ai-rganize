# Release Notes — v1.0.0

## Security & Safety

- No-clobber moves (existing destinations get `_1`, `_2`, … suffixes)
- Destination paths confined under the target root
- Symlinks skipped during scan and execute
- `~/Library` removed from default scan targets; credential paths blocked
- Manifest restore validates paths under the source directory
- Empty-folder cleanup only deletes allowlisted junk (`.DS_Store`, etc.)
- Hard `--max-cost` stop (no soft overrun)
- Prompt injection guardrails around file-derived content
- Vision uploads capped by size
- OpenRouter client fixed (`import openai`, uses `_build_prompt`)
- Permission checks require read **and** write; sudo guidance removed

## Features

- **Profiles** (`~/.ai_rganize/profiles/`, project `.airganize.yaml`)
- **Exclusions** (`.airganizeignore` + defaults)
- **Review plans** before execute
- **Provenance ledger** (`~/.ai_rganize/ledger.jsonl`)
- **LLM Council** multi-provider majority vote (`council-organize` / `--council`)
- **Duplicates** exact-hash detection
- **Archives** inventory + zip-slip-safe opt-in unpack
- **Cloud sync discovery** for local Dropbox / iCloud / Google Drive folders
- **Watch mode** with debounce and quiet hours
- **Feedback memory** and NL policy heuristics
- **GUI dashboard** on `127.0.0.1:8765` (`ai-rganize-gui`)
- Expanded CLI group: `organize`, `restore`, `review`, `duplicates`, `archives`, `cloud`, `watch`, `profiles`, `ledger`, `explain`, `council-organize`

## CI / Contributor Gating

- GitHub Actions CI (Python 3.10–3.12, ruff, pytest)
- Security workflow (pip-audit + CodeQL)
- Dependabot for pip and Actions
- CODEOWNERS + PR / issue templates for first-time contributors

## Breaking / Behavior Changes

- Multi-directory organize now writes into **each** directory (no longer consolidates into the first)
- Dry-run destinations align with execute paths when `target_dir` is provided
- Legacy package-root `ai_organizer.py` removed (use `organizer.ai_organizer`)
- Version aligned to **1.0.0**

## Requirements

- Python 3.10+
- ffmpeg (video/audio analysis)
- Optional: `uv pip install -e ".[gui]"` for dashboard/watch/YAML extras (YAML is now a core dep)
- API key for your chosen LLM provider (env vars preferred over `--api-key`)
