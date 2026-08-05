---
name: airganize-safe-usage
description: Use this skill when the user wants to run, extend, or automate AI-rganize (the AI-powered file organizer in this repo). Covers safe dry-run-first workflows, profile/exclusion configuration, archive handling, and undo via the provenance ledger/manifest.
---

# AI-rganize: Safe Usage Skill

AI-rganize moves real files on a user's disk. This skill captures the safety
rules an agent (or the CLI) must follow, and points to the specific modules
that implement each capability.

## Golden rules (never violate these)

1. **Always dry-run first.** Before executing any organization plan, run with
   `--dry-run` (CLI) or build a `ReviewPlan` (see `ai_rganize/review/plan.py`)
   and show it to the user. Never move files without an explicit approval
   step.
2. **Never widen scope past what the user asked for.** Respect
   `ai_rganize/config/exclusions.py`'s `DEFAULT_PROTECTED_PATTERNS`
   (`.git/`, `node_modules/`, `.venv/`, `~/Library/`, the manifest file
   itself) plus anything in a `.airganizeignore` or profile's `exclusions`
   list. When in doubt, exclude rather than include.
3. **Never extract archives unsafely.** Only use
   `ai_rganize/features/archives.py::safe_extract`, which blocks zip-slip
   path traversal, absolute paths, symlink members, and enforces file-count /
   total-size limits. Never call `zipfile`/`tarfile` extraction APIs
   directly on untrusted archives.
4. **Preserve undo-ability.** Every real run should produce a manifest
   (`ai_rganize/utils/metadata.py`) and/or ledger records
   (`ai_rganize/provenance/ledger.py`) so the user can restore files later
   with `--restore <manifest_path>`.
5. **The GUI dashboard is localhost-only and read-mostly.** It binds to
   `127.0.0.1:8765` and its `/api/organize` endpoint defaults to
   `dry_run=true`; it does not execute real moves. Do not change it to bind
   to `0.0.0.0` or accept remote connections.
6. **Validate every destination path.** Use
   `ai_rganize/utils/safety.py` (`ensure_destination_safe`,
   `is_within_directory`, `unique_destination`, `validate_restore_path`,
   `sanitize_folder_name`) whenever computing where a file will land, whether
   that destination came from a rule, an LLM, or a natural-language policy.

## Typical safe workflow

1. Resolve configuration: `ai_rganize.config.resolve_effective_profile()`
   (checks an explicit profile name, then a project-local `.airganize.yaml`,
   then the global default profile).
2. Build an exclusion matcher:
   `ai_rganize.config.build_matcher_for_directory(directory, profile.exclusions)`
   and skip any file where `matcher.is_excluded(path, root)` is True.
3. Scan and categorize files (existing `RuleBasedOrganizer` / `AIOrganizer`,
   or `ai_rganize.council.LLMCouncil` for multi-provider consensus).
4. Wrap the resulting plan in a `ReviewPlan`
   (`ReviewPlan.from_organization_plan(plan, target_dir)`), inspect
   low-confidence or colliding items (`plan.low_confidence()`), and get
   explicit approval (`plan.approve_all()` / `plan.reject(source)` /
   `plan.edit_destination(source, new_dest)`).
5. Execute only the approved subset (`plan.to_execution_plan()`) through the
   existing `BaseOrganizer.execute_organization`, which already preserves
   metadata and writes a manifest.
6. Append provenance records for the run:
   `ai_rganize.provenance.append_records([...])`.
7. If the user wants to undo, preview first with
   `ai_rganize.provenance.undo_preview(run_id)` before calling the CLI's
   `--restore` flow.

## Feature module map

| Capability | Module |
|---|---|
| Profiles (`~/.ai_rganize/profiles/<name>.yaml`, project `.airganize.yaml`) | `ai_rganize/config/profile.py` |
| Exclusion / `.airganizeignore` matching | `ai_rganize/config/exclusions.py` |
| Pre-execution human review | `ai_rganize/review/plan.py` |
| Append-only run history / undo preview | `ai_rganize/provenance/ledger.py` |
| Multi-provider consensus categorization | `ai_rganize/council/council.py` |
| Exact duplicate detection (sha256) | `ai_rganize/features/duplicates.py` |
| Safe zip/tar inspection & extraction | `ai_rganize/features/archives.py` |
| Local cloud-sync folder discovery (no APIs) | `ai_rganize/features/cloud_sync.py` |
| Non-AI quick triage / clustering | `ai_rganize/features/triage.py` |
| Folder watching with debounce + quiet hours | `ai_rganize/watch/watcher.py` |
| Per-profile decision memory | `ai_rganize/memory/feedback.py` |
| Localhost dashboard | `ai_rganize/gui/app.py` |
| Natural-language policy heuristics | `ai_rganize/policies/nl_policy.py` |
| Path-safety primitives | `ai_rganize/utils/safety.py` |

## When extending AI-rganize

- Add new tests under `tests/` for any new path-handling or archive logic --
  these are the highest-risk areas (traversal, symlinks, clobbering).
- Keep new features dependency-optional where reasonable (e.g. `watchdog`,
  `flask`, `pyyaml` all have fallbacks or are only imported where used) so
  the core CLI keeps working in minimal environments.
- Never remove or weaken `DEFAULT_PROTECTED_PATTERNS` or the zip-slip checks
  in `safe_extract` without an explicit, well-reasoned user request.
