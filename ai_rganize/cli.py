"""CLI for AI-rganize."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from .config.exclusions import ExclusionMatcher, build_matcher_for_directory, load_ignore_file
from .config.profile import (
    Profile,
    default_profile,
    list_profiles,
    load_profile,
    resolve_effective_profile,
    save_profile,
)
from .council.council import LLMCouncil
from .features.archives import organize_archives_in_dir
from .features.cloud_sync import discover_cloud_roots, known_providers
from .features.duplicates import find_duplicate_groups
from .features.triage import explain_file
from .memory.feedback import record_decision
from .organizers import AIOrganizer, RuleBasedOrganizer
from .provenance.ledger import (
    LedgerRecord,
    append_records,
    compute_file_hash,
    list_run_ids,
    query,
    undo_preview,
)
from .review.plan import ReviewPlan
from .utils.metadata import (
    cleanup_empty_folders,
    create_manifest,
    load_manifest,
    move_preserving_metadata,
    restore_from_manifest,
    save_manifest,
)
from .utils.safety import (
    ensure_destination_safe,
    is_protected_path,
    is_symlink_or_through_symlink,
    sanitize_folder_name,
    unique_destination,
    validate_restore_path,
)
from .watch.watcher import OrganizationWatcher

console = Console()

ORGANIZE_DEFAULTS: dict[str, Any] = {
    "api_key": None,
    "directory": None,
    "dry_run": False,
    "backup": False,
    "ai_limit": 50,
    "max_file_size": None,
    "batch_size": 5,
    "max_cost": 1.0,
    "max_folders": None,
    "llm_provider": "openai",
    "model": None,
    "no_ai": False,
    "summary_only": False,
    "save_manifest": True,
    "verbose": False,
    "profile": None,
    "ignore_file": None,
    "review": False,
    "council": True,
    "council_providers": None,
    "unpack_archives": False,
    "allow_protected": False,
}

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "claude": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.5-pro-exp",
    "ollama": "llama3.1",
    "mistral": "mistral-large-latest",
    "openrouter": "anthropic/claude-3.5-sonnet",
}


def _resolve_model(llm_provider: str, model: Optional[str]) -> str:
    if model:
        return model
    return DEFAULT_MODELS.get(llm_provider.lower(), "gpt-4o")


def _build_organizer(
    no_ai: bool,
    api_key: Optional[str],
    max_file_size: Optional[int],
    batch_size: int,
    max_cost: float,
    model: str,
    llm_provider: str,
    max_folders: Optional[int],
):
    max_file_size_mb = max_file_size if max_file_size is not None else 10000
    if no_ai:
        return RuleBasedOrganizer(max_file_size_mb=max_file_size_mb)
    return AIOrganizer(
        api_key,
        max_file_size_mb=max_file_size_mb,
        batch_size=batch_size,
        max_cost=max_cost,
        model=model,
        llm_provider=llm_provider,
        max_folders=max_folders,
    )


def _build_exclusion_matcher(
    root: Path,
    profile: Optional[Profile],
    ignore_file: Optional[Path],
) -> ExclusionMatcher:
    extra_patterns: list[str] = []
    if profile and profile.exclusions:
        extra_patterns.extend(profile.exclusions)
    if ignore_file:
        extra_patterns.extend(load_ignore_file(ignore_file))
    return build_matcher_for_directory(root, extra_patterns=extra_patterns or None)


def _filter_scanned_files(
    files: list[dict[str, Any]],
    root: Path,
    matcher: ExclusionMatcher,
    allow_protected: bool,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for file_info in files:
        path = Path(file_info["path"])
        if is_symlink_or_through_symlink(path):
            continue
        if not allow_protected and is_protected_path(path):
            continue
        if matcher.is_excluded(path, root):
            continue
        kept.append(file_info)
    return kept


def _categorize_with_council(
    files: list[dict[str, Any]],
    council: LLMCouncil,
    batch_size: int,
    max_folders: Optional[int],
    verbose: bool,
) -> dict[str, list[dict[str, Any]]]:
    plan: dict[str, list[dict[str, Any]]] = {}
    for index in range(0, len(files), batch_size):
        batch = files[index : index + batch_size]
        prepared = [
            {
                "name": item["name"],
                "path": str(item["path"]),
                "size": item.get("size", 0),
            }
            for item in batch
        ]
        folder_names, confidences = council.categorize(
            prepared,
            verbose=verbose,
            max_folders=max_folders,
        )
        if not folder_names:
            continue
        for file_info, folder_name, confidence in zip(batch, folder_names, confidences):
            folder = sanitize_folder_name(folder_name)
            enriched = dict(file_info)
            enriched["confidence"] = confidence
            plan.setdefault(folder, []).append(enriched)
    return plan


def _display_plan_for_target(plan: dict[str, Any], target_dir: Path, summary_only: bool) -> None:
    review = ReviewPlan.from_organization_plan(plan, target_dir)
    if summary_only:
        summary = plan.get("summary", {})
        console.print(
            f"\n📊 [bold]Summary:[/bold] {summary.get('total_files', len(review.items))} files "
            f"into {summary.get('total_folders', len({i.folder for i in review.items}))} folders"
        )
        return
    review.display()


def _append_ledger_from_manifest(manifest, run_id: str) -> int:
    records = []
    for move in manifest.moves:
        source = Path(move.original)
        records.append(
            LedgerRecord(
                run_id=run_id,
                source=move.original,
                dest=move.destination,
                folder=move.category,
                provider=manifest.ai_provider,
                model=manifest.model,
                file_hash=compute_file_hash(source) if source.exists() else None,
            )
        )
    return append_records(records)


def _execute_plan_moves(
    plan: dict[str, Any],
    target_dir: Path,
    save_manifest_file: bool,
    ai_provider: Optional[str],
    model: Optional[str],
    verbose: bool,
) -> tuple[bool, str]:
    run_id = str(uuid.uuid4())
    manifest = create_manifest(target_dir, ai_provider, model)
    files_moved = 0

    for folder_name, files in plan.items():
        if folder_name == "summary" or not files:
            continue

        safe_folder = sanitize_folder_name(folder_name)
        dest_folder = target_dir / safe_folder
        dest_folder.mkdir(parents=True, exist_ok=True)

        for file_info in files:
            source = Path(file_info["path"])
            if not source.exists():
                if verbose:
                    console.print(f"[yellow]⚠️  File not found: {source}[/yellow]")
                continue

            if file_info.get("destination_override"):
                dest = Path(file_info["destination_override"])
            else:
                dest = dest_folder / source.name

            try:
                dest = ensure_destination_safe(dest, target_dir)
                dest = unique_destination(dest)
                if source.resolve() == dest.resolve():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                metadata = move_preserving_metadata(source, dest)
                manifest.add_move(source, dest, safe_folder, metadata)
                files_moved += 1
                if verbose:
                    console.print(f"[green]✅ Moved: {source.name} → {safe_folder}/{dest.name}[/green]")
            except Exception as exc:
                console.print(f"[red]❌ Failed to move {source.name}: {exc}[/red]")

    if files_moved > 0:
        if save_manifest_file:
            manifest_path = save_manifest(manifest, target_dir)
            if verbose:
                console.print(f"[blue]📋 Manifest saved: {manifest_path}[/blue]")
        _append_ledger_from_manifest(manifest, run_id)

    return files_moved > 0, run_id


def _create_plan_for_directory(
    organizer,
    files: list[dict[str, Any]],
    no_ai: bool,
    ai_limit: int,
    use_council: bool,
    council: Optional[LLMCouncil],
    batch_size: int,
    max_folders: Optional[int],
    verbose: bool,
) -> dict[str, Any]:
    if use_council:
        if council is None:
            raise click.ClickException("--council requires at least one council provider")
        return _categorize_with_council(files, council, batch_size, max_folders, verbose)
    if no_ai:
        return organizer.create_organization_plan(files, verbose=verbose)
    return organizer.create_organization_plan(files, ai_limit, verbose=verbose)


def _organize_single_directory(
    target_dir: Path,
    organizer,
    *,
    no_ai: bool,
    ai_limit: int,
    dry_run: bool,
    backup: bool,
    summary_only: bool,
    save_manifest: bool,
    review: bool,
    use_council: bool,
    council: Optional[LLMCouncil],
    batch_size: int,
    max_folders: Optional[int],
    llm_provider: str,
    model: str,
    verbose: bool,
    profile_name: Optional[str],
    matcher: ExclusionMatcher,
    allow_protected: bool,
    unpack_archives: bool,
) -> bool:
    if unpack_archives:
        result = organize_archives_in_dir(target_dir, unpack=True)
        unpacked = sum(1 for item in result.get("archives", []) if item.get("unpacked"))
        if unpacked and verbose:
            console.print(f"[blue]Unpacked {unpacked} archive(s) in {target_dir}[/blue]")

    console.print(f"[blue]Scanning {target_dir}...[/blue]")
    scanned = organizer.scan_files(target_dir)
    files = _filter_scanned_files(scanned, target_dir, matcher, allow_protected)
    console.print(f"[green]Found {len(files)} files in {target_dir.name}[/green]")

    if not files:
        console.print("[yellow]No files found to organize.[/yellow]")
        return True

    if use_council:
        console.print("[blue]Using multi-provider council categorization[/blue]")
    elif no_ai:
        console.print("[yellow]Using rule-based categorization[/yellow]")
    else:
        console.print("[blue]Using AI-powered categorization[/blue]")

    plan = _create_plan_for_directory(
        organizer,
        files,
        no_ai,
        ai_limit,
        use_council,
        council,
        batch_size,
        max_folders,
        verbose,
    )

    plan_folders = [key for key in plan if key != "summary"]
    if not plan_folders:
        console.print("[red]Error: No organization plan was created.[/red]")
        if not verbose:
            console.print("[yellow]Tip: Try running with --verbose[/yellow]")
        return False

    _display_plan_for_target(plan, target_dir, summary_only)

    if dry_run:
        console.print("[yellow]Dry run complete. No files were moved.[/yellow]")
        return True

    execution_plan = plan
    if review:
        review_plan = ReviewPlan.from_organization_plan(plan, target_dir)
        review_plan.display()
        if not Confirm.ask("Approve all moves?"):
            console.print("[yellow]Organization cancelled.[/yellow]")
            return False
        review_plan.approve_all()
        execution_plan = review_plan.to_execution_plan()
        if profile_name:
            for item in review_plan.approved():
                record_decision(profile_name, item.source.name, item.folder, "approved")
    elif not Confirm.ask("Do you want to proceed with the organization?"):
        console.print("[yellow]Organization cancelled.[/yellow]")
        return False

    if backup:
        if not organizer.create_backup(files):
            if not Confirm.ask("Backup failed. Continue without backup?"):
                console.print("[yellow]Organization cancelled.[/yellow]")
                return False

    console.print("[blue]Organizing files...[/blue]")
    success, _run_id = _execute_plan_moves(
        execution_plan,
        target_dir,
        save_manifest,
        None if no_ai else llm_provider,
        None if no_ai else model,
        verbose,
    )

    if success:
        console.print("[green]File organization completed successfully![/green]")
        if save_manifest:
            console.print("[dim]💡 Tip: Use `ai-rganize restore` to undo this organization[/dim]")
    else:
        console.print("[red]File organization failed.[/red]")
    return success


def _run_organize(**kwargs: Any) -> None:
    profile = resolve_effective_profile(kwargs.get("profile"))
    llm_provider = kwargs.get("llm_provider") or profile.provider
    model = _resolve_model(llm_provider, kwargs.get("model") or profile.model)
    max_cost = kwargs.get("max_cost") if kwargs.get("max_cost") is not None else profile.max_cost
    max_folders = kwargs.get("max_folders") if kwargs.get("max_folders") is not None else profile.max_folders
    # Click always passes booleans for --council/--no-council.
    # Do not OR with profile defaults (False or True would ignore --no-council).
    use_council = bool(kwargs.get("council"))
    # Unpack only when explicitly requested on CLI, or via a named profile.
    unpack_archives = bool(kwargs.get("unpack_archives"))
    if kwargs.get("profile") and profile.auto_unpack_archives:
        unpack_archives = True

    if kwargs.get("council_providers"):
        council_providers = [part.strip() for part in kwargs["council_providers"].split(",") if part.strip()]
    else:
        # Default council panel: primary provider plus common alternatives.
        # Unavailable providers are skipped gracefully by LLMCouncil.
        default_panel = [llm_provider, "openai", "claude", "ollama"]
        seen: set[str] = set()
        council_providers = []
        for name in default_panel:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                council_providers.append(key)

    council = None
    if use_council:
        council = LLMCouncil(council_providers, api_key=kwargs.get("api_key"))

    try:
        organizer = _build_organizer(
            kwargs["no_ai"],
            kwargs.get("api_key"),
            kwargs.get("max_file_size"),
            kwargs.get("batch_size", 5),
            max_cost,
            model,
            llm_provider,
            max_folders,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not organizer.check_permissions():
        raise click.ClickException(
            "Cannot proceed without proper permissions. "
            "Run: ai-rganize-permissions for setup help"
        )

    if kwargs.get("directory"):
        target_dirs = [Path(kwargs["directory"]).expanduser().resolve()]
    elif profile.roots:
        target_dirs = [Path(root).expanduser().resolve() for root in profile.roots]
    else:
        target_dirs = [path for path in organizer.target_dirs.values() if path and path.exists()]

    if not target_dirs:
        raise click.ClickException("No accessible directories found.")

    console.print(
        Panel.fit(
            "[bold blue]AI-rganize[/bold blue]\n"
            "Intelligently organize your files using AI",
            border_style="blue",
        )
    )

    if kwargs.get("verbose") and not kwargs.get("model"):
        console.print(f"[blue]Using default model for {llm_provider}: {model}[/blue]")

    overall_success = True
    for target_dir in target_dirs:
        ignore_path = Path(kwargs["ignore_file"]).expanduser() if kwargs.get("ignore_file") else None
        matcher = _build_exclusion_matcher(target_dir, profile, ignore_path)
        success = _organize_single_directory(
            target_dir,
            organizer,
            no_ai=kwargs["no_ai"],
            ai_limit=kwargs.get("ai_limit", 50),
            dry_run=kwargs.get("dry_run", False),
            backup=kwargs.get("backup", False),
            summary_only=kwargs.get("summary_only", False),
            save_manifest=kwargs.get("save_manifest", True),
            review=kwargs.get("review", False),
            use_council=use_council,
            council=council,
            batch_size=kwargs.get("batch_size", 5),
            max_folders=max_folders,
            llm_provider=llm_provider,
            model=model,
            verbose=kwargs.get("verbose", False),
            profile_name=kwargs.get("profile") or profile.name,
            matcher=matcher,
            allow_protected=kwargs.get("allow_protected", False),
            unpack_archives=unpack_archives,
        )
        overall_success = overall_success and success

    if not overall_success:
        raise SystemExit(1)


def _run_restore(manifest_path: Path, verbose: bool) -> None:
    console.print(
        Panel.fit(
            "[bold blue]AI-rganize - Restore[/bold blue]\n"
            "Restoring files from manifest",
            border_style="blue",
        )
    )

    manifest = load_manifest(manifest_path)
    allowed_root = Path(manifest.source_directory).expanduser().resolve()

    console.print(f"[blue]Manifest created: {manifest.created}[/blue]")
    console.print(f"[blue]Source directory: {manifest.source_directory}[/blue]")
    console.print(f"[blue]Files to restore: {len(manifest.moves)}[/blue]")

    for move in manifest.moves:
        validate_restore_path(Path(move.original), allowed_root)

    if not Confirm.ask("Do you want to restore these files to their original locations?"):
        console.print("[yellow]Restore cancelled.[/yellow]")
        return

    console.print("[blue]Restoring files...[/blue]")
    successful, failed = restore_from_manifest(manifest, verbose=verbose)

    if successful > 0:
        console.print(f"[green]✅ Successfully restored {successful} files[/green]")
        removed = cleanup_empty_folders(allowed_root, verbose=verbose)
        if removed > 0:
            console.print(f"[green]🗑️  Removed {removed} empty folders[/green]")

    if failed > 0:
        console.print(f"[red]❌ Failed to restore {failed} files[/red]")


def organize_options(func):
    """Shared options for organize flows."""
    options = [
        click.option(
            "--api-key",
            help="[deprecated] Prefer provider env vars (e.g. OPENAI_API_KEY). "
            "API key for the selected LLM provider.",
        ),
        click.option("--directory", "-d", help="Specific directory to organize (default: common dirs)"),
        click.option("--dry-run", is_flag=True, help="Show organization plan without moving files"),
        click.option("--backup/--no-backup", default=False, help="Create backup before organizing"),
        click.option("--ai-limit", default=50, help="Maximum files to analyze with AI"),
        click.option(
            "--max-file-size",
            default=None,
            type=int,
            help="Maximum file size in MB for content analysis API calls",
        ),
        click.option("--batch-size", default=5, help="Files per AI batch"),
        click.option("--max-cost", default=1.0, help="Maximum AI cost in USD"),
        click.option("--max-folders", type=int, help="Maximum folders to create"),
        click.option(
            "--llm-provider",
            default="openai",
            type=click.Choice(
                ["openai", "claude", "gemini", "ollama", "mistral", "openrouter"],
                case_sensitive=False,
            ),
            help="LLM provider for AI categorization",
        ),
        click.option("--model", help="Model name (provider-specific default if omitted)"),
        click.option("--no-ai", is_flag=True, help="Disable AI categorization"),
        click.option("--summary-only", is_flag=True, help="Show only summary in dry run"),
        click.option(
            "--save-manifest/--no-manifest",
            default=True,
            help="Save organization manifest for undo capability",
        ),
        click.option("--verbose", "-v", is_flag=True, help="Show detailed processing information"),
        click.option("--profile", help="Named profile from ~/.ai_rganize/profiles/"),
        click.option("--ignore-file", type=click.Path(exists=True), help="Additional ignore patterns file"),
        click.option(
            "--review/--no-review",
            default=False,
            help="Review and approve moves before executing (recommended for live runs)",
        ),
        click.option(
            "--council/--no-council",
            default=True,
            help="Use multi-provider LLM council (default: on; use --no-council for single provider)",
        ),
        click.option(
            "--council-providers",
            help="Comma-separated council providers (default: primary llm-provider)",
        ),
        click.option(
            "--unpack-archives",
            is_flag=True,
            help="Safely unpack supported archives before organizing",
        ),
        click.option(
            "--allow-protected",
            is_flag=True,
            default=False,
            help="Allow scanning protected system/credential paths",
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
@organize_options
@click.option(
    "--restore",
    type=click.Path(exists=True),
    help="Restore files from a manifest file (legacy; prefer `ai-rganize restore`)",
)
def cli(ctx, restore, **kwargs):
    """AI-rganize - Intelligently organize your files using AI."""
    if restore:
        _run_restore(Path(restore), kwargs.get("verbose", False))
        return
    if ctx.invoked_subcommand is None:
        ctx.invoke(organize, **kwargs)


@cli.command()
@organize_options
def organize(**kwargs):
    """Organize files using AI or rule-based categorization."""
    _run_organize(**kwargs)


@cli.command()
@click.argument("manifest", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed restore output")
def restore(manifest, verbose):
    """Restore files from an organization manifest."""
    _run_restore(Path(manifest), verbose)


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@organize_options
def review(directory, **kwargs):
    """Build and interactively approve an organization review plan."""
    kwargs["directory"] = directory
    kwargs["review"] = True
    kwargs["dry_run"] = False
    _run_organize(**kwargs)


@cli.command("council-organize")
@organize_options
def council_organize(**kwargs):
    """Organize files using a multi-provider LLM council."""
    kwargs["council"] = True
    _run_organize(**kwargs)


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--verbose", "-v", is_flag=True)
def duplicates(directory, verbose):
    """Find exact duplicate files in a directory."""
    root = Path(directory).expanduser().resolve()
    files = [path for path in root.rglob("*") if path.is_file() and not is_symlink_or_through_symlink(path)]
    groups = find_duplicate_groups(files)

    if not groups:
        console.print("[green]No duplicate files found.[/green]")
        return

    table = Table(title=f"Duplicate groups in {root}")
    table.add_column("Hash", overflow="fold")
    table.add_column("Count", justify="right")
    table.add_column("Wasted MB", justify="right")
    table.add_column("Paths")

    for group in groups:
        paths = group["paths"]
        path_lines = "\n".join(str(path) for path in paths[:5])
        if len(paths) > 5:
            path_lines += f"\n... and {len(paths) - 5} more"
        table.add_row(
            str(group["hash"])[:16] + "...",
            str(group["count"]),
            f"{group['wasted_bytes'] / (1024 * 1024):.2f}",
            path_lines,
        )
        if verbose:
            for path in paths:
                console.print(f"  {path}")

    console.print(table)


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--unpack", is_flag=True, help="Safely unpack supported archives")
def archives(directory, unpack):
    """List or unpack archives in a directory."""
    root = Path(directory).expanduser().resolve()
    result = organize_archives_in_dir(root, unpack=unpack)

    for skipped in result.get("skipped_rar", []):
        console.print(f"[yellow]Skipped unsupported RAR archive: {skipped}[/yellow]")

    inventory = result.get("archives", [])
    if not inventory:
        console.print("[yellow]No supported archives found.[/yellow]")
        return

    for item in inventory:
        archive = item["archive"]
        console.print(f"\n[bold]{archive.name}[/bold]")
        if item.get("error"):
            console.print(f"[red]Error: {item['error']}[/red]")
            continue
        contents = item.get("contents", [])
        console.print(f"  {len(contents)} member(s)")
        if unpack and item.get("unpacked"):
            console.print(f"[green]Extracted to: {item.get('extracted_to')}[/green]")


@cli.command()
def cloud():
    """Discover local cloud-sync folder roots."""
    roots = discover_cloud_roots()
    if not roots:
        console.print("[yellow]No cloud sync folders detected.[/yellow]")
        console.print(f"Known providers: {', '.join(known_providers())}")
        return

    table = Table(title="Detected cloud sync roots")
    table.add_column("Provider")
    table.add_column("Path")
    for provider, path in sorted(roots.items()):
        table.add_row(provider, str(path))
    console.print(table)


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--quiet-hours", help="Hold batches during quiet hours, e.g. 22:00-07:00")
@click.option("--debounce", default=30, show_default=True, help="Debounce seconds")
def watch(directory, quiet_hours, debounce):
    """Watch a directory and organize new files in debounced batches."""
    root = Path(directory).expanduser().resolve()

    def on_batch(paths: list[Path]) -> None:
        console.print(f"[blue]Organizing batch of {len(paths)} file(s)...[/blue]")
        kwargs = {**ORGANIZE_DEFAULTS, "directory": str(root), "verbose": True}
        _run_organize(**kwargs)

    watcher = OrganizationWatcher(
        [root],
        on_batch_callback=on_batch,
        quiet_hours=quiet_hours,
        debounce_seconds=debounce,
    )
    backend = "watchdog" if watcher.using_watchdog() else "polling"
    console.print(f"[blue]Watching {root} ({backend} backend). Press Ctrl+C to stop.[/blue]")
    watcher.run_forever()


@cli.group()
def profiles():
    """Manage AI-rganize configuration profiles."""


@profiles.command("list")
def profiles_list():
    """List saved profiles."""
    names = list_profiles()
    if not names:
        default_profile()
        names = list_profiles()
    for name in names:
        console.print(name)


@profiles.command("show")
@click.argument("name", required=False, default="default")
def profiles_show(name):
    """Show a profile's settings."""
    profile = load_profile(name) if name in list_profiles() else default_profile()
    for key, value in profile.to_dict().items():
        console.print(f"{key}: {value}")


@profiles.command("save")
@click.argument("name", required=False, default="default")
@click.option("--roots", help="Comma-separated scan roots")
@click.option("--provider", default="openai")
@click.option("--model")
@click.option("--max-cost", type=float, default=1.0)
@click.option("--max-folders", type=int)
def profiles_save(name, roots, provider, model, max_cost, max_folders):
    """Save a profile to ~/.ai_rganize/profiles/."""
    profile = Profile(
        name=name,
        roots=[part.strip() for part in roots.split(",")] if roots else [],
        provider=provider,
        model=model,
        max_cost=max_cost,
        max_folders=max_folders,
    )
    path = save_profile(profile)
    console.print(f"[green]Saved profile '{name}' to {path}[/green]")


@cli.group()
def ledger():
    """Query the provenance ledger."""


@ledger.command("list")
def ledger_list():
    """List run IDs recorded in the ledger."""
    run_ids = list_run_ids()
    if not run_ids:
        console.print("[yellow]Ledger is empty.[/yellow]")
        return
    for run_id in run_ids:
        console.print(run_id)


@ledger.command("query")
@click.option("--run-id")
@click.option("--path-contains")
@click.option("--since", help="ISO-8601 timestamp lower bound")
def ledger_query(run_id, path_contains, since):
    """Query ledger records."""
    records = query(run_id=run_id, path_contains=path_contains, since=since)
    if not records:
        console.print("[yellow]No matching ledger records.[/yellow]")
        return
    for record in records:
        console.print(
            f"{record.get('timestamp')} [{record.get('run_id')}] "
            f"{record.get('source')} -> {record.get('dest')}"
        )


@ledger.command("undo-preview")
@click.argument("run_id")
def ledger_undo_preview(run_id):
    """Preview undo moves for a ledger run ID."""
    preview = undo_preview(run_id)
    if not preview:
        console.print("[yellow]No records found for that run ID.[/yellow]")
        return
    for item in preview:
        console.print(
            f"{item['current_location']} -> {item['would_restore_to']} "
            f"({item.get('folder', '')})"
        )


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def explain(path):
    """Explain why a file might be categorized a certain way (no AI)."""
    info = explain_file(Path(path))
    console.print(Panel.fit(f"[bold]{info['name']}[/bold]", border_style="blue"))
    console.print(f"Category: {info['category']}")
    console.print(f"Extension: {info['extension'] or '(none)'}")
    console.print(f"Size: {info['size_bytes']} bytes")
    if info.get("modified"):
        console.print(f"Modified: {info['modified']}")
    for reason in info.get("reasons", []):
        console.print(f"- {reason}")


def main():
    """Entry point for console scripts."""
    cli()


if __name__ == "__main__":
    main()
