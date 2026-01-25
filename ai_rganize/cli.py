"""CLI for AI-rganize."""

import click
from pathlib import Path
from .organizers import RuleBasedOrganizer, AIOrganizer
from .utils.metadata import (
    load_manifest,
    find_manifest,
    restore_from_manifest,
    cleanup_empty_folders
)
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel


@click.command()
@click.option('--api-key', help='API key for the selected LLM provider (or set corresponding env var)')
@click.option('--directory', '-d', help='Specific directory to organize (default: all common dirs)')
@click.option('--dry-run', is_flag=True, help='Show organization plan without moving files')
@click.option('--backup/--no-backup', default=False, help='Create backup before organizing (default: no-backup)')
@click.option('--ai-limit', default=50, help='Maximum number of files to analyze with AI (default: 50)')
@click.option('--max-file-size', default=None, type=int, help='Maximum file size in MB for content analysis API calls (cost control). Files larger than this will use metadata/filename only. If not set, no size limit is applied.')
@click.option('--batch-size', default=5, help='Number of files to process in each AI batch. Auto-calculated when --max-folders is set (default: 5)')
@click.option('--max-cost', default=1.0, help='Maximum cost in USD for AI processing (default: 1.0)')
@click.option('--max-folders', type=int, help='Maximum number of folders to create (e.g., 6 folders for 200 files). If not specified, LLM creates as many folders as needed.')
@click.option('--llm-provider', default='openai', type=click.Choice(['openai', 'claude', 'gemini', 'ollama', 'mistral', 'openrouter'], case_sensitive=False), help='LLM provider to use for AI categorization (default: openai)')
@click.option('--model', help='Model name to use (defaults vary by provider). Examples: gpt-4o, claude-sonnet-4, gemini-2.5-pro, llama3.1, mistral-large')
@click.option('--no-ai', is_flag=True, help='Disable AI categorization, use only rule-based')
@click.option('--summary-only', is_flag=True, help='Show only summary in dry run (no file details)')
@click.option('--save-manifest/--no-manifest', default=True, help='Save organization manifest for undo capability (default: save)')
@click.option('--restore', type=click.Path(exists=True), help='Restore files from a manifest file (undo organization)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed processing information')
def main(api_key, directory, dry_run, backup, ai_limit, max_file_size, batch_size, max_cost, max_folders, llm_provider, model, no_ai, summary_only, save_manifest, restore, verbose):
    """AI-rganize - Intelligently organize your files using AI."""
    
    console = Console()
    
    # Handle restore operation
    if restore:
        console.print(Panel.fit(
            "[bold blue]AI-rganize - Restore[/bold blue]\n"
            "Restoring files from manifest",
            border_style="blue"
        ))
        
        try:
            manifest_path = Path(restore)
            manifest = load_manifest(manifest_path)
            
            console.print(f"[blue]Manifest created: {manifest.created}[/blue]")
            console.print(f"[blue]Source directory: {manifest.source_directory}[/blue]")
            console.print(f"[blue]Files to restore: {len(manifest.moves)}[/blue]")
            
            if not Confirm.ask("Do you want to restore these files to their original locations?"):
                console.print("[yellow]Restore cancelled.[/yellow]")
                return
            
            console.print("[blue]Restoring files...[/blue]")
            successful, failed = restore_from_manifest(manifest, verbose=True)
            
            if successful > 0:
                console.print(f"[green]✅ Successfully restored {successful} files[/green]")
                
                # Clean up empty folders
                source_dir = Path(manifest.source_directory)
                removed = cleanup_empty_folders(source_dir, verbose=verbose)
                if removed > 0:
                    console.print(f"[green]🗑️  Removed {removed} empty folders[/green]")
            
            if failed > 0:
                console.print(f"[red]❌ Failed to restore {failed} files[/red]")
            
        except FileNotFoundError:
            console.print(f"[red]Error: Manifest file not found: {restore}[/red]")
        except Exception as e:
            console.print(f"[red]Error restoring files: {e}[/red]")
        
        return
    
    console.print(Panel.fit(
        "[bold blue]AI-rganize[/bold blue]\n"
        "Intelligently organize your files using AI",
        border_style="blue"
    ))
    
    # Set default model based on provider if not specified
    if not model:
        default_models = {
            'openai': 'gpt-4o',
            'claude': 'claude-sonnet-4-20250514',  # Latest Claude Sonnet 4.5
            'gemini': 'gemini-2.5-pro-exp',  # Latest Gemini 2.5 Pro
            'ollama': 'llama3.1',
            'mistral': 'mistral-large-latest',
            'openrouter': 'anthropic/claude-3.5-sonnet',
        }
        model = default_models.get(llm_provider.lower(), 'gpt-4o')
        if verbose:
            console.print(f"[blue]Using default model for {llm_provider}: {model}[/blue]")
    
    try:
        # Set default max_file_size_mb to None if not specified (no limit)
        max_file_size_mb = max_file_size if max_file_size is not None else 10000  # Very high default, effectively no limit
        if no_ai:
            organizer = RuleBasedOrganizer(max_file_size_mb=max_file_size_mb)
        else:
            organizer = AIOrganizer(api_key, max_file_size_mb=max_file_size_mb, 
                                  batch_size=batch_size, max_cost=max_cost, model=model, 
                                  llm_provider=llm_provider, max_folders=max_folders)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return
    
    # Check permissions
    if not organizer.check_permissions():
        console.print("[red]Cannot proceed without proper permissions.[/red]")
        console.print("[yellow]Run: python -m airganizer.permissions for setup help[/yellow]")
        return
    
    # Determine target directories
    if directory:
        target_dirs = [Path(directory)]
    else:
        target_dirs = [path for path in organizer.target_dirs.values() if path.exists()]
    
    if not target_dirs:
        console.print("[red]No accessible directories found.[/red]")
        return
    
    # Scan files
    all_files = []
    for target_dir in target_dirs:
        console.print(f"[blue]Scanning {target_dir}...[/blue]")
        files = organizer.scan_files(target_dir)
        all_files.extend(files)
        console.print(f"[green]Found {len(files)} files in {target_dir.name}[/green]")
    
    if not all_files:
        console.print("[yellow]No files found to organize.[/yellow]")
        return
    
    # Create organization plan
    if no_ai:
        console.print("[yellow]Using rule-based categorization[/yellow]")
        plan = organizer.create_organization_plan(all_files, verbose=verbose)
    else:
        console.print("[blue]Using AI-powered categorization[/blue]")
        plan = organizer.create_organization_plan(all_files, ai_limit, verbose=verbose)
    
    # Check if plan is empty
    plan_folders = [k for k in plan.keys() if k != 'summary']
    if not plan_folders:
        console.print("[red]Error: No organization plan was created. No files were categorized.[/red]")
        if not verbose:
            console.print("[yellow]Tip: Try running with --verbose flag to see detailed processing information[/yellow]")
        return
    
    organizer.display_organization_plan(plan, show_details=not summary_only)
    
    if dry_run:
        console.print("[yellow]Dry run complete. No files were moved.[/yellow]")
        return
    
    # Confirm with user
    if not Confirm.ask("Do you want to proceed with the organization?"):
        console.print("[yellow]Organization cancelled.[/yellow]")
        return
    
    # Create backup if requested
    if backup:
        if not organizer.create_backup(all_files):
            if not Confirm.ask("Backup failed. Continue without backup?"):
                console.print("[yellow]Organization cancelled.[/yellow]")
                return
    
    # Execute organization
    console.print("[blue]Organizing files...[/blue]")
    success = organizer.execute_organization(
        plan, 
        target_dirs[0],
        save_manifest_file=save_manifest,
        ai_provider=None if no_ai else llm_provider,
        model=None if no_ai else model
    )
    
    if success:
        console.print("[green]File organization completed successfully![/green]")
        if save_manifest:
            console.print("[dim]💡 Tip: Use --restore <manifest_path> to undo this organization[/dim]")
    else:
        console.print("[red]File organization failed.[/red]")


if __name__ == '__main__':
    main()
