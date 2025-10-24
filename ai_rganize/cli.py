"""
Command-line interface for AIrganizer
"""

import click
from pathlib import Path
from .organizers import RuleBasedOrganizer, AIOrganizer
from rich.console import Console
from rich.panel import Panel


@click.command()
@click.option('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--directory', '-d', help='Specific directory to organize (default: all common dirs)')
@click.option('--dry-run', is_flag=True, help='Show organization plan without moving files')
@click.option('--backup/--no-backup', default=True, help='Create backup before organizing')
@click.option('--ai-limit', default=50, help='Maximum number of files to analyze with AI (default: 50)')
@click.option('--max-file-size', default=10, help='Maximum file size in MB for AI analysis (default: 10)')
@click.option('--batch-size', default=5, help='Number of files to process in each AI batch (default: 5)')
@click.option('--max-cost', default=1.0, help='Maximum cost in USD for AI processing (default: 1.0)')
@click.option('--llm-provider', default='openai', type=click.Choice(['openai', 'claude', 'gemini'], case_sensitive=False), help='LLM provider to use for AI categorization (default: openai)')
@click.option('--openai-model', default='gpt-4o', help='OpenAI model to use for AI categorization (default: gpt-4o)')
@click.option('--no-ai', is_flag=True, help='Disable AI categorization, use only rule-based')
@click.option('--summary-only', is_flag=True, help='Show only summary in dry run (no file details)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed processing information')
def main(api_key, directory, dry_run, backup, ai_limit, max_file_size, batch_size, max_cost, llm_provider, openai_model, no_ai, summary_only, verbose):
    """AI-rganize - Intelligently organize your files using AI."""
    
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]AI-rganize[/bold blue]\n"
        "Intelligently organize your files using AI",
        border_style="blue"
    ))
    
    # Validate LLM provider and model combination
    if llm_provider != 'openai' and openai_model != 'gpt-4o':
        console.print(f"[yellow]Warning: --openai-model is only used with OpenAI provider. Using default model for {llm_provider}.[/yellow]")
    
    try:
        if no_ai:
            organizer = RuleBasedOrganizer(max_file_size_mb=max_file_size)
        else:
            organizer = AIOrganizer(api_key, max_file_size_mb=max_file_size, 
                                  batch_size=batch_size, max_cost=max_cost, model=openai_model, 
                                  llm_provider=llm_provider)
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
    
    organizer.display_organization_plan(plan, show_details=not summary_only)
    
    if dry_run:
        console.print("[yellow]Dry run complete. No files were moved.[/yellow]")
        return
    
    # Confirm with user
    if not console.confirm("Do you want to proceed with the organization?"):
        console.print("[yellow]Organization cancelled.[/yellow]")
        return
    
    # Create backup if requested
    if backup:
        if not organizer.create_backup(all_files):
            if not console.confirm("Backup failed. Continue without backup?"):
                console.print("[yellow]Organization cancelled.[/yellow]")
                return
    
    # Execute organization
    console.print("[blue]Organizing files...[/blue]")
    success = organizer.execute_organization(plan, target_dirs[0])
    
    if success:
        console.print("[green]File organization completed successfully![/green]")
    else:
        console.print("[red]File organization failed.[/red]")


if __name__ == '__main__':
    main()
