"""
Migration Assistant Command
Phase 4: Help users migrate from legacy structure to new structure
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich.progress import track

from src.utils.system.logger import logger

console = Console()


@dataclass
class MigrationTask:
    """Single migration task."""
    
    description: str
    old_path: Path
    new_path: Path
    action: str  # 'move', 'rename', 'create', 'copy'
    
    def execute(self, dry_run: bool = False) -> bool:
        """
        Execute the migration task.
        
        Args:
            dry_run: If True, only simulate the action
            
        Returns:
            True if successful
        """
        try:
            if dry_run:
                console.print(f"[dim]Would {self.action}: {self.old_path} → {self.new_path}[/dim]")
                return True
            
            if self.action == 'move':
                if self.old_path.exists():
                    shutil.move(str(self.old_path), str(self.new_path))
                    console.print(f"✅ Moved {self.old_path} → {self.new_path}")
            elif self.action == 'rename':
                if self.old_path.exists():
                    self.old_path.rename(self.new_path)
                    console.print(f"✅ Renamed {self.old_path} → {self.new_path}")
            elif self.action == 'copy':
                if self.old_path.exists():
                    if self.old_path.is_file():
                        shutil.copy2(str(self.old_path), str(self.new_path))
                    else:
                        shutil.copytree(str(self.old_path), str(self.new_path))
                    console.print(f"✅ Copied {self.old_path} → {self.new_path}")
            elif self.action == 'create':
                self.new_path.mkdir(parents=True, exist_ok=True)
                console.print(f"✅ Created {self.new_path}")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to {self.action} {self.old_path}: {e}[/red]")
            return False


def check_legacy_structure() -> List[Tuple[str, str, str]]:
    """
    Check for legacy project structure.
    
    Returns:
        List of (old_path, new_path, description) tuples
    """
    legacy_items = []
    
    # Check legacy directories
    checks = [
        ("config", "configs", "Config directory"),
        ("models/recipes", "recipes", "Recipe directory"),
        (".env", ".env.local", "Environment file"),
    ]
    
    for old, new, name in checks:
        old_path = Path(old)
        if old_path.exists():
            legacy_items.append((old, new, name))
    
    # Check for config files without environment suffix
    config_dir = Path("config") if Path("config").exists() else Path("configs")
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            if config_file.stem not in ["base", "local", "dev", "prod", "test"]:
                if not config_file.stem.endswith(("_local", "_dev", "_prod", "_test")):
                    legacy_items.append(
                        (str(config_file), 
                         str(config_file.parent / f"{config_file.stem}_local.yaml"),
                         f"Config file {config_file.name}")
                    )
    
    return legacy_items


def analyze_project_structure() -> List[MigrationTask]:
    """
    Analyze project and create migration tasks.
    
    Returns:
        List of migration tasks
    """
    tasks = []
    
    # 1. Check directory structure
    if Path("config").exists() and not Path("configs").exists():
        tasks.append(MigrationTask(
            description="Rename config/ to configs/",
            old_path=Path("config"),
            new_path=Path("configs"),
            action="rename"
        ))
    
    # 2. Check Recipe directory
    if Path("models/recipes").exists() and not Path("recipes").exists():
        tasks.append(MigrationTask(
            description="Move recipes to project root",
            old_path=Path("models/recipes"),
            new_path=Path("recipes"),
            action="move"
        ))
    
    # 3. Check environment files
    if Path(".env").exists():
        # Detect environment from .env content
        env_name = "local"  # Default
        if Path(".env").exists():
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("ENV_NAME="):
                        env_name = line.split("=")[1].strip()
                        break
        
        new_env_file = Path(f".env.{env_name}")
        if not new_env_file.exists():
            tasks.append(MigrationTask(
                description=f"Rename .env to .env.{env_name}",
                old_path=Path(".env"),
                new_path=new_env_file,
                action="rename"
            ))
    
    # 4. Create .env templates for common environments
    for env in ["local", "dev", "prod"]:
        env_file = Path(f".env.{env}")
        template_file = Path(f".env.{env}.template")
        
        if not env_file.exists() and not template_file.exists():
            tasks.append(MigrationTask(
                description=f"Create .env.{env}.template",
                old_path=Path("."),  # dummy
                new_path=template_file,
                action="create"
            ))
    
    return tasks


def migrate_config_files(dry_run: bool = False) -> None:
    """
    Migrate config files to environment-specific structure.
    
    Args:
        dry_run: If True, only simulate actions
    """
    config_dir = Path("configs") if Path("configs").exists() else Path("config")
    
    if not config_dir.exists():
        console.print("[yellow]No config directory found[/yellow]")
        return
    
    # Check for base.yaml + config.yaml pattern (legacy)
    base_file = config_dir / "base.yaml"
    config_file = config_dir / "config.yaml"
    
    if base_file.exists() and config_file.exists():
        console.print("\n[yellow]Legacy config structure detected (base.yaml + config.yaml)[/yellow]")
        console.print("Recommended action: Create environment-specific configs")
        
        if not dry_run:
            if Confirm.ask("Create local.yaml from base.yaml + config.yaml?"):
                # Merge base and config into local.yaml
                import yaml
                
                with open(base_file, 'r') as f:
                    base_config = yaml.safe_load(f) or {}
                
                with open(config_file, 'r') as f:
                    main_config = yaml.safe_load(f) or {}
                
                # Deep merge
                merged = {**base_config, **main_config}
                
                local_file = config_dir / "local.yaml"
                with open(local_file, 'w') as f:
                    yaml.dump(merged, f, default_flow_style=False)
                
                console.print(f"✅ Created {local_file}")
                console.print("[dim]Note: Review the merged config and adjust as needed[/dim]")


def migrate_command(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without applying"),
    interactive: bool = typer.Option(True, "--interactive", help="Interactive mode"),
    force: bool = typer.Option(False, "--force", help="Force migration without confirmation"),
) -> None:
    """
    Migrate legacy project structure to new structure.
    
    This command helps you:
    - Rename config/ to configs/
    - Move recipes to project root
    - Create environment-specific .env files
    - Update config file structure
    
    Examples:
        mmp migrate                  # Interactive migration
        mmp migrate --dry-run        # Preview changes
        mmp migrate --force          # Apply all changes without confirmation
    """
    console.print("[bold cyan]🔄 Migration Assistant[/bold cyan]\n")
    
    # 1. Check for legacy structure
    legacy_items = check_legacy_structure()
    
    if legacy_items:
        console.print("[yellow]⚠️ Legacy structure detected:[/yellow]\n")
        
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Current", style="red")
        table.add_column("Should be", style="green")
        table.add_column("Type")
        
        for old, new, desc in legacy_items:
            table.add_row(old, new, desc)
        
        console.print(table)
        console.print()
    
    # 2. Analyze and create tasks
    tasks = analyze_project_structure()
    
    if not tasks:
        console.print("[green]✅ Your project structure is up to date![/green]")
        return
    
    # 3. Show migration plan
    console.print("[bold]Migration Plan:[/bold]\n")
    for i, task in enumerate(tasks, 1):
        console.print(f"  {i}. {task.description}")
    
    console.print()
    
    # 4. Confirm and execute
    if not dry_run:
        if interactive and not force:
            if not Confirm.ask("Proceed with migration?"):
                console.print("[yellow]Migration cancelled[/yellow]")
                return
        
        # Execute tasks
        console.print("\n[bold]Executing migration...[/bold]\n")
        
        success_count = 0
        for task in track(tasks, description="Migrating..."):
            if task.execute(dry_run=False):
                success_count += 1
        
        console.print(f"\n[green]✅ Migration complete! ({success_count}/{len(tasks)} tasks successful)[/green]")
        
        # Additional instructions
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Review the migrated files")
        console.print("2. Update your scripts to use --env-name parameter")
        console.print("3. Test your workflows with the new structure")
        console.print("4. Remove any backup files once confirmed")
        
    else:
        console.print("\n[dim]Dry run mode - no changes made[/dim]")
        console.print("\n[bold]Would execute:[/bold]")
        for task in tasks:
            task.execute(dry_run=True)
    
    # 5. Check config file migration
    if not dry_run:
        console.print("\n[bold]Checking config files...[/bold]")
        migrate_config_files(dry_run=False)