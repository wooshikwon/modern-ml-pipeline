"""
Rich Console Manager for Unified Logging System
Simple, clean output without unnecessary boxes or complex hierarchies.
"""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import threading
import time


class RichConsoleManager:
    """
    Unified console manager for ML pipeline output.
    Provides clean, emoji-based hierarchical output without boxes.
    """
    
    def __init__(self):
        self.console = Console()
        self.current_pipeline = None
        self.progress_bars = {}
        self.iteration_counters = {}
        self.active_progress = None
    
    def print(self, *args, **kwargs):
        """Direct console print wrapper"""
        self.console.print(*args, **kwargs)
    
    def log_milestone(self, message: str, level: str = "info"):
        """
        Log important milestones with emoji prefixes.
        
        Args:
            message: The message to log
            level: info, success, warning, error
        """
        emoji_map = {
            "info": "ℹ️",
            "success": "✅", 
            "warning": "⚠️",
            "error": "❌",
            "start": "🚀",
            "data": "📊",
            "model": "🤖",
            "optimization": "🎯",
            "mlflow": "📤",
            "finish": "🏁"
        }
        
        emoji = emoji_map.get(level, "📝")
        self.console.print(f"{emoji} {message}")
    
    @contextmanager
    def pipeline_context(self, name: str, description: str):
        """
        Pipeline-level context manager with simple emoji header.
        
        Args:
            name: Pipeline name
            description: Additional context information
        """
        self.current_pipeline = name
        self.log_milestone(f"{name}", "start")
        if description:
            self.console.print(description)
        self.console.print()  # Empty line for separation
        
        try:
            yield
        finally:
            self.log_milestone(f"{name} completed", "finish")
            self.current_pipeline = None
            self.console.print()  # Empty line after completion
    
    @contextmanager
    def progress_tracker(self, task_id: str, total: int, description: str, show_progress: bool = True):
        """
        Progress bar context for iterative processes.
        
        Args:
            task_id: Unique identifier for the task
            total: Total number of items to process
            description: Description of the task
            show_progress: Whether to show progress bar (False for hyperopt)
        """
        if not show_progress:
            # Simple text-only mode for hyperparameter optimization
            self.log_milestone(description, "optimization")
            yield lambda current=0: None  # No-op progress update function
            return
        
        # Create progress bar for tasks that need visual progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False
        ) as progress:
            
            task = progress.add_task(description, total=total)
            self.progress_bars[task_id] = (progress, task)
            
            def update_progress(current: int):
                progress.update(task, completed=current)
            
            try:
                yield update_progress
            finally:
                progress.update(task, completed=total)
                if task_id in self.progress_bars:
                    del self.progress_bars[task_id]
    
    def log_periodic(self, process_id: str, iteration: int, data: Dict[str, Any], every_n: int = 10):
        """
        Periodic output for high-iteration processes (simple line output).
        
        Args:
            process_id: Process identifier
            iteration: Current iteration (0-based)
            data: Data to display
            every_n: Show output every N iterations
        """
        if (iteration + 1) % every_n == 0 or iteration == 0:
            if process_id == "optuna_trials":
                trial_num = data.get("trial", iteration + 1)
                total_trials = data.get("total_trials", "?")
                score = data.get("score", 0.0)
                best_score = data.get("best_score", score)
                params = data.get("params", {})
                
                # Display trial info in single line
                status = "🔥" if score >= best_score else ""
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                self.console.print(f"Trial {trial_num}/{total_trials}: score={score:.4f} (best: {best_score:.4f}) {status} | {param_str}")
            else:
                # Generic periodic output for other processes
                self.console.print(f"[{process_id}] Iteration {iteration + 1}: {data}")
    
    def log_phase(self, phase_name: str, emoji: str = "📝"):
        """
        Log a new phase with emoji header.
        
        Args:
            phase_name: Name of the phase
            emoji: Emoji to use (defaults to 📝)
        """
        self.console.print()  # Empty line before phase
        self.console.print(f"{emoji} {phase_name}")
    
    def display_metrics_table(self, metrics: Dict[str, float], title: str = "Metrics"):
        """
        Display final metrics in a clean table format.
        Only used for final results, not progress tracking.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Table title
        """
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta", justify="right")
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        
        self.console.print(table)
        self.console.print()  # Empty line after table
    
    def display_run_info(self, run_id: str, model_uri: str = None, tracking_uri: str = None):
        """
        Display MLflow run information in a clean format.
        
        Args:
            run_id: MLflow run ID
            model_uri: Model URI if available
            tracking_uri: MLflow tracking URI if available
        """
        self.console.print()
        self.console.print(f"🎯 Run ID: [bold cyan]{run_id}[/bold cyan]")
        
        if model_uri:
            self.console.print(f"📦 Model URI: [bold green]{model_uri}[/bold green]")
        
        if tracking_uri:
            # Extract experiment and run info from tracking URI
            if "experiments" in tracking_uri:
                self.console.print(f"🔗 MLflow URI: [link]{tracking_uri}[/link]")
            else:
                self.console.print(f"🔗 Tracking URI: [link]{tracking_uri}[/link]")
        
        self.console.print()
    
    def log_artifacts_progress(self, artifacts: List[str]):
        """
        Log artifact upload progress with simple checkmarks.
        
        Args:
            artifacts: List of artifact names being uploaded
        """
        self.log_phase("MLflow Experiment Tracking", "📤")
        
        total = len(artifacts)
        with self.progress_tracker("mlflow_artifacts", total, f"Uploading {total} artifacts") as update:
            for i, artifact in enumerate(artifacts):
                time.sleep(0.1)  # Simulate upload time
                update(i + 1)
                self.console.print(f"✅ {artifact} logged")
    
    def cleanup_completed_tasks(self):
        """Remove completed progress bars to save memory"""
        completed = [task_id for task_id, (progress, task) in self.progress_bars.items() 
                    if task.finished]
        for task_id in completed:
            del self.progress_bars[task_id]
    
    def is_ci_environment(self) -> bool:
        """Check if running in CI/CD environment"""
        import os
        return any(env in os.environ for env in ['CI', 'GITHUB_ACTIONS', 'JENKINS_URL'])
    
    def get_console_mode(self) -> str:
        """Determine appropriate console mode"""
        import sys
        if self.is_ci_environment():
            return "plain"  # No colors, no animations
        elif not sys.stdout.isatty():
            return "plain"  # Pipe/redirect detected
        else:
            return "rich"   # Full Rich experience

    # ===== Enhanced Methods for Unified Console Integration =====
    
    def log_component_init(self, component_name: str, status: str = "success"):
        """Log component initialization with consistent formatting"""
        emoji = "✅" if status == "success" else "❌" if status == "error" else "🔄"
        self.console.print(f"{emoji} {component_name} initialized")
    
    def log_processing_step(self, step_name: str, details: str = ""):
        """Log processing steps with optional details"""
        self.console.print(f"   🔄 {step_name}")
        if details:
            self.console.print(f"      [dim]{details}[/dim]")
    
    def log_warning_with_context(self, message: str, context: Dict[str, Any] = None):
        """Enhanced warning with context information"""
        self.console.print(f"⚠️  [yellow]{message}[/yellow]")
        if context:
            for key, value in context.items():
                self.console.print(f"      [dim]{key}: {value}[/dim]")
    
    def log_database_operation(self, operation: str, details: str = ""):
        """Database-specific logging with database emoji"""
        self.console.print(f"🗄️  {operation}")
        if details:
            self.console.print(f"      [dim]{details}[/dim]")
    
    def log_feature_engineering(self, step: str, columns: List[str], result_info: str = ""):
        """Feature engineering specific logging"""
        self.console.print(f"🔬 {step}")
        if columns:
            cols_display = ', '.join(columns[:5])
            if len(columns) > 5:
                cols_display += f"... (+{len(columns)-5} more)"
            self.console.print(f"   [dim]Columns: {cols_display}[/dim]")
        if result_info:
            self.console.print(f"   [dim]Result: {result_info}[/dim]")
    
    def log_data_operation(self, operation: str, shape: tuple = None, details: str = ""):
        """Data operation logging with shape information"""
        shape_str = f" ({shape[0]} rows, {shape[1]} columns)" if shape else ""
        self.console.print(f"📊 {operation}{shape_str}")
        if details:
            self.console.print(f"   [dim]{details}[/dim]")
    
    def log_model_operation(self, operation: str, model_info: str = ""):
        """Model-specific operations"""
        self.console.print(f"🤖 {operation}")
        if model_info:
            self.console.print(f"   [dim]{model_info}[/dim]")
    
    def log_file_operation(self, operation: str, file_path: str, details: str = ""):
        """File operations with path information"""
        # Show only filename for cleaner output
        from pathlib import Path
        filename = Path(file_path).name if file_path else "file"
        self.console.print(f"📁 {operation}: [cyan]{filename}[/cyan]")
        if details:
            self.console.print(f"   [dim]{details}[/dim]")
    
    def log_error_with_context(self, error_message: str, context: Dict[str, Any] = None, suggestion: str = None):
        """Enhanced error logging with context and suggestions"""
        self.console.print(f"❌ [red]Error: {error_message}[/red]")
        if context:
            for key, value in context.items():
                self.console.print(f"   [dim]{key}: {value}[/dim]")
        if suggestion:
            self.console.print(f"   [blue]💡 Suggestion: {suggestion}[/blue]")
    
    def log_validation_result(self, item: str, status: str, details: str = ""):
        """Validation results with clear status indicators"""
        emoji = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
        color = "green" if status == "pass" else "red" if status == "fail" else "yellow"
        self.console.print(f"{emoji} [{color}]{item}[/{color}]")
        if details:
            self.console.print(f"   [dim]{details}[/dim]")
    
    def log_connection_status(self, service: str, status: str, details: str = ""):
        """Connection status for external services"""
        emoji = "🔗" if status == "connected" else "❌" if status == "failed" else "🔄"
        color = "green" if status == "connected" else "red" if status == "failed" else "yellow"
        self.console.print(f"{emoji} [{color}]{service}: {status.title()}[/{color}]")
        if details:
            self.console.print(f"   [dim]{details}[/dim]")


# ===== Unified Console Class for Dual Output =====

class UnifiedConsole:
    """
    Unified console that provides both rich interactive output and structured logging.
    Automatically detects environment and adjusts output accordingly.
    """
    
    def __init__(self, settings=None):
        self.rich_console = RichConsoleManager()
        from src.utils.system.logger import logger
        self.logger = logger
        self.mode = self._detect_output_mode(settings)
    
    def info(self, message: str, rich_message: str = None, **rich_kwargs):
        """Unified info logging with dual output"""
        self.logger.info(message)  # Always log to file/system
        
        if self.mode in ["rich", "test"] and rich_message:
            self.rich_console.console.print(rich_message, **rich_kwargs)
        elif self.mode in ["rich", "test"] and not rich_message:
            self.rich_console.log_milestone(message, "info")
        elif self.mode == "plain":
            print(f"INFO: {message}")
    
    def error(self, message: str, rich_message: str = None, context: Dict[str, Any] = None, suggestion: str = None):
        """Unified error logging"""
        self.logger.error(message)
        
        if self.mode in ["rich", "test"]:
            if rich_message:
                self.rich_console.console.print(rich_message, style="red")
            else:
                self.rich_console.log_error_with_context(message, context, suggestion)
        elif self.mode == "plain":
            print(f"ERROR: {message}")
    
    def warning(self, message: str, rich_message: str = None, context: Dict[str, Any] = None):
        """Unified warning logging"""
        self.logger.warning(message)
        
        if self.mode in ["rich", "test"]:
            if rich_message:
                self.rich_console.console.print(rich_message, style="yellow")
            else:
                self.rich_console.log_warning_with_context(message, context)
        elif self.mode == "plain":
            print(f"WARNING: {message}")
    
    def debug(self, message: str, rich_message: str = None, **rich_kwargs):
        """Unified debug logging"""
        self.logger.debug(message)
        
        if self.mode in ["rich", "test"] and rich_message:
            self.rich_console.console.print(rich_message, style="dim", **rich_kwargs)
        elif self.mode in ["rich", "test"] and not rich_message:
            self.rich_console.console.print(f"🔍 {message}", style="dim")
        elif self.mode == "plain":
            print(f"DEBUG: {message}")
    
    def component_init(self, component_name: str, status: str = "success"):
        """Component initialization logging"""
        if self.mode in ["rich", "test"]:
            self.rich_console.log_component_init(component_name, status)
        else:
            status_text = "✓" if status == "success" else "✗" if status == "error" else "~"
            print(f"{status_text} {component_name} initialized")
    
    def data_operation(self, operation: str, shape: tuple = None, details: str = ""):
        """Data operation logging"""
        if self.mode in ["rich", "test"]:
            self.rich_console.log_data_operation(operation, shape, details)
        else:
            shape_str = f" ({shape[0]} rows, {shape[1]} columns)" if shape else ""
            print(f"DATA: {operation}{shape_str}")
    
    def _detect_output_mode(self, settings) -> str:
        """Detect appropriate output mode"""
        import os
        # Test environment detection
        if os.environ.get('PYTEST_CURRENT_TEST') or 'pytest' in os.environ.get('_', ''):
            return "test"  # New test mode for pytest
        elif self.rich_console.is_ci_environment():
            return "plain"
        elif settings and hasattr(settings, 'console_mode'):
            return settings.console_mode
        else:
            return "rich"


# Global instances for easy access
console_manager = RichConsoleManager()
unified_console = UnifiedConsole()

# ===== Global Helper Functions for Easy Import =====

def get_console(settings=None) -> UnifiedConsole:
    """
    Get a UnifiedConsole instance with proper settings.
    Use this in modules that need console output.
    
    Args:
        settings: Optional Settings object for configuration
        
    Returns:
        UnifiedConsole: Configured console instance
    """
    return UnifiedConsole(settings)

def get_rich_console(settings=None) -> RichConsoleManager:
    """
    Get a RichConsoleManager instance for advanced Rich features.
    Use this when you need progress bars, tables, or complex formatting.
    
    Args:
        settings: Optional Settings object for configuration
        
    Returns:
        RichConsoleManager: Rich console manager instance
    """
    return RichConsoleManager()

# ===== CLI Helper Functions =====

def cli_print(message: str, style: str = None, emoji: str = None):
    """
    CLI-optimized print function that uses Rich Console formatting.
    Replaces typer.echo for consistent Rich integration.
    
    Args:
        message: Message to print
        style: Rich style (e.g., 'bold green', 'red', 'cyan')
        emoji: Emoji prefix to add
    """
    if emoji:
        message = f"{emoji} {message}"
    
    if style:
        console_manager.console.print(message, style=style)
    else:
        console_manager.console.print(message)

def cli_success(message: str):
    """Print success message with consistent styling"""
    cli_print(message, style="bold green", emoji="✅")

def cli_error(message: str):
    """Print error message with consistent styling"""
    cli_print(message, style="bold red", emoji="❌")

def cli_warning(message: str):
    """Print warning message with consistent styling"""
    cli_print(message, style="bold yellow", emoji="⚠️")

def cli_info(message: str):
    """Print info message with consistent styling"""
    cli_print(message, style="bold blue", emoji="ℹ️")

# ===== Test Helper Functions =====

def test_print(message: str, emoji: str = "📝"):
    """
    Test-optimized print function for E2E tests.
    Maintains Rich formatting but optimized for test output.
    
    Args:
        message: Message to print
        emoji: Emoji prefix (default: 📝)
    """
    # Use a simple Rich console for tests to avoid animation delays
    from rich.console import Console
    test_console = Console()
    test_console.print(f"{emoji} {message}")

def phase_print(phase_name: str, emoji: str = "🔍"):
    """Print test phase header with consistent formatting"""
    test_print(f"\n=== {phase_name} ===", emoji=emoji)

def success_print(message: str):
    """Print test success with consistent styling"""
    test_print(message, emoji="✅")

def test_info(message: str):
    """Print test info with consistent styling"""  
    test_print(message, emoji="ℹ️")

# ===== Quick Access Aliases =====
# For backward compatibility and convenience
rich_console = console_manager
console = unified_console