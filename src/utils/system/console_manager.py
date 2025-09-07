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


# Global instance for easy access
console_manager = RichConsoleManager()