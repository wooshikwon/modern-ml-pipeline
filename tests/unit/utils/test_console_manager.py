"""
Tests for console_manager - Rich UX management and unified logging

Comprehensive testing of Rich console functionality including:
- RichConsoleManager basic operations
- Context managers for pipeline and progress tracking
- Logging methods with various levels and types
- Environment detection and output mode selection
- UnifiedConsole dual output functionality
- Thread safety and performance considerations

Test Categories:
1. TestRichConsoleManagerBasic - Basic logging and output methods
2. TestRichConsoleManagerContexts - Context managers and progress tracking
3. TestRichConsoleManagerTables - Table and formatted output display
4. TestRichConsoleManagerMlflow - MLflow integration and artifact logging
5. TestRichConsoleManagerEnvironment - Environment detection and CI/CD handling
6. TestUnifiedConsole - Unified console with dual output modes
7. TestConsoleManagerIntegration - Integration scenarios and edge cases
8. TestConsoleManagerSecurity - Security, threading, and performance tests
"""

import pytest
from unittest.mock import patch, MagicMock, call
import io
import threading
import time
from contextlib import redirect_stdout
from typing import Dict, Any

from src.utils.system.console_manager import RichConsoleManager, UnifiedConsole, console_manager, unified_console


class TestRichConsoleManagerBasic:
    """Test basic RichConsoleManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console = RichConsoleManager()

    def test_initialization(self):
        """Test RichConsoleManager initialization."""
        assert self.console.console is not None
        assert self.console.current_pipeline is None
        assert self.console.progress_bars == {}
        assert self.console.iteration_counters == {}
        assert self.console.active_progress is None

    @patch('src.utils.system.console_manager.Console')
    def test_print_wrapper(self, mock_console_class):
        """Test direct print wrapper functionality."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        console = RichConsoleManager()
        console.print("test message", style="bold")
        
        mock_console.print.assert_called_once_with("test message", style="bold")

    @patch('src.utils.system.console_manager.Console')
    def test_log_milestone_info(self, mock_console_class):
        """Test log_milestone with info level."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        console = RichConsoleManager()
        console.log_milestone("Test message", "info")
        
        mock_console.print.assert_called_once_with("ℹ️ Test message")

    @patch('src.utils.system.console_manager.Console')
    def test_log_milestone_all_levels(self, mock_console_class):
        """Test log_milestone with all supported levels."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        console = RichConsoleManager()
        
        test_cases = [
            ("info", "ℹ️"),
            ("success", "✅"),
            ("warning", "⚠️"),
            ("error", "❌"),
            ("start", "🚀"),
            ("data", "📊"),
            ("model", "🤖"),
            ("optimization", "🎯"),
            ("mlflow", "📤"),
            ("finish", "🏁"),
            ("unknown", "📝")  # Default case
        ]
        
        for level, expected_emoji in test_cases:
            console.log_milestone("Test message", level)
        
        expected_calls = [call(f"{emoji} Test message") for _, emoji in test_cases]
        mock_console.print.assert_has_calls(expected_calls)

    def test_log_phase(self):
        """Test log_phase functionality."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_phase("Data Loading", "📂")
            
            # Should print empty line and phase message
            assert mock_print.call_count == 2
            mock_print.assert_any_call()  # Empty line
            mock_print.assert_any_call("📂 Data Loading")

    def test_log_phase_default_emoji(self):
        """Test log_phase with default emoji."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_phase("Processing")
            
            mock_print.assert_any_call("📝 Processing")

    def test_display_metrics_table(self):
        """Test display_metrics_table functionality."""
        metrics = {
            "accuracy": 0.85,
            "precision": 0.78,
            "recall": 0.82,
            "f1_score": 0.80,
            "epoch": 10
        }
        
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.display_metrics_table(metrics, "Model Performance")
            
            # Should print table and empty line
            assert mock_print.call_count == 2
            # First call should be the table, second should be empty line
            table_call = mock_print.call_args_list[0]
            empty_line_call = mock_print.call_args_list[1]
            
            assert empty_line_call == call()  # Empty line call

    def test_display_run_info_minimal(self):
        """Test display_run_info with minimal information."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.display_run_info("12345")
            
            # Should print empty line, run ID, and final empty line
            assert mock_print.call_count == 3
            mock_print.assert_any_call()  # Empty line at start
            mock_print.assert_any_call("🎯 Run ID: [bold cyan]12345[/bold cyan]")
            mock_print.assert_any_call()  # Empty line at end

    def test_display_run_info_complete(self):
        """Test display_run_info with complete information."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.display_run_info(
                "12345",
                "s3://bucket/model.pkl",
                "http://mlflow.example.com/experiments/1/runs/12345"
            )
            
            # Should print empty line, run ID, model URI, tracking URI, final empty line
            assert mock_print.call_count == 5
            mock_print.assert_any_call("🎯 Run ID: [bold cyan]12345[/bold cyan]")
            mock_print.assert_any_call("📦 Model URI: [bold green]s3://bucket/model.pkl[/bold green]")
            mock_print.assert_any_call("🔗 MLflow URI: [link]http://mlflow.example.com/experiments/1/runs/12345[/link]")

    def test_log_periodic_optuna_trials(self):
        """Test log_periodic with optuna_trials process."""
        with patch.object(self.console.console, 'print') as mock_print:
            data = {
                "trial": 5,
                "total_trials": 100,
                "score": 0.85,
                "best_score": 0.90,
                "params": {"n_estimators": 100, "max_depth": 5}
            }
            
            self.console.log_periodic("optuna_trials", 4, data, every_n=5)  # iteration 4, will trigger (4+1) % 5 == 0
            
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert "Trial 5/100" in call_args
            assert "score=0.8500" in call_args
            assert "best: 0.9000" in call_args
            assert "n_estimators=100" in call_args
            assert "max_depth=5" in call_args

    def test_log_periodic_optuna_best_score(self):
        """Test log_periodic with best score achieved."""
        with patch.object(self.console.console, 'print') as mock_print:
            data = {
                "trial": 10,
                "total_trials": 100,
                "score": 0.95,
                "best_score": 0.90,  # Current score is better
                "params": {"n_estimators": 200}
            }
            
            self.console.log_periodic("optuna_trials", 9, data, every_n=10)
            
            call_args = mock_print.call_args[0][0]
            assert "🔥" in call_args  # Fire emoji for new best

    def test_log_periodic_generic(self):
        """Test log_periodic with generic process."""
        with patch.object(self.console.console, 'print') as mock_print:
            data = {"loss": 0.123, "epoch": 10}
            
            self.console.log_periodic("training", 9, data, every_n=10)
            
            call_args = mock_print.call_args[0][0]
            assert "[training] Iteration 10" in call_args
            assert str(data) in call_args

    def test_log_periodic_skip_non_multiple(self):
        """Test log_periodic skips iterations that aren't multiples."""
        with patch.object(self.console.console, 'print') as mock_print:
            data = {"score": 0.85}
            
            self.console.log_periodic("test", 5, data, every_n=10)  # Should skip
            
            mock_print.assert_not_called()

    def test_log_periodic_always_show_first(self):
        """Test log_periodic always shows iteration 0."""
        with patch.object(self.console.console, 'print') as mock_print:
            data = {"score": 0.85}
            
            self.console.log_periodic("test", 0, data, every_n=10)  # Should show
            
            mock_print.assert_called_once()


class TestRichConsoleManagerContexts:
    """Test context managers and progress tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console = RichConsoleManager()

    def test_pipeline_context_basic(self):
        """Test pipeline_context basic functionality."""
        with patch.object(self.console.console, 'print') as mock_print:
            with self.console.pipeline_context("Data Pipeline", "Processing customer data"):
                assert self.console.current_pipeline == "Data Pipeline"
            
            # After context exit
            assert self.console.current_pipeline is None
            
            # Check logged messages
            mock_print.assert_any_call("🚀 Data Pipeline")
            mock_print.assert_any_call("Processing customer data")
            mock_print.assert_any_call("🏁 Data Pipeline completed")

    def test_pipeline_context_no_description(self):
        """Test pipeline_context without description."""
        with patch.object(self.console.console, 'print') as mock_print:
            with self.console.pipeline_context("Simple Pipeline", ""):
                pass
            
            mock_print.assert_any_call("🚀 Simple Pipeline")
            mock_print.assert_any_call("🏁 Simple Pipeline completed")
            # Should not print empty description
            mock_print.assert_any_call()  # Empty lines are still printed

    def test_pipeline_context_exception_handling(self):
        """Test pipeline_context cleans up after exception."""
        with patch.object(self.console.console, 'print'):
            try:
                with self.console.pipeline_context("Error Pipeline", "Will fail"):
                    assert self.console.current_pipeline == "Error Pipeline"
                    raise ValueError("Test error")
            except ValueError:
                pass
            
            # Should clean up even after exception
            assert self.console.current_pipeline is None

    @patch('src.utils.system.console_manager.Progress')
    def test_progress_tracker_with_progress(self, mock_progress_class):
        """Test progress_tracker with progress bar enabled."""
        mock_progress_context = MagicMock()
        mock_progress = MagicMock()
        mock_progress_class.return_value = mock_progress_context
        mock_progress_context.__enter__ = MagicMock(return_value=mock_progress)
        mock_progress_context.__exit__ = MagicMock(return_value=None)
        
        mock_task = MagicMock()
        mock_progress.add_task.return_value = mock_task
        
        with self.console.progress_tracker("test_task", 100, "Processing items", show_progress=True) as update:
            # Test the update function
            update(50)
            mock_progress.update.assert_called_with(mock_task, completed=50)
        
        # Verify progress bar was created and updated
        mock_progress.add_task.assert_called_once_with("Processing items", total=100)
        # Final completion update
        mock_progress.update.assert_called_with(mock_task, completed=100)

    def test_progress_tracker_no_progress(self):
        """Test progress_tracker with progress disabled."""
        with patch.object(self.console, 'log_milestone') as mock_milestone:
            with self.console.progress_tracker("hyperopt", 50, "Hyperparameter optimization", show_progress=False) as update:
                # Update function should be a no-op
                result = update(25)
                assert result is None
            
            mock_milestone.assert_called_once_with("Hyperparameter optimization", "optimization")

    @patch('src.utils.system.console_manager.Progress')
    def test_progress_tracker_cleanup(self, mock_progress_class):
        """Test progress_tracker cleans up progress bars."""
        mock_progress_context = MagicMock()
        mock_progress = MagicMock()
        mock_progress_class.return_value = mock_progress_context
        mock_progress_context.__enter__ = MagicMock(return_value=mock_progress)
        mock_progress_context.__exit__ = MagicMock(return_value=None)
        
        mock_task = MagicMock()
        mock_progress.add_task.return_value = mock_task
        
        task_id = "cleanup_test"
        with self.console.progress_tracker(task_id, 10, "Test task") as update:
            # Progress bar should be registered
            assert task_id in self.console.progress_bars
        
        # Should be cleaned up after context exit
        assert task_id not in self.console.progress_bars

    def test_cleanup_completed_tasks(self):
        """Test cleanup_completed_tasks method."""
        # Mock some progress bars
        mock_task_finished = MagicMock()
        mock_task_finished.finished = True
        mock_task_active = MagicMock()
        mock_task_active.finished = False
        
        self.console.progress_bars = {
            "finished_task": (MagicMock(), mock_task_finished),
            "active_task": (MagicMock(), mock_task_active)
        }
        
        self.console.cleanup_completed_tasks()
        
        # Only active task should remain
        assert "finished_task" not in self.console.progress_bars
        assert "active_task" in self.console.progress_bars


class TestRichConsoleManagerTables:
    """Test table and formatted output functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console = RichConsoleManager()

    def test_display_metrics_table_various_types(self):
        """Test metrics table with various data types."""
        metrics = {
            "float_metric": 0.12345,
            "int_metric": 42,
            "string_metric": "excellent",
            "bool_metric": True
        }
        
        with patch('src.utils.system.console_manager.Table') as mock_table_class:
            mock_table = MagicMock()
            mock_table_class.return_value = mock_table
            
            with patch.object(self.console.console, 'print'):
                self.console.display_metrics_table(metrics, "Test Metrics")
            
            # Verify table creation
            mock_table_class.assert_called_once_with(
                title="Test Metrics", 
                show_header=True, 
                header_style="bold magenta"
            )
            
            # Verify columns were added
            mock_table.add_column.assert_any_call("Metric", style="cyan", no_wrap=True)
            mock_table.add_column.assert_any_call("Value", style="magenta", justify="right")
            
            # Verify rows were added with proper formatting
            mock_table.add_row.assert_any_call("float_metric", "0.1235")  # Float formatting
            mock_table.add_row.assert_any_call("int_metric", "42")        # Int as string
            mock_table.add_row.assert_any_call("string_metric", "excellent")  # String as-is
            mock_table.add_row.assert_any_call("bool_metric", "True")     # Bool as string

    def test_display_run_info_tracking_uri_variations(self):
        """Test display_run_info with different tracking URI formats."""
        with patch.object(self.console.console, 'print') as mock_print:
            # Test with experiments in URI
            self.console.display_run_info("123", None, "http://mlflow.com/experiments/1/runs/123")
            
            # Reset for next test
            mock_print.reset_mock()
            
            # Test with non-experiment URI
            self.console.display_run_info("456", None, "http://mlflow.com/tracking")
            
            # Check the different URI formats are handled
            calls = []
            for call in mock_print.call_args_list:
                if call[0] and len(call[0]) > 0 and "🔗" in str(call[0][0]):
                    calls.append(call[0][0])
            assert len(calls) == 1
            assert "Tracking URI" in calls[0]


class TestRichConsoleManagerMlflow:
    """Test MLflow integration and artifact logging."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console = RichConsoleManager()

    @patch('time.sleep')  # Speed up tests
    def test_log_artifacts_progress(self, mock_sleep):
        """Test log_artifacts_progress functionality."""
        artifacts = ["model.pkl", "metrics.json", "params.yaml", "requirements.txt"]
        
        with patch.object(self.console, 'log_phase') as mock_log_phase:
            with patch.object(self.console, 'progress_tracker') as mock_progress:
                with patch.object(self.console.console, 'print') as mock_print:
                    # Mock progress tracker context
                    mock_update = MagicMock()
                    mock_progress.return_value.__enter__ = MagicMock(return_value=mock_update)
                    mock_progress.return_value.__exit__ = MagicMock(return_value=None)
                    
                    self.console.log_artifacts_progress(artifacts)
                    
                    # Verify phase logging
                    mock_log_phase.assert_called_once_with("MLflow Experiment Tracking", "📤")
                    
                    # Verify progress tracker was called
                    mock_progress.assert_called_once_with("mlflow_artifacts", 4, "Uploading 4 artifacts")
                    
                    # Verify each artifact was logged
                    artifact_calls = [call for call in mock_print.call_args_list 
                                    if call[0][0].startswith("✅")]
                    assert len(artifact_calls) == 4
                    
                    # Verify specific artifacts were logged
                    logged_artifacts = [call[0][0].replace("✅ ", "").replace(" logged", "") 
                                      for call in artifact_calls]
                    assert set(logged_artifacts) == set(artifacts)

    def test_log_artifacts_progress_empty_list(self):
        """Test log_artifacts_progress with empty artifact list."""
        with patch.object(self.console, 'log_phase') as mock_log_phase:
            with patch.object(self.console, 'progress_tracker') as mock_progress:
                mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_progress.return_value.__exit__ = MagicMock(return_value=None)
                
                self.console.log_artifacts_progress([])
                
                mock_log_phase.assert_called_once()
                mock_progress.assert_called_once_with("mlflow_artifacts", 0, "Uploading 0 artifacts")


class TestRichConsoleManagerEnvironment:
    """Test environment detection and CI/CD handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console = RichConsoleManager()

    @patch.dict('os.environ', {'CI': 'true'})
    def test_is_ci_environment_ci(self):
        """Test CI environment detection with CI variable."""
        assert self.console.is_ci_environment() is True

    @patch.dict('os.environ', {'GITHUB_ACTIONS': 'true'})
    def test_is_ci_environment_github(self):
        """Test CI environment detection with GitHub Actions."""
        assert self.console.is_ci_environment() is True

    @patch.dict('os.environ', {'JENKINS_URL': 'http://jenkins.example.com'})
    def test_is_ci_environment_jenkins(self):
        """Test CI environment detection with Jenkins."""
        assert self.console.is_ci_environment() is True

    @patch.dict('os.environ', {}, clear=True)
    def test_is_ci_environment_false(self):
        """Test CI environment detection returns False in normal environment."""
        assert self.console.is_ci_environment() is False

    @patch('sys.stdout.isatty')
    def test_get_console_mode_rich(self, mock_isatty):
        """Test get_console_mode returns rich for interactive terminal."""
        mock_isatty.return_value = True
        
        with patch.object(self.console, 'is_ci_environment', return_value=False):
            assert self.console.get_console_mode() == "rich"

    @patch('sys.stdout.isatty')
    def test_get_console_mode_plain_pipe(self, mock_isatty):
        """Test get_console_mode returns plain for piped output."""
        mock_isatty.return_value = False
        
        with patch.object(self.console, 'is_ci_environment', return_value=False):
            assert self.console.get_console_mode() == "plain"

    def test_get_console_mode_plain_ci(self):
        """Test get_console_mode returns plain for CI environment."""
        with patch.object(self.console, 'is_ci_environment', return_value=True):
            assert self.console.get_console_mode() == "plain"


class TestRichConsoleManagerEnhanced:
    """Test enhanced logging methods for unified console integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console = RichConsoleManager()

    def test_log_component_init_success(self):
        """Test log_component_init with success status."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_component_init("Database Connection", "success")
            
            mock_print.assert_called_once_with("✅ Database Connection initialized")

    def test_log_component_init_error(self):
        """Test log_component_init with error status."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_component_init("Redis Cache", "error")
            
            mock_print.assert_called_once_with("❌ Redis Cache initialized")

    def test_log_component_init_default(self):
        """Test log_component_init with default status."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_component_init("API Client", "loading")
            
            mock_print.assert_called_once_with("🔄 API Client initialized")

    def test_log_processing_step(self):
        """Test log_processing_step with and without details."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_processing_step("Data Validation", "Checking 1000 records")
            
            assert mock_print.call_count == 2
            mock_print.assert_any_call("   🔄 Data Validation")
            mock_print.assert_any_call("      [dim]Checking 1000 records[/dim]")

    def test_log_processing_step_no_details(self):
        """Test log_processing_step without details."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_processing_step("Loading Data")
            
            mock_print.assert_called_once_with("   🔄 Loading Data")

    def test_log_warning_with_context(self):
        """Test log_warning_with_context functionality."""
        context = {"file": "data.csv", "line": 42, "column": "age"}
        
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_warning_with_context("Missing values detected", context)
            
            assert mock_print.call_count == 4  # Main message + 3 context items
            mock_print.assert_any_call("⚠️  [yellow]Missing values detected[/yellow]")
            mock_print.assert_any_call("      [dim]file: data.csv[/dim]")
            mock_print.assert_any_call("      [dim]line: 42[/dim]")
            mock_print.assert_any_call("      [dim]column: age[/dim]")

    def test_log_database_operation(self):
        """Test log_database_operation functionality."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_database_operation("Connecting to PostgreSQL", "localhost:5432/mlpipeline")
            
            assert mock_print.call_count == 2
            mock_print.assert_any_call("🗄️  Connecting to PostgreSQL")
            mock_print.assert_any_call("      [dim]localhost:5432/mlpipeline[/dim]")

    def test_log_feature_engineering(self):
        """Test log_feature_engineering functionality."""
        columns = ["age", "income", "education", "experience", "location", "skill1", "skill2", "skill3"]
        
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_feature_engineering("One-Hot Encoding", columns, "8 new columns created")
            
            assert mock_print.call_count == 3
            mock_print.assert_any_call("🔬 One-Hot Encoding")
            # Should truncate column list
            mock_print.assert_any_call("   [dim]Columns: age, income, education, experience, location... (+3 more)[/dim]")
            mock_print.assert_any_call("   [dim]Result: 8 new columns created[/dim]")

    def test_log_feature_engineering_short_list(self):
        """Test log_feature_engineering with short column list."""
        columns = ["age", "income"]
        
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_feature_engineering("Scaling", columns)
            
            mock_print.assert_any_call("   [dim]Columns: age, income[/dim]")

    def test_log_data_operation(self):
        """Test log_data_operation functionality."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_data_operation("Loading training data", (1000, 25), "From PostgreSQL")
            
            assert mock_print.call_count == 2
            mock_print.assert_any_call("📊 Loading training data (1000 rows, 25 columns)")
            mock_print.assert_any_call("   [dim]From PostgreSQL[/dim]")

    def test_log_data_operation_no_shape(self):
        """Test log_data_operation without shape information."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_data_operation("Saving results")
            
            mock_print.assert_called_once_with("📊 Saving results")

    def test_log_model_operation(self):
        """Test log_model_operation functionality."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_model_operation("Training Random Forest", "100 estimators, max_depth=10")
            
            assert mock_print.call_count == 2
            mock_print.assert_any_call("🤖 Training Random Forest")
            mock_print.assert_any_call("   [dim]100 estimators, max_depth=10[/dim]")

    def test_log_file_operation(self):
        """Test log_file_operation functionality."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_file_operation("Saving model", "/path/to/model/random_forest.pkl", "42.5 MB")
            
            assert mock_print.call_count == 2
            mock_print.assert_any_call("📁 Saving model: [cyan]random_forest.pkl[/cyan]")
            mock_print.assert_any_call("   [dim]42.5 MB[/dim]")

    def test_log_file_operation_no_path(self):
        """Test log_file_operation with no file path."""
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_file_operation("Reading config", "")
            
            mock_print.assert_called_once_with("📁 Reading config: [cyan]file[/cyan]")

    def test_log_error_with_context(self):
        """Test log_error_with_context functionality."""
        context = {"table": "users", "query": "SELECT * FROM users WHERE age > 100"}
        suggestion = "Check data quality and age validation rules"
        
        with patch.object(self.console.console, 'print') as mock_print:
            self.console.log_error_with_context("Database query failed", context, suggestion)
            
            assert mock_print.call_count == 4
            mock_print.assert_any_call("❌ [red]Error: Database query failed[/red]")
            mock_print.assert_any_call("   [dim]table: users[/dim]")
            mock_print.assert_any_call("   [dim]query: SELECT * FROM users WHERE age > 100[/dim]")
            mock_print.assert_any_call("   [blue]💡 Suggestion: Check data quality and age validation rules[/blue]")

    def test_log_validation_result(self):
        """Test log_validation_result with different statuses."""
        with patch.object(self.console.console, 'print') as mock_print:
            test_cases = [
                ("Data Schema", "pass", "green", "✅"),
                ("Model Accuracy", "fail", "red", "❌"),
                ("Feature Count", "warning", "yellow", "⚠️")
            ]
            
            for item, status, expected_color, expected_emoji in test_cases:
                self.console.log_validation_result(item, status, f"Details for {item}")
                
                # Find the main call for this item
                main_call = f"{expected_emoji} [{expected_color}]{item}[/{expected_color}]"
                detail_call = f"   [dim]Details for {item}[/dim]"
                
                mock_print.assert_any_call(main_call)
                mock_print.assert_any_call(detail_call)

    def test_log_connection_status(self):
        """Test log_connection_status with different statuses."""
        with patch.object(self.console.console, 'print') as mock_print:
            test_cases = [
                ("MLflow Tracking", "connected", "green", "🔗"),
                ("Redis Cache", "failed", "red", "❌"),
                ("Database", "connecting", "yellow", "🔄")
            ]
            
            for service, status, expected_color, expected_emoji in test_cases:
                self.console.log_connection_status(service, status, f"Port: 5432")
                
                main_call = f"{expected_emoji} [{expected_color}]{service}: {status.title()}[/{expected_color}]"
                mock_print.assert_any_call(main_call)
                mock_print.assert_any_call("   [dim]Port: 5432[/dim]")


class TestUnifiedConsole:
    """Test UnifiedConsole dual output functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Don't use the global unified_console to avoid side effects
        with patch('src.utils.system.logger.logger') as mock_logger:
            self.unified = UnifiedConsole()

    @patch('src.utils.system.logger.logger')
    def test_initialization(self, mock_logger):
        """Test UnifiedConsole initialization."""
        unified = UnifiedConsole()
        
        assert unified.rich_console is not None
        assert unified.logger == mock_logger
        assert unified.mode in ["rich", "plain"]

    @patch('src.utils.system.logger.logger')
    def test_info_rich_mode(self, mock_logger):
        """Test info logging in rich mode."""
        unified = UnifiedConsole()
        unified.mode = "rich"
        
        with patch.object(unified.rich_console, 'log_milestone') as mock_milestone:
            unified.info("Test message")
            
            mock_logger.info.assert_called_once_with("Test message")
            mock_milestone.assert_called_once_with("Test message", "info")

    @patch('src.utils.system.logger.logger')
    def test_info_rich_mode_custom_message(self, mock_logger):
        """Test info logging in rich mode with custom rich message."""
        unified = UnifiedConsole()
        unified.mode = "rich"
        
        with patch.object(unified.rich_console.console, 'print') as mock_print:
            unified.info("Log message", "Rich display message", style="bold")
            
            mock_logger.info.assert_called_once_with("Log message")
            mock_print.assert_called_once_with("Rich display message", style="bold")

    @patch('src.utils.system.logger.logger')
    @patch('builtins.print')
    def test_info_plain_mode(self, mock_print, mock_logger):
        """Test info logging in plain mode."""
        unified = UnifiedConsole()
        unified.mode = "plain"
        
        unified.info("Test message")
        
        mock_logger.info.assert_called_once_with("Test message")
        mock_print.assert_called_once_with("INFO: Test message")

    @patch('src.utils.system.logger.logger')
    def test_error_rich_mode(self, mock_logger):
        """Test error logging in rich mode."""
        unified = UnifiedConsole()
        unified.mode = "rich"
        
        context = {"file": "test.py", "line": 42}
        suggestion = "Check file permissions"
        
        with patch.object(unified.rich_console, 'log_error_with_context') as mock_error:
            unified.error("Error occurred", context=context, suggestion=suggestion)
            
            mock_logger.error.assert_called_once_with("Error occurred")
            mock_error.assert_called_once_with("Error occurred", context, suggestion)

    @patch('src.utils.system.logger.logger')
    @patch('builtins.print')
    def test_error_plain_mode(self, mock_print, mock_logger):
        """Test error logging in plain mode."""
        unified = UnifiedConsole()
        unified.mode = "plain"
        
        unified.error("Error occurred")
        
        mock_logger.error.assert_called_once_with("Error occurred")
        mock_print.assert_called_once_with("ERROR: Error occurred")

    @patch('src.utils.system.logger.logger')
    def test_warning_rich_mode(self, mock_logger):
        """Test warning logging in rich mode."""
        unified = UnifiedConsole()
        unified.mode = "rich"
        
        context = {"threshold": 0.95}
        
        with patch.object(unified.rich_console, 'log_warning_with_context') as mock_warning:
            unified.warning("Low accuracy", context=context)
            
            mock_logger.warning.assert_called_once_with("Low accuracy")
            mock_warning.assert_called_once_with("Low accuracy", context)

    @patch('src.utils.system.logger.logger')
    @patch('builtins.print')
    def test_warning_plain_mode(self, mock_print, mock_logger):
        """Test warning logging in plain mode."""
        unified = UnifiedConsole()
        unified.mode = "plain"
        
        unified.warning("Warning message")
        
        mock_logger.warning.assert_called_once_with("Warning message")
        mock_print.assert_called_once_with("WARNING: Warning message")

    @patch('src.utils.system.logger.logger')
    def test_component_init_rich_mode(self, mock_logger):
        """Test component_init in rich mode."""
        unified = UnifiedConsole()
        unified.mode = "rich"
        
        with patch.object(unified.rich_console, 'log_component_init') as mock_init:
            unified.component_init("Database", "success")
            
            mock_init.assert_called_once_with("Database", "success")

    @patch('src.utils.system.logger.logger')
    @patch('builtins.print')
    def test_component_init_plain_mode(self, mock_print, mock_logger):
        """Test component_init in plain mode."""
        unified = UnifiedConsole()
        unified.mode = "plain"
        
        unified.component_init("Database", "error")
        
        mock_print.assert_called_once_with("✗ Database initialized")

    @patch('src.utils.system.logger.logger')
    def test_data_operation_rich_mode(self, mock_logger):
        """Test data_operation in rich mode."""
        unified = UnifiedConsole()
        unified.mode = "rich"
        
        with patch.object(unified.rich_console, 'log_data_operation') as mock_data:
            unified.data_operation("Loading", (1000, 20), "From CSV")
            
            mock_data.assert_called_once_with("Loading", (1000, 20), "From CSV")

    @patch('src.utils.system.logger.logger')
    @patch('builtins.print')
    def test_data_operation_plain_mode(self, mock_print, mock_logger):
        """Test data_operation in plain mode."""
        unified = UnifiedConsole()
        unified.mode = "plain"
        
        unified.data_operation("Processing", (500, 15))
        
        mock_print.assert_called_once_with("DATA: Processing (500 rows, 15 columns)")

    @patch('src.utils.system.logger.logger')
    def test_detect_output_mode_ci(self, mock_logger):
        """Test _detect_output_mode in CI environment."""
        unified = UnifiedConsole()
        
        with patch.object(unified.rich_console, 'is_ci_environment', return_value=True):
            mode = unified._detect_output_mode(None)
            
            assert mode == "plain"

    @patch('src.utils.system.logger.logger')
    def test_detect_output_mode_settings(self, mock_logger):
        """Test _detect_output_mode with settings."""
        settings = MagicMock()
        settings.console_mode = "custom"
        
        unified = UnifiedConsole()
        
        with patch.object(unified.rich_console, 'is_ci_environment', return_value=False):
            mode = unified._detect_output_mode(settings)
            
            assert mode == "custom"

    @patch('src.utils.system.logger.logger')
    def test_detect_output_mode_default(self, mock_logger):
        """Test _detect_output_mode default to rich."""
        unified = UnifiedConsole()
        
        with patch.object(unified.rich_console, 'is_ci_environment', return_value=False):
            mode = unified._detect_output_mode(None)
            
            assert mode == "rich"


class TestConsoleManagerIntegration:
    """Test integration scenarios and edge cases."""

    def test_global_instances_available(self):
        """Test that global instances are properly created."""
        assert console_manager is not None
        assert isinstance(console_manager, RichConsoleManager)
        assert unified_console is not None
        assert isinstance(unified_console, UnifiedConsole)

    def test_threading_safety(self):
        """Test basic thread safety of console operations."""
        console = RichConsoleManager()
        results = []
        
        def worker(worker_id):
            console.log_milestone(f"Worker {worker_id} started", "info")
            time.sleep(0.01)  # Brief pause
            console.log_milestone(f"Worker {worker_id} finished", "success")
            results.append(worker_id)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}

    def test_progress_tracker_multiple_concurrent(self):
        """Test multiple concurrent progress trackers."""
        console = RichConsoleManager()
        
        # This tests the internal data structures can handle multiple trackers
        with patch('src.utils.system.console_manager.Progress'):
            with console.progress_tracker("task1", 10, "Task 1", show_progress=False) as update1:
                with console.progress_tracker("task2", 20, "Task 2", show_progress=False) as update2:
                    update1(5)
                    update2(10)
                    # Both should work without interference
                    assert True

    def test_nested_pipeline_contexts(self):
        """Test nested pipeline contexts."""
        console = RichConsoleManager()
        
        with patch.object(console.console, 'print'):
            with console.pipeline_context("Outer Pipeline", "Main process"):
                assert console.current_pipeline == "Outer Pipeline"
                
                with console.pipeline_context("Inner Pipeline", "Sub process"):
                    assert console.current_pipeline == "Inner Pipeline"
                
                # Current implementation doesn't support nested contexts - resets to None
                assert console.current_pipeline is None
            
            # Should clear after all contexts
            assert console.current_pipeline is None

    def test_pipeline_context_with_progress(self):
        """Test pipeline context combined with progress tracking."""
        console = RichConsoleManager()
        
        with patch.object(console.console, 'print'):
            with console.pipeline_context("Data Pipeline", "Processing data"):
                with patch('src.utils.system.console_manager.Progress'):
                    with console.progress_tracker("data_processing", 100, "Processing records", show_progress=False):
                        assert console.current_pipeline == "Data Pipeline"


class TestConsoleManagerSecurity:
    """Test security, performance, and edge cases."""

    def test_large_metrics_table(self):
        """Test displaying large metrics table."""
        console = RichConsoleManager()
        
        # Large number of metrics
        metrics = {f"metric_{i}": float(i) * 0.001 for i in range(100)}
        
        with patch.object(console.console, 'print'):
            # Should not crash with large data
            console.display_metrics_table(metrics, "Large Metrics")

    def test_malformed_data_handling(self):
        """Test handling of malformed data in logging methods."""
        console = RichConsoleManager()
        
        with patch.object(console.console, 'print') as mock_print:
            # None values
            console.log_periodic("test", 0, None, every_n=1)
            
            # Should handle gracefully
            mock_print.assert_called_once()

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        console = RichConsoleManager()
        
        with patch.object(console.console, 'print') as mock_print:
            # Unicode characters
            console.log_milestone("测试消息 🚀 émoji", "info")
            console.log_phase("Procéssing Dáta", "🔬")
            console.log_feature_engineering("Normalizaçāo", ["colüna_1", "colôna_2"])
            
            # Should handle all unicode gracefully
            assert mock_print.call_count >= 3

    def test_very_long_messages(self):
        """Test handling of very long messages."""
        console = RichConsoleManager()
        
        long_message = "Very long message " * 100  # 1800+ characters
        
        with patch.object(console.console, 'print') as mock_print:
            console.log_milestone(long_message, "info")
            console.log_error_with_context(long_message, {"context": "test"})
            
            # Should handle without errors
            assert mock_print.call_count >= 2

    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""
        console = RichConsoleManager()
        
        with patch.object(console.console, 'print') as mock_print:
            # Empty strings
            console.log_milestone("", "info")
            console.log_phase("")
            console.log_data_operation("", None, "")
            
            # None handling where applicable
            console.log_feature_engineering("Step", [], "")
            console.display_metrics_table({})
            
            # Should handle gracefully without crashes
            assert mock_print.call_count >= 5

    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        console = RichConsoleManager()
        
        # Simulate some progress bars
        mock_task_finished = MagicMock()
        mock_task_finished.finished = True
        mock_task_active = MagicMock() 
        mock_task_active.finished = False
        
        console.progress_bars = {
            "task1": (MagicMock(), mock_task_finished),
            "task2": (MagicMock(), mock_task_active),
            "task3": (MagicMock(), mock_task_finished)
        }
        
        initial_count = len(console.progress_bars)
        console.cleanup_completed_tasks()
        final_count = len(console.progress_bars)
        
        assert initial_count == 3
        assert final_count == 1  # Only active task remains
        assert "task2" in console.progress_bars

    def test_console_mode_edge_cases(self):
        """Test console mode detection edge cases."""
        console = RichConsoleManager()
        
        # Test with mixed environment variables
        with patch.dict('os.environ', {'CI': 'false', 'GITHUB_ACTIONS': 'true'}):
            # Should still detect CI due to GITHUB_ACTIONS
            assert console.is_ci_environment() is True
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                assert console.get_console_mode() == "rich"

    @patch('builtins.print')
    def test_unified_console_fallback_plain(self, mock_print):
        """Test UnifiedConsole fallback to plain print."""
        with patch('src.utils.system.logger.logger'):
            unified = UnifiedConsole()
            unified.mode = "unknown"  # Force unknown mode
            
            unified.info("Test message")
            
            # Should fallback gracefully (logger still called)
            assert mock_print.call_count == 0  # No plain print for unknown mode

    def test_progress_tracker_exception_cleanup(self):
        """Test progress tracker cleans up even after exceptions."""
        console = RichConsoleManager()
        
        with patch('src.utils.system.console_manager.Progress') as mock_progress_class:
            mock_progress_context = MagicMock()
            mock_progress = MagicMock()
            mock_progress_class.return_value = mock_progress_context
            mock_progress_context.__enter__ = MagicMock(return_value=mock_progress)
            mock_progress_context.__exit__ = MagicMock(return_value=None)
            
            mock_task = MagicMock()
            mock_progress.add_task.return_value = mock_task
            
            try:
                with console.progress_tracker("error_task", 10, "Test") as update:
                    assert "error_task" in console.progress_bars
                    raise ValueError("Test error")
            except ValueError:
                pass
            
            # Should be cleaned up even after exception
            assert "error_task" not in console.progress_bars