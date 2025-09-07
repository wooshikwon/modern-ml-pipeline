# Console Output Refactoring Plan
**Rich Console Integration for Unified Logging System**

## Overview
Complete refactoring of console output across the entire ML pipeline system using Rich library for unified, hierarchical, and professional logging with progress tracking.

## Core Architecture

### RichConsoleManager (New Component)
**Location**: `src/utils/system/console_manager.py`

```python
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from rich.table import Table
from rich.text import Text
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import threading
import time

class RichConsoleManager:
    def __init__(self):
        self.console = Console()
        self.current_pipeline = None
        self.progress_bars = {}
        self.iteration_counters = {}
        self.live_displays = {}
    
    @contextmanager
    def pipeline_context(self, name: str, description: str):
        # Pipeline-level context manager with hierarchical display
        
    @contextmanager 
    def progress_tracker(self, task_id: str, total: int, description: str):
        # Progress bar context for iterative processes
        
    def log_periodic(self, process_id: str, iteration: int, data: Dict[str, Any], every_n: int = 10):
        # Periodic output for high-iteration processes like Optuna
        
    def log_milestone(self, message: str, level: str = "info"):
        # Important milestone logging with Rich formatting
```

### Integration Points

#### 1. Pipeline Level Integration
**Files to Modify**:
- `src/pipelines/train_pipeline.py`
- `src/pipelines/batch_inference_pipeline.py`
- `src/pipelines/api_pipeline.py`

**Implementation Pattern**:
```python
from src.utils.system.console_manager import RichConsoleManager

def run_train_pipeline(settings, context_params=None):
    console = RichConsoleManager()
    
    with console.pipeline_context("Training Pipeline", f"Task: {settings.recipe.data.data_interface.task_type}"):
        console.log_milestone("🚀 Starting training pipeline", "info")
        
        # Data loading phase
        with console.progress_tracker("data_loading", 100, "Loading and preparing data"):
            # Existing data loading code with progress updates
            
        # Training phase  
        with console.progress_tracker("model_training", 100, "Training model"):
            # Existing training code with progress updates
            
        # Evaluation phase
        console.log_milestone("📊 Model evaluation completed", "success")
```

#### 2. Hyperparameter Optimization Integration
**File**: `src/components/trainer/modules/optimizer.py`

**Current Issue**: Uses print statements and basic logging
**Solution**: Progress bars + periodic detailed outputs

```python
class OptunaOptimizer:
    def __init__(self, settings, factory_provider):
        self.console = RichConsoleManager()
        
    def optimize(self, train_df, training_func):
        n_trials = self.settings.recipe.model.hyperparameters.n_trials
        
        with self.console.progress_tracker("hyperopt", n_trials, "Hyperparameter Optimization"):
            def objective(trial):
                # Existing objective function
                trial_result = training_func(train_df, params, seed)
                score = trial_result['score']
                
                # Periodic detailed output every 10 trials
                self.console.log_periodic(
                    "optuna_trials", 
                    trial.number, 
                    {
                        "trial": trial.number,
                        "score": score,
                        "params": trial.params,
                        "best_score": study.best_value if hasattr(study, 'best_value') else None
                    },
                    every_n=10
                )
                
                return score
```

**Output Format**:
```
🎯 Starting hyperparameter optimization: 100 trials

Trial 10/100: score=0.8234 (best: 0.8456) | n_estimators=150, max_depth=8, learning_rate=0.1
Trial 20/100: score=0.8445 (best: 0.8456) 🔥 | n_estimators=200, max_depth=6, learning_rate=0.05
Trial 30/100: score=0.8123 (best: 0.8456) | n_estimators=100, max_depth=10, learning_rate=0.2
```

#### 3. MLflow Integration
**Files**: 
- `src/utils/integrations/mlflow_integration.py`
- `src/components/trainer/trainer.py`

**Current Issue**: Basic logging without progress indication
**Solution**: Progress tracking for model logging and artifact upload

```python
class MLflowIntegration:
    def __init__(self):
        self.console = RichConsoleManager()
    
    def log_model_with_progress(self, model, model_path, metadata):
        artifacts = ["model", "metadata", "preprocessing", "evaluation_results"]
        
        with self.console.progress_tracker("mlflow_logging", len(artifacts), "Logging to MLflow"):
            for artifact in artifacts:
                # Log each artifact with progress update
                self.console.log_milestone(f"✅ Logged {artifact}", "success")
    
    def track_experiment_metrics(self, metrics: Dict[str, float]):
        # Formatted metric display
        metrics_table = Table(title="Experiment Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta")
        
        for metric, value in metrics.items():
            metrics_table.add_row(metric, f"{value:.4f}")
            
        self.console.console.print(metrics_table)
```

#### 4. Data Processing Integration
**Files**:
- `src/components/datahandler/modules/tabular_handler.py`
- `src/components/datahandler/modules/timeseries_handler.py`
- `src/components/fetcher/fetcher.py`

**Implementation**:
```python
class TabularDataHandler:
    def __init__(self, settings):
        self.console = RichConsoleManager()
    
    def split_data(self, df):
        self.console.log_milestone(f"📊 Splitting data: {len(df)} rows", "info")
        
        with self.console.progress_tracker("data_split", 100, "Splitting train/test"):
            # Existing split logic with progress updates
            
        self.console.log_milestone(f"✅ Split complete: {len(train_df)} train, {len(test_df)} test", "success")
        
    def prepare_data(self, df):
        with self.console.progress_tracker("data_prep", 100, "Preparing features and targets"):
            # Existing prepare logic with progress updates
```

#### 5. Model Training Integration
**Files**:
- `src/models/custom/ft_transformer.py`
- `src/components/trainer/trainer.py`

**Implementation**:
```python
class Trainer:
    def train(self, df, model, fetcher, datahandler, preprocessor, evaluator, context_params=None):
        console = RichConsoleManager()
        
        # Training phases with progress tracking
        phases = ["data_preparation", "preprocessing", "model_training", "evaluation"]
        
        for phase in phases:
            with console.progress_tracker(phase, 100, phase.replace("_", " ").title()):
                if phase == "model_training":
                    # For iterative models, show epoch progress
                    if hasattr(model, 'n_epochs') and model.n_epochs > 1:
                        with console.progress_tracker("epochs", model.n_epochs, "Training Epochs"):
                            # Training loop with epoch progress
                            pass
```

#### 6. External Library Output Capture
**New Component**: `src/utils/system/output_capture.py`

```python
import sys
import re
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

class ExternalLibraryOutputCapture:
    """Capture and format output from external libraries like sklearn, optuna"""
    
    def __init__(self, console_manager):
        self.console = console_manager
        
    @contextmanager
    def capture_sklearn_output(self):
        # Capture sklearn verbose output and format with Rich
        
    @contextmanager  
    def capture_optuna_output(self):
        # Capture Optuna study progress and format with Rich
```

## Hierarchical Output Structure

### Level 1: Pipeline Context
```
🚀 Training Pipeline
Environment: local | Task: classification | Model: RandomForest

📊 Phase 1: Data Loading
[████████████████████████████████████] 1,000/1,000 rows 100%
✅ Data loaded successfully

🤖 Phase 2: Model Training
[██████████████████████            ] 15/25 epochs 60%

🎯 Hyperparameter Optimization
Trial 40/100: score=0.8234 (best: 0.8456) | n_estimators=150, max_depth=8
Trial 50/100: score=0.8445 (best: 0.8456) 🔥 | n_estimators=200, max_depth=6
```

### Level 2: Component-Specific Progress
```
📤 MLflow Experiment Tracking
[████████████████████████████████] 5/5 artifacts 100%

✅ Model logged
✅ Metadata saved
✅ Preprocessing pipeline stored
✅ Evaluation results uploaded
✅ Experiment tags updated

🎯 Run ID: abc123def456
🔗 MLflow URI: http://localhost:5000/#/experiments/1/runs/abc123
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create `RichConsoleManager` class
- [ ] Implement basic progress tracking
- [ ] Add milestone logging
- [ ] Create output capture utilities

### Phase 2: Pipeline Integration (Week 2)
- [ ] Integrate with `train_pipeline.py`
- [ ] Add data loading progress tracking
- [ ] Implement model training progress
- [ ] MLflow integration with progress bars

### Phase 3: Advanced Features (Week 3)
- [ ] Hyperparameter optimization progress bars
- [ ] Periodic output for high-iteration processes
- [ ] External library output capture
- [ ] Error handling and graceful degradation

### Phase 4: Batch & API Pipeline Integration (Week 4)
- [ ] Batch inference pipeline integration
- [ ] API pipeline status display
- [ ] Real-time inference monitoring
- [ ] Comprehensive testing

## Configuration Integration

### Settings Schema Updates
**File**: `src/settings/recipe.py`

```python
class ConsoleConfig(BaseModel):
    """Console output configuration"""
    enabled: bool = Field(True, description="Enable Rich console output")
    progress_bars: bool = Field(True, description="Show progress bars")
    periodic_output_interval: int = Field(10, description="Show detailed output every N iterations")
    color_scheme: str = Field("auto", description="Color scheme: auto, light, dark")
    verbose_level: int = Field(1, description="Verbosity level 0-3")
```

### Environment-Specific Behavior
```python
# Local environment: Full Rich output with colors and animations
# Production environment: Minimal output, structured logging
# CI/CD environment: Plain text output, no animations

def get_console_config(env_name: str) -> ConsoleConfig:
    if env_name == "local":
        return ConsoleConfig(enabled=True, progress_bars=True, verbose_level=2)
    elif env_name == "prod":
        return ConsoleConfig(enabled=False, progress_bars=False, verbose_level=0)
    else:  # dev, test
        return ConsoleConfig(enabled=True, progress_bars=True, verbose_level=1)
```

## Backward Compatibility

### Logger Integration
**File**: `src/utils/system/logger.py`

```python
def setup_logging(settings: "Settings"):
    # Existing logger setup
    
    # Rich console integration
    if hasattr(settings, 'console') and settings.console.enabled:
        from .console_manager import RichConsoleManager
        console = RichConsoleManager()
        
        # Add Rich handler to logger
        rich_handler = RichHandler(console=console.console)
        root_logger.addHandler(rich_handler)
```

### Gradual Migration Strategy
1. **Week 1-2**: Add Rich alongside existing logging (no breaking changes)
2. **Week 3-4**: Gradually replace print statements with Rich output
3. **Week 5-6**: Remove old logging patterns, full Rich integration
4. **Week 7+**: Performance optimization and advanced features

## Performance Considerations

### Resource Management
```python
class RichConsoleManager:
    def __init__(self):
        self.max_concurrent_progress_bars = 5
        self.output_buffer_size = 1000
        self.refresh_rate = 10  # Hz
        
    def cleanup_completed_tasks(self):
        # Remove completed progress bars to save memory
        
    def throttle_output(self, min_interval: float = 0.1):
        # Prevent excessive console updates
```

### CI/CD Environment Detection
```python
def is_ci_environment() -> bool:
    return any(env in os.environ for env in ['CI', 'GITHUB_ACTIONS', 'JENKINS_URL'])

def get_console_mode() -> str:
    if is_ci_environment():
        return "plain"  # No colors, no animations
    elif not sys.stdout.isatty():
        return "plain"  # Pipe/redirect detected
    else:
        return "rich"   # Full Rich experience
```

## Testing Strategy

### Unit Tests
**File**: `tests/unit/utils/test_console_manager.py`
```python
def test_progress_tracker_context_manager():
    # Test progress bar creation and cleanup
    
def test_periodic_output_timing():
    # Test every_n iteration output
    
def test_hierarchical_display():
    # Test nested progress contexts
```

### Integration Tests  
**File**: `tests/integration/test_console_integration.py`
```python
def test_full_pipeline_console_output():
    # Test complete pipeline with Rich output
    
def test_optuna_progress_tracking():
    # Test hyperparameter optimization progress
```

## Migration Checklist

### Files to Modify
- [ ] `src/utils/system/console_manager.py` (NEW)
- [ ] `src/utils/system/output_capture.py` (NEW)
- [ ] `src/utils/system/logger.py` (MODIFY)
- [ ] `src/pipelines/train_pipeline.py` (MODIFY)
- [ ] `src/pipelines/batch_inference_pipeline.py` (MODIFY)
- [ ] `src/components/trainer/modules/optimizer.py` (MODIFY)
- [ ] `src/utils/integrations/mlflow_integration.py` (MODIFY)
- [ ] `src/components/datahandler/modules/*.py` (MODIFY)
- [ ] `src/settings/recipe.py` (MODIFY - add ConsoleConfig)

### Dependencies to Add
```bash
# Add to pyproject.toml
rich = "^13.0.0"
```

### Environment Variables
```bash
# New environment variables for console control
RICH_CONSOLE_ENABLED=true
RICH_PROGRESS_BARS=true
RICH_PERIODIC_INTERVAL=10
RICH_COLOR_SCHEME=auto
RICH_VERBOSE_LEVEL=1
```

## Success Metrics

### User Experience Goals
1. **Clarity**: Users immediately understand pipeline progress and status
2. **Engagement**: Rich, informative output maintains user attention
3. **Debugging**: Enhanced error display with context and suggestions
4. **Performance**: No significant performance impact on pipeline execution

### Technical Metrics
1. **Progress Accuracy**: Progress bars reflect actual completion percentage
2. **Memory Usage**: Rich console uses <10MB additional memory
3. **Output Performance**: Console updates don't slow pipeline by >5%
4. **Error Handling**: Graceful degradation when Rich features unavailable

## Future Enhancements

### Advanced Features (Post-MVP)
1. **Interactive Console**: Arrow key navigation, real-time controls
2. **Web Dashboard**: Rich output streamed to web interface
3. **Notification System**: Desktop/Slack notifications for long-running processes
4. **Custom Themes**: User-configurable color schemes and layouts
5. **Export Capabilities**: Save Rich output as HTML/PNG for reports

### Integration Opportunities
1. **Jupyter Notebook**: Rich output in notebook environments
2. **Docker Logs**: Structured Rich output in containerized environments
3. **Monitoring Integration**: Send Rich metrics to monitoring systems
4. **Documentation**: Auto-generate docs from Rich pipeline outputs

---

This comprehensive plan provides a complete roadmap for integrating Rich Console across the entire ML pipeline system, with specific attention to progress bars and periodic outputs for iterative processes as requested.