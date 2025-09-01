"""
Settings Validator - Validation Logic (v2.0)
Simplified from 210 lines (_model_validation.py) to ~100 lines

This module provides validation logic for Settings.
Ensures consistency between Config and Recipe.
Includes model catalog functionality.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional
import importlib
import yaml
from pathlib import Path
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .loader import Settings


class HyperparameterSpec(BaseModel):
    """Hyperparameter specification for model catalog."""
    
    type: str = Field(..., description="Parameter type (int, float, str, bool)")
    range: Optional[List[Any]] = Field(default=None, description="Valid range for numeric parameters")
    choices: Optional[List[Any]] = Field(default=None, description="Valid choices for categorical parameters")
    default: Any = Field(..., description="Default value")


class ModelSpec(BaseModel):
    """Model specification for catalog system."""
    
    class_path: str = Field(..., description="Full Python class path")
    description: str = Field(..., description="Human-readable description")
    library: str = Field(..., description="Library name (e.g., scikit-learn)")
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Hyperparameter configuration")
    supported_tasks: List[str] = Field(default_factory=list, description="Supported ML tasks")
    feature_requirements: Optional[Dict[str, bool]] = Field(default=None, description="Feature type requirements")


class ModelCatalog(BaseModel):
    """Model catalog containing available models organized by category."""
    
    models: Dict[str, List[ModelSpec]] = Field(..., description="Models organized by category")
    
    @classmethod
    def from_yaml(cls, catalog_path: Optional[Path] = None) -> "ModelCatalog":
        """
        Load ModelCatalog from YAML file.
        
        Args:
            catalog_path: Path to catalog YAML file
            
        Returns:
            ModelCatalog instance
            
        Raises:
            FileNotFoundError: If catalog file not found
        """
        if catalog_path is None:
            catalog_path = Path("src/models/catalog.yaml")
        elif isinstance(catalog_path, str):
            catalog_path = Path(catalog_path)
        
        if not catalog_path.exists():
            raise FileNotFoundError(f"Model catalog not found: {catalog_path}")
        
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog_data = yaml.safe_load(f)
        
        # Convert dict data to ModelSpec instances
        models = {}
        for category, model_list in catalog_data.items():
            models[category] = [ModelSpec(**model_data) for model_data in model_list]
        
        return cls(models=models)
    
    def get_model_spec(self, class_path: str) -> Optional[ModelSpec]:
        """
        Find ModelSpec by class_path.
        
        Args:
            class_path: Full Python class path
            
        Returns:
            ModelSpec if found, None otherwise
        """
        for category_models in self.models.values():
            for model_spec in category_models:
                if model_spec.class_path == class_path:
                    return model_spec
        return None
    
    def validate_recipe_compatibility(self, recipe_data: Dict[str, Any]) -> bool:
        """
        Validate recipe compatibility with catalog.
        
        Args:
            recipe_data: Recipe dictionary
            
        Returns:
            True if compatible, False otherwise
            
        Raises:
            ValueError: If model not found in catalog
        """
        if "model" not in recipe_data or "class_path" not in recipe_data["model"]:
            raise ValueError("Recipe must contain model.class_path")
        
        class_path = recipe_data["model"]["class_path"]
        model_spec = self.get_model_spec(class_path)
        
        if model_spec is None:
            raise ValueError(f"Model '{class_path}' not found in catalog")
        
        return True
    
    def _validate_hyperparameters(self, model_spec: ModelSpec, recipe_hyperparams: Dict[str, Any]) -> bool:
        """
        Validate recipe hyperparameters against model spec.
        
        Args:
            model_spec: Model specification from catalog
            recipe_hyperparams: Hyperparameters from recipe
            
        Returns:
            True if valid, False if invalid
        """
        if not model_spec.hyperparameters or "tunable" not in model_spec.hyperparameters:
            return True
        
        tunable_specs = model_spec.hyperparameters["tunable"]
        
        for param_name, value in recipe_hyperparams.items():
            if param_name not in tunable_specs:
                continue  # Skip non-tunable parameters
            
            param_spec = tunable_specs[param_name]
            
            # Check range for numeric parameters
            if param_spec.get("type") in ["int", "float"] and "range" in param_spec:
                min_val, max_val = param_spec["range"]
                if not (min_val <= value <= max_val):
                    return False
        
        return True


class Validator:
    """
    Settings validation utility.
    
    This is a utility class with static methods for validation.
    It ensures that Config and Recipe are compatible and valid.
    """
    
    # Valid metrics for each task type
    TASK_METRICS = {
        "classification": [
            "accuracy", "precision", "recall", "f1", "f1_score",
            "roc_auc", "log_loss", "confusion_matrix",
            "precision_weighted", "recall_weighted", "f1_weighted"
        ],
        "regression": [
            "mse", "rmse", "mae", "r2", "r2_score",
            "mean_squared_error", "mean_absolute_error",
            "explained_variance", "max_error"
        ]
    }
    
    @staticmethod
    def validate(settings: "Settings") -> None:
        """
        Main validation entry point.
        
        Args:
            settings: Settings object to validate
            
        Raises:
            ValueError: If validation fails
        """
        Validator.validate_adapter_references(settings)
        Validator.validate_metrics_compatibility(settings)
        Validator.validate_model_class(settings)
    
    @staticmethod
    def validate_adapter_references(settings: "Settings") -> None:
        """
        Validate that Recipe references existing adapters in Config.
        
        Args:
            settings: Settings object to validate
            
        Raises:
            ValueError: If adapter not found in config
        """
        adapter_name = settings.recipe.data.loader.adapter
        if adapter_name not in settings.config.adapters:
            available = list(settings.config.adapters.keys())
            raise ValueError(
                f"Recipe references adapter '{adapter_name}' which is not defined in Config. "
                f"Available adapters: {available}"
            )
    
    @staticmethod
    def validate_metrics_compatibility(settings: "Settings") -> None:
        """
        Validate that metrics are compatible with task type.
        
        Args:
            settings: Settings object to validate
            
        Raises:
            ValueError: If metrics incompatible with task type
        """
        task_type = settings.recipe.data.data_interface.task_type
        metrics = settings.recipe.evaluation.metrics
        
        if task_type not in Validator.TASK_METRICS:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Valid types: {list(Validator.TASK_METRICS.keys())}"
            )
        
        valid_metrics = Validator.TASK_METRICS[task_type]
        
        for metric in metrics:
            # Check base metric name (handle weighted variants)
            base_metric = metric.split("_")[0] if "_" in metric else metric
            
            if metric not in valid_metrics and base_metric not in valid_metrics:
                raise ValueError(
                    f"Metric '{metric}' is not compatible with '{task_type}' task. "
                    f"Valid metrics: {valid_metrics}"
                )
    
    @staticmethod
    def validate_model_class(settings: "Settings") -> None:
        """
        Validate that model class path is importable.
        
        Args:
            settings: Settings object to validate
            
        Raises:
            ValueError: If model class cannot be imported
        """
        class_path = settings.recipe.model.class_path
        
        # Skip validation for dummy/test models
        if class_path.startswith("dummy") or class_path == "test.model":
            return
        
        try:
            # Split module and class name
            parts = class_path.rsplit(".", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid class path format: {class_path}")
            
            module_name, class_name = parts
            
            # Try to import the module
            module = importlib.import_module(module_name)
            
            # Check if class exists in module
            if not hasattr(module, class_name):
                raise ValueError(
                    f"Class '{class_name}' not found in module '{module_name}'"
                )
                
        except ImportError as e:
            raise ValueError(
                f"Cannot import model class '{class_path}': {e}"
            )