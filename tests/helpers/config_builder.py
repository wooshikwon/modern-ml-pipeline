from typing import Dict, Any
from datetime import datetime
from pathlib import Path
from src.settings import Settings
from src.settings.config import Config
from .recipe_builder import RecipeBuilder


class ConfigBuilder:
    """Builder for creating Config objects for testing."""

    @staticmethod
    def build(
        env_name: str = "test",
        mlflow_tracking_uri: str = "./mlruns",
        adapter_type: str = "storage",
        feature_store_provider: str = "none",
        **overrides: Dict[str, Any]
    ) -> Config:
        config_dict: Dict[str, Any] = {
            'environment': {'name': env_name},
            'mlflow': {
                'tracking_uri': mlflow_tracking_uri,
                'experiment_name': f'{env_name}_experiment'
            },
            'data_source': {
                'name': f'{env_name}_datasource',
                'adapter_type': adapter_type,
                'config': {'base_path': './data'}
            },
            'feature_store': {'provider': feature_store_provider}
        }

        for key, value in overrides.items():
            if '.' in key:
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            else:
                config_dict[key] = value

        return Config(**config_dict)


class SettingsBuilder:
    """Builder for creating Settings objects for testing."""

    @staticmethod
    def build(
        env_name: str = "test",
        recipe_name: str = "test_recipe",
        **overrides: Dict[str, Any]
    ) -> Settings:
        # Separate config, recipe, and direct recipe parameters
        config_overrides = overrides.get('config', {})
        recipe_overrides = overrides.get('recipe', {})
        
        # Handle direct recipe parameters (for CLI-first architecture compatibility)
        # These should be forwarded to RecipeBuilder instead of being lost
        recipe_direct_params = {}
        for key, value in overrides.items():
            if key not in ['config', 'recipe'] and not key.startswith('config.') and not key.startswith('recipe.'):
                # Direct parameters that should go to RecipeBuilder
                if key in ['source_uri', 'task_choice', 'model_class_path', 'fetcher_type', 'enable_tuning',
                          'target_column', 'entity_columns', 'timestamp_column', 'feature_columns']:
                    recipe_direct_params[key] = value
        
        # Handle dotted notation for recipe (recipe.data.loader.source_uri format)
        for key, value in overrides.items():
            if key.startswith('recipe.'):
                recipe_key = key[7:]  # Remove 'recipe.' prefix
                recipe_overrides[recipe_key] = value
        
        # Create builders with appropriate parameters
        config = ConfigBuilder.build(env_name=env_name, **config_overrides)
        recipe = RecipeBuilder.build(name=recipe_name, **recipe_overrides, **recipe_direct_params)
        settings = Settings(config=config, recipe=recipe)
        
        # Add computed fields like production (_add_computed_fields logic)
        SettingsBuilder._add_computed_fields(settings, recipe_name)
        
        return settings

    @staticmethod
    def _add_computed_fields(settings: Settings, recipe_name: str) -> None:
        """
        Add computed fields to settings (mirrors src.settings.loader._add_computed_fields).
        This ensures test Settings objects have the same structure as production.
        """
        # Initialize computed fields if not present
        if not settings.recipe.model.computed:
            settings.recipe.model.computed = {}
        
        # Generate run_name if not already present
        if "run_name" not in settings.recipe.model.computed:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{recipe_name}_{timestamp}"
            settings.recipe.model.computed["run_name"] = run_name
        
        # Add environment info
        settings.recipe.model.computed["environment"] = settings.config.environment.name
        
        # Add tuning info if enabled
        if hasattr(settings.recipe, 'is_tuning_enabled') and settings.recipe.is_tuning_enabled():
            settings.recipe.model.computed["tuning_enabled"] = True
            if hasattr(settings.recipe, 'get_tunable_params'):
                tunable = settings.recipe.get_tunable_params()
                if tunable:
                    settings.recipe.model.computed["tunable_param_count"] = len(tunable)
        else:
            settings.recipe.model.computed["tuning_enabled"] = False
            
        # Add seed if not present (for reproducibility)
        if "seed" not in settings.recipe.model.computed:
            settings.recipe.model.computed["seed"] = 42  # Default test seed

    @staticmethod
    def build_classification_config() -> Settings:
        """Build a Settings object configured for classification tasks."""
        return SettingsBuilder.build(
            recipe_name="test_classification",
            recipe={
                "task_choice": "classification",
                "data.data_interface.target_column": "target",
                "data.data_interface.entity_columns": ["user_id"]
            }
        )
    
    @staticmethod
    def build_regression_config() -> Settings:
        """Build a Settings object configured for regression tasks."""
        return SettingsBuilder.build(
            recipe_name="test_regression",
            recipe={
                "task_choice": "regression",
                "data.data_interface.target_column": "target",
                "data.data_interface.entity_columns": ["user_id"]
            }
        )
    
    @staticmethod
    def build_clustering_config() -> Settings:
        """Build a Settings object configured for clustering tasks."""
        return SettingsBuilder.build(
            recipe_name="test_clustering",
            recipe={
                "task_choice": "clustering",
                "data.data_interface.target_column": "cluster_label",
                "data.data_interface.entity_columns": ["user_id"]
            }
        )
    
    @staticmethod
    def build_causal_config() -> Settings:
        """Build a Settings object configured for causal tasks."""
        return SettingsBuilder.build(
            recipe_name="test_causal",
            recipe={
                "task_choice": "causal",
                "data.data_interface.target_column": "outcome",
                "data.data_interface.treatment_column": "treatment",
                "data.data_interface.entity_columns": ["user_id"]
            }
        )
    
    @staticmethod
    def build_config_with_exclusions(
        entity_columns=None, 
        timestamp_column=None
    ) -> Settings:
        """Build config with specified entity/timestamp column exclusions."""
        overrides = {}
        if entity_columns:
            overrides["data.data_interface.entity_columns"] = entity_columns
        if timestamp_column:
            overrides["data.fetcher.timestamp_column"] = timestamp_column
            
        return SettingsBuilder.build(
            recipe_name="test_exclusions",
            recipe=overrides
        )
    
    
    @staticmethod
    def build_config_with_auto_features(task_choice: str) -> Settings:
        """Build config with feature_columns=None for auto selection."""
        base_config = {
            "task_choice": task_choice,
            "data.data_interface.feature_columns": None,  # Auto selection
            "data.data_interface.entity_columns": ["user_id"]
        }
        
        if task_choice == "classification":
            base_config["data.data_interface.target_column"] = "target"
        elif task_choice == "regression":
            base_config["data.data_interface.target_column"] = "target"
        elif task_choice == "clustering":
            pass  # No target for clustering
        elif task_choice == "causal":
            base_config["data.data_interface.target_column"] = "outcome"
            base_config["data.data_interface.treatment_column"] = "treatment"
        
        return SettingsBuilder.build(
            recipe_name=f"test_{task_choice}_auto",
            recipe=base_config
        )
    
    @staticmethod
    def build_config_with_explicit_features(
        task_choice: str, 
        feature_columns: list
    ) -> Settings:
        """Build config with explicitly specified feature columns."""
        base_config = {
            "task_choice": task_choice,
            "data.data_interface.feature_columns": feature_columns,
            "data.data_interface.entity_columns": ["user_id"]
        }
        
        if task_choice == "classification":
            base_config["data.data_interface.target_column"] = "target"
        elif task_choice == "regression":
            base_config["data.data_interface.target_column"] = "target"
        elif task_choice == "causal":
            base_config["data.data_interface.target_column"] = "outcome"
            base_config["data.data_interface.treatment_column"] = "treatment"
        
        return SettingsBuilder.build(
            recipe_name=f"test_{task_choice}_explicit",
            recipe=base_config
        )
    
    @staticmethod
    def build_tuning_enabled_config(
        task_choice: str = "classification",
        n_trials: int = 10,
        timeout: int = 300
    ) -> Settings:
        """Build config with Optuna tuning enabled."""
        return SettingsBuilder.build(
            recipe_name=f"test_{task_choice}_tuning",
            recipe={
                "task_choice": task_choice,
                "data.data_interface.target_column": "target",
                "data.data_interface.entity_columns": ["user_id"],
                "model.hyperparameters.tuning_enabled": True,
                "model.hyperparameters.optimization_metric": "accuracy",
                "model.hyperparameters.n_trials": n_trials,
                "model.hyperparameters.timeout": timeout
            },
            config={},
            enable_tuning=True
        )
    
    @staticmethod
    def build_tuning_disabled_config(task_choice: str = "classification") -> Settings:
        """Build config with Optuna tuning disabled."""
        return SettingsBuilder.build(
            recipe_name=f"test_{task_choice}_no_tuning",
            recipe={
                "task_choice": task_choice,
                "data.data_interface.target_column": "target",
                "data.data_interface.entity_columns": ["user_id"]
            },
            config={},
            enable_tuning=False
        )
