from typing import Dict, Any
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
        config = ConfigBuilder.build(env_name=env_name, **overrides.get('config', {}))
        recipe = RecipeBuilder.build(name=recipe_name, **overrides.get('recipe', {}))
        return Settings(config=config, recipe=recipe)

    @staticmethod
    def build_classification_config() -> Settings:
        """Build a Settings object configured for classification tasks."""
        return SettingsBuilder.build(
            recipe_name="test_classification",
            recipe={
                "data.data_interface.task_type": "classification",
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
                "data.data_interface.task_type": "regression",
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
                "data.data_interface.task_type": "clustering",
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
                "data.data_interface.task_type": "causal",
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
    def build_config_with_auto_features(task_type: str) -> Settings:
        """Build config with feature_columns=None for auto selection."""
        base_config = {
            "data.data_interface.task_type": task_type,
            "data.data_interface.feature_columns": None,  # Auto selection
            "data.data_interface.entity_columns": ["user_id"]
        }
        
        if task_type == "classification":
            base_config["data.data_interface.target_column"] = "target"
        elif task_type == "regression":
            base_config["data.data_interface.target_column"] = "target"
        elif task_type == "clustering":
            pass  # No target for clustering
        elif task_type == "causal":
            base_config["data.data_interface.target_column"] = "outcome"
            base_config["data.data_interface.treatment_column"] = "treatment"
        
        return SettingsBuilder.build(
            recipe_name=f"test_{task_type}_auto",
            recipe=base_config
        )
    
    @staticmethod
    def build_config_with_explicit_features(
        task_type: str, 
        feature_columns: list
    ) -> Settings:
        """Build config with explicitly specified feature columns."""
        base_config = {
            "data.data_interface.task_type": task_type,
            "data.data_interface.feature_columns": feature_columns,
            "data.data_interface.entity_columns": ["user_id"]
        }
        
        if task_type == "classification":
            base_config["data.data_interface.target_column"] = "target"
        elif task_type == "regression":
            base_config["data.data_interface.target_column"] = "target"
        elif task_type == "causal":
            base_config["data.data_interface.target_column"] = "outcome"
            base_config["data.data_interface.treatment_column"] = "treatment"
        
        return SettingsBuilder.build(
            recipe_name=f"test_{task_type}_explicit",
            recipe=base_config
        )
    
    @staticmethod
    def build_tuning_enabled_config(
        task_type: str = "classification",
        n_trials: int = 10,
        timeout: int = 300
    ) -> Settings:
        """Build config with Optuna tuning enabled."""
        return SettingsBuilder.build(
            recipe_name=f"test_{task_type}_tuning",
            recipe={
                "data.data_interface.task_type": task_type,
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
    def build_tuning_disabled_config(task_type: str = "classification") -> Settings:
        """Build config with Optuna tuning disabled."""
        return SettingsBuilder.build(
            recipe_name=f"test_{task_type}_no_tuning",
            recipe={
                "data.data_interface.task_type": task_type,
                "data.data_interface.target_column": "target",
                "data.data_interface.entity_columns": ["user_id"]
            },
            config={},
            enable_tuning=False
        )
