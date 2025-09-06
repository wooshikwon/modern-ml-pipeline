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
