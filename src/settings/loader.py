"""
Settings Loader - Unified Loading Logic (v2.0)
Simplified from 192 lines (_builder.py) + 129 lines (loaders.py) to ~150 lines

This module provides the simplified loading logic for Settings.
Reduces the 7-step process to 3 simple steps.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import os
from datetime import datetime

from .config import Config
from .recipe import Recipe
from src.utils.system.logger import logger


class Settings:
    """
    Unified Settings container (v2.0).
    
    Holds both Config (infrastructure) and Recipe (workflow) settings.
    This is a simple container class, not a Pydantic model, for flexibility.
    """
    
    def __init__(self, config: Config, recipe: Recipe):
        """
        Initialize Settings with Config and Recipe.
        
        Args:
            config: Infrastructure configuration
            recipe: Workflow definition
        """
        self.config = config
        self.recipe = recipe
        
        # Validate consistency between config and recipe
        self._validate()
    
    def _validate(self) -> None:
        """Basic validation to ensure config and recipe are compatible."""
        # Check that recipe's adapter exists in config (only if explicitly specified)
        adapter_name = self.recipe.data.loader.adapter
        if adapter_name:  # Only validate if adapter is explicitly specified
            if adapter_name not in self.config.adapters:
                raise ValueError(
                    f"Recipe references adapter '{adapter_name}' which is not defined in Config. "
                    f"Available adapters: {list(self.config.adapters.keys())}"
                )
        # If no adapter specified, it will be auto-detected from source_uri pattern
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.dict(),
            "recipe": self.recipe.dict()
        }


def _resolve_env_variables(data: Any) -> Any:
    """
    Recursively resolve environment variables in configuration.
    Supports ${VAR_NAME:default} pattern.
    
    Args:
        data: Configuration data (dict, list, or scalar)
        
    Returns:
        Data with environment variables resolved
    """
    if isinstance(data, str):
        # Check for ${VAR:default} pattern
        if data.startswith("${") and data.endswith("}"):
            var_expr = data[2:-1]
            if ":" in var_expr:
                var_name, default = var_expr.split(":", 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var_expr, data)
        return data
    elif isinstance(data, dict):
        return {k: _resolve_env_variables(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_variables(item) for item in data]
    return data


def load_settings(recipe_file: str, env_name: str) -> Settings:
    """
    Load Settings with simplified 3-step process (v2.0).
    
    Step 1: Load Config from configs/{env_name}.yaml
    Step 2: Load Recipe from recipes/{recipe_file}
    Step 3: Create and validate Settings
    
    Args:
        recipe_file: Path to Recipe file (relative or absolute)
        env_name: Environment name (required)
        
    Returns:
        Settings object containing both Config and Recipe
        
    Raises:
        FileNotFoundError: If config or recipe file not found
        ValueError: If validation fails
    """
    # Step 1: Load Config
    config = _load_config(env_name)
    logger.info(f"Config loaded for environment: {env_name}")
    
    # Step 2: Load Recipe
    recipe = _load_recipe(recipe_file)
    logger.info(f"Recipe loaded: {recipe.name}")
    
    # Step 3: Create Settings (validation happens in __init__)
    settings = Settings(config, recipe)
    
    # Add computed fields to recipe
    _add_computed_fields(settings, recipe_file)
    
    logger.info(f"Settings loaded successfully for {recipe.name} in {env_name} environment")
    return settings


def _load_config(env_name: str) -> Config:
    """
    Load Config from configs directory.
    
    Args:
        env_name: Environment name
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If config file not found
    """
    # Try environment-specific config first
    config_path = Path("configs") / f"{env_name}.yaml"
    
    # Fallback to base.yaml if environment-specific not found
    if not config_path.exists():
        base_path = Path("configs") / "base.yaml"
        if base_path.exists():
            logger.warning(f"Config for '{env_name}' not found, using base.yaml")
            config_path = base_path
        else:
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Run 'mmp get-config --env-name {env_name}' to create it."
            )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    # Resolve environment variables
    config_data = _resolve_env_variables(config_data)
    
    return Config(**config_data)


def _load_recipe(recipe_file: str) -> Recipe:
    """
    Load Recipe from recipes directory or absolute path.
    
    Args:
        recipe_file: Recipe file path
        
    Returns:
        Recipe object
        
    Raises:
        FileNotFoundError: If recipe file not found
    """
    recipe_path = Path(recipe_file)
    
    # Add .yaml extension if not present
    if not recipe_path.suffix:
        recipe_path = recipe_path.with_suffix(".yaml")
    
    # Try different locations
    if not recipe_path.exists():
        # Try in recipes directory
        recipes_path = Path("recipes") / recipe_path.name
        if recipes_path.exists():
            recipe_path = recipes_path
        else:
            raise FileNotFoundError(f"Recipe file not found: {recipe_file}")
    
    with open(recipe_path, "r", encoding="utf-8") as f:
        recipe_data = yaml.safe_load(f)
    
    return Recipe(**recipe_data)


def _add_computed_fields(settings: Settings, recipe_file: str) -> None:
    """
    Add computed fields to Recipe model.
    
    Args:
        settings: Settings object
        recipe_file: Original recipe file path (for run_name generation)
    """
    # Generate run_name if not present
    if not settings.recipe.model.computed:
        settings.recipe.model.computed = {}
    
    if "run_name" not in settings.recipe.model.computed:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recipe_name = Path(recipe_file).stem
        settings.recipe.model.computed["run_name"] = f"{recipe_name}_{timestamp}"


# Backward compatibility function
def load_settings_by_file(recipe_file: str, env_name: str, **kwargs) -> Settings:
    """
    Legacy compatibility wrapper for load_settings.
    
    This function exists for backward compatibility.
    New code should use load_settings() directly.
    """
    import warnings
    warnings.warn(
        "load_settings_by_file is deprecated. Use load_settings() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return load_settings(recipe_file, env_name)