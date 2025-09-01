"""
Settings Module Public API (v2.0)
Simplified and unified settings management.

This module provides the public API for loading and managing settings.
It combines infrastructure (Config) and workflow (Recipe) settings.
"""

from .loader import Settings, load_settings
from .config import Config, Environment, MLflow, Adapter, Serving, ArtifactStore
from .recipe import Recipe, Model, Data, Loader, DataInterface, Evaluation, Preprocessor  
from .validator import Validator, ModelSpec, ModelCatalog, HyperparameterSpec

# For inference-only operations (batch inference, serving)
def create_settings_for_inference(config_data):
    """
    Create Settings for inference operations.
    
    Args:
        config_data: Dictionary containing config data
        
    Returns:
        Settings object with minimal Recipe
    """
    # Create Config from data
    config = Config(**config_data)
    
    # Create minimal Recipe for inference
    recipe = Recipe(
        name="inference",
        model=Model(
            class_path="inference.model",
            hyperparameters={}
        ),
        data=Data(
            loader=Loader(
                name="inference_loader", 
                adapter=list(config.adapters.keys())[0] if config.adapters else "storage",
                source_uri="inference_data"
            ),
            data_interface=DataInterface(
                task_type="classification",
                target_column="target"
            )
        ),
        evaluation=Evaluation(
            metrics=["accuracy"],
            validation={"method": "split"}
        )
    )
    
    return Settings(config, recipe)

# Alias for backward compatibility 
load_settings_by_file = load_settings

def load_config_files(env_name: str):
    """
    Load config files for environment.
    
    Args:
        env_name: Environment name
        
    Returns:
        Config dictionary
    """
    from .loader import _load_config
    config = _load_config(env_name)
    return config.dict()

__all__ = [
    # Core classes
    "Settings",
    "Config", 
    "Recipe",
    "Validator",
    
    # Schema classes
    "Environment",
    "MLflow", 
    "Adapter",
    "Serving",
    "ArtifactStore",
    "Model",
    "Data", 
    "Loader",
    "DataInterface",
    "Evaluation",
    "Preprocessor",
    
    # Model catalog classes
    "ModelSpec",
    "ModelCatalog", 
    "HyperparameterSpec",
    
    # Loading functions
    "load_settings",
    "load_settings_by_file",
    "create_settings_for_inference", 
    "load_config_files",
]