"""
Custom assertion helpers for testing.
Provides additional assertion methods beyond standard pytest assertions.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


def assert_config_valid(config: Any) -> None:
    """Assert that a Config object is valid."""
    assert hasattr(config, 'environment'), "Config must have environment"
    assert hasattr(config, 'data_source'), "Config must have data_source"
    assert config.environment.name, "Environment must have a name"
    assert config.data_source.adapter_type in ['sql', 'storage', 'bigquery'], \
        f"Invalid adapter type: {config.data_source.adapter_type}"


def assert_recipe_valid(recipe: Any) -> None:
    """Assert that a Recipe object is valid."""
    assert hasattr(recipe, 'name'), "Recipe must have name"
    assert hasattr(recipe, 'model'), "Recipe must have model"
    assert hasattr(recipe, 'data'), "Recipe must have data"
    assert recipe.model.class_path, "Model must have class_path"
    # Note: data_interface is now on recipe.data, not recipe.model
    assert recipe.data.data_interface, "Data must have data_interface"


def assert_settings_valid(settings: Any) -> None:
    """Assert that a Settings object is valid."""
    assert hasattr(settings, 'config'), "Settings must have config"
    assert hasattr(settings, 'recipe'), "Settings must have recipe"
    assert_config_valid(settings.config)
    assert_recipe_valid(settings.recipe)


def assert_dataframes_almost_equal(
    df1: pd.DataFrame, 
    df2: pd.DataFrame,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> None:
    """Assert that two dataframes are almost equal (for floating point comparison)."""
    assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"
    assert list(df1.columns) == list(df2.columns), f"Columns mismatch: {df1.columns} vs {df2.columns}"
    
    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]):
            np.testing.assert_allclose(
                df1[col].values, 
                df2[col].values, 
                rtol=rtol, 
                atol=atol,
                err_msg=f"Column {col} values differ"
            )
        else:
            pd.testing.assert_series_equal(
                df1[col], 
                df2[col],
                check_names=False
            )


def assert_registry_contains(registry: Any, key: str) -> None:
    """Assert that a registry contains a specific key."""
    if hasattr(registry, 'adapters'):
        assert key in registry.adapters, f"Registry does not contain adapter: {key}"
    elif hasattr(registry, 'fetchers'):
        assert key in registry.fetchers, f"Registry does not contain fetcher: {key}"
    elif hasattr(registry, 'evaluators'):
        assert key in registry.evaluators, f"Registry does not contain evaluator: {key}"
    else:
        raise ValueError(f"Unknown registry type: {type(registry)}")


def assert_factory_can_create(factory: Any, component_type: str) -> None:
    """Assert that a factory can create a specific component type."""
    creators = {
        'adapter': 'create_data_adapter',
        'fetcher': 'create_fetcher',
        'preprocessor': 'create_preprocessor',
        'model': 'create_model',
        'evaluator': 'create_evaluator'
    }
    
    assert component_type in creators, f"Unknown component type: {component_type}"
    creator_method = creators[component_type]
    assert hasattr(factory, creator_method), f"Factory does not have method: {creator_method}"
    
    # Try to create the component
    method = getattr(factory, creator_method)
    if component_type == 'adapter':
        component = method('storage')  # Default to storage adapter
    else:
        component = method()
    
    assert component is not None, f"Failed to create component: {component_type}"


def assert_mlflow_logged(mock_mlflow: Dict[str, Any], metric_names: Optional[List[str]] = None) -> None:
    """Assert that MLflow logging was called with expected metrics."""
    assert mock_mlflow['start_run'].called, "MLflow run was not started"
    
    if metric_names:
        logged_metrics = {}
        
        # Collect all logged metrics
        for call in mock_mlflow['log_metric'].call_args_list:
            args = call[0]
            if len(args) >= 2:
                logged_metrics[args[0]] = args[1]
        
        for call in mock_mlflow['log_metrics'].call_args_list:
            args = call[0]
            if len(args) >= 1 and isinstance(args[0], dict):
                logged_metrics.update(args[0])
        
        # Check that expected metrics were logged
        for metric_name in metric_names:
            assert metric_name in logged_metrics, \
                f"Expected metric '{metric_name}' was not logged. Logged metrics: {list(logged_metrics.keys())}"


def assert_file_exists(path: Any) -> None:
    """Assert that a file exists at the given path."""
    from pathlib import Path
    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    assert path.is_file(), f"Path is not a file: {path}"


def assert_directory_exists(path: Any) -> None:
    """Assert that a directory exists at the given path."""
    from pathlib import Path
    path = Path(path)
    assert path.exists(), f"Directory does not exist: {path}"
    assert path.is_dir(), f"Path is not a directory: {path}"


def assert_yaml_valid(yaml_content: str) -> None:
    """Assert that a string contains valid YAML."""
    import yaml
    try:
        yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise AssertionError(f"Invalid YAML content: {e}")


def assert_json_valid(json_content: str) -> None:
    """Assert that a string contains valid JSON."""
    import json
    try:
        json.loads(json_content)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON content: {e}")


def assert_between(value: float, min_val: float, max_val: float, name: str = "Value") -> None:
    """Assert that a value is between min and max (inclusive)."""
    assert min_val <= value <= max_val, \
        f"{name} {value} is not between {min_val} and {max_val}"


def assert_shape(array_like: Any, expected_shape: tuple) -> None:
    """Assert that an array-like object has the expected shape."""
    if hasattr(array_like, 'shape'):
        actual_shape = array_like.shape
    elif isinstance(array_like, list):
        actual_shape = (len(array_like),)
        if array_like and isinstance(array_like[0], list):
            actual_shape = (len(array_like), len(array_like[0]))
    else:
        raise ValueError(f"Cannot determine shape of {type(array_like)}")
    
    assert actual_shape == expected_shape, \
        f"Shape mismatch: expected {expected_shape}, got {actual_shape}"