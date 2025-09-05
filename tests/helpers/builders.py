"""
Test object builders for creating test data and objects.
Provides factory methods for creating complex test objects with sensible defaults.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import random
import string

from src.settings import Settings, Config, Recipe
from src.settings.config import Environment, MLflow, DataSource, FeatureStore
from src.settings.recipe import Model, Data, Loader, EntitySchema, DataInterface, Fetcher


class ConfigBuilder:
    """Builder for creating Config objects for testing."""
    
    @staticmethod
    def build(
        env_name: str = "test",
        mlflow_tracking_uri: str = "./mlruns",
        adapter_type: str = "storage",
        feature_store_provider: str = "none",
        **overrides
    ) -> Config:
        """Build a Config object with defaults and overrides."""
        config_dict = {
            'environment': {
                'name': env_name
            },
            'mlflow': {
                'tracking_uri': mlflow_tracking_uri,
                'experiment_name': f'{env_name}_experiment'
            },
            'data_source': {
                'name': f'{env_name}_datasource',
                'adapter_type': adapter_type,
                'config': {
                    'base_path': './data'
                }
            },
            'feature_store': {
                'provider': feature_store_provider
            }
        }
        
        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like 'mlflow.tracking_uri'
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
        **overrides
    ) -> Settings:
        """Build a Settings object with defaults and overrides."""
        config = ConfigBuilder.build(env_name=env_name, **overrides.get('config', {}))
        recipe = RecipeBuilder.build(name=recipe_name, **overrides.get('recipe', {}))
        return Settings(config=config, recipe=recipe)


class RecipeBuilder:
    """Builder for creating Recipe objects for testing."""
    
    @staticmethod
    def build(
        name: str = "test_recipe",
        model_class_path: str = "sklearn.ensemble.RandomForestClassifier",
        task_type: str = "classification",
        source_uri: str = "./data/sample.csv",
        fetcher_type: str = "pass_through",
        **overrides
    ) -> Recipe:
        """Build a Recipe object with defaults and overrides."""
        recipe_dict = {
            'name': name,
            'model': {
                'name': f'{name}_model',
                'class_path': model_class_path,
                'library': model_class_path.split('.')[0],
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {
                        'random_state': 42
                    }
                },
                'data_interface': {
                    'task_type': task_type,
                    'target_column': 'target',
                    'feature_columns': ['feature1', 'feature2']
                },
                'loader': {
                    'adapter': 'storage',
                    'source_uri': source_uri,
                    'entity_schema': {
                        'entity_columns': ['user_id'],
                        'timestamp_column': 'event_timestamp'
                    }
                },
                'fetcher': {
                    'type': fetcher_type
                }
            },
            'data': {
                'loader': {
                    'adapter': 'storage',
                    'source_uri': source_uri,
                    'entity_schema': {
                        'entity_columns': ['user_id'],
                        'timestamp_column': 'event_timestamp'
                    }
                },
                'fetcher': {
                    'type': fetcher_type
                },
                'data_interface': {
                    'task_type': task_type,
                    'target_column': 'target',
                    'feature_columns': ['feature1', 'feature2']
                }
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                'split_strategy': 'time_based_split',
                'test_size': 0.2
            }
        }
        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                parts = key.split('.')
                current = recipe_dict
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            else:
                recipe_dict[key] = value
        
        return Recipe(**recipe_dict)


class SettingsBuilder:
    """Builder for creating Settings objects for testing."""
    
    @staticmethod
    def build(
        env_name: str = "test",
        recipe_name: str = "test_recipe",
        **overrides
    ) -> Settings:
        """Build a Settings object with defaults."""
        config = ConfigBuilder.build(env_name=env_name)
        recipe = RecipeBuilder.build(name=recipe_name)
        return Settings(config=config, recipe=recipe)


class DataFrameBuilder:
    """Builder for creating test DataFrames."""
    
    @staticmethod
    def build_classification_data(
        n_samples: int = 100,
        n_features: int = 5,
        n_classes: int = 2,
        add_entity_column: bool = True,
        add_timestamp: bool = False,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Build a classification dataset."""
        random.seed(random_state)
        
        data = {}
        
        # Add entity column if requested
        if add_entity_column:
            data['user_id'] = list(range(n_samples))
        
        # Add timestamp if requested
        if add_timestamp:
            base_time = datetime.now()
            data['timestamp'] = [
                base_time + timedelta(hours=i) 
                for i in range(n_samples)
            ]
        
        # Add feature columns
        for i in range(n_features):
            data[f'feature_{i}'] = [
                random.gauss(0, 1) for _ in range(n_samples)
            ]
        
        # Add target column
        data['target'] = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
        
        return pd.DataFrame(data)
    
    @staticmethod
    def build_regression_data(
        n_samples: int = 100,
        n_features: int = 5,
        add_entity_column: bool = True,
        add_timestamp: bool = False,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Build a regression dataset."""
        random.seed(random_state)
        
        data = {}
        
        # Add entity column if requested
        if add_entity_column:
            data['user_id'] = list(range(n_samples))
        
        # Add timestamp if requested  
        if add_timestamp:
            base_time = datetime.now()
            data['timestamp'] = [
                base_time + timedelta(hours=i)
                for i in range(n_samples)
            ]
        
        # Add feature columns
        for i in range(n_features):
            data[f'feature_{i}'] = [
                random.gauss(0, 1) for _ in range(n_samples)
            ]
        
        # Add target column (continuous)
        data['target'] = [
            sum(data[f'feature_{i}'][j] for i in range(n_features)) + random.gauss(0, 0.1)
            for j in range(n_samples)
        ]
        
        return pd.DataFrame(data)
    
    @staticmethod
    def build_time_series_data(
        n_samples: int = 100,
        n_features: int = 3,
        freq: str = 'H',
        random_state: int = 42
    ) -> pd.DataFrame:
        """Build a time series dataset."""
        random.seed(random_state)
        
        # Create time index
        time_index = pd.date_range(
            start=datetime.now(),
            periods=n_samples,
            freq=freq
        )
        
        data = {'timestamp': time_index}
        
        # Add feature columns with some temporal correlation
        for i in range(n_features):
            values = []
            current = random.gauss(0, 1)
            for _ in range(n_samples):
                current = 0.8 * current + 0.2 * random.gauss(0, 1)
                values.append(current)
            data[f'feature_{i}'] = values
        
        # Add target with temporal pattern
        data['target'] = [
            sum(data[f'feature_{i}'][j] for i in range(n_features)) + 
            0.1 * j + random.gauss(0, 0.1)
            for j in range(n_samples)
        ]
        
        return pd.DataFrame(data)


class FileBuilder:
    """Builder for creating test files."""
    
    @staticmethod
    def create_yaml_file(
        path: Path,
        content: Dict[str, Any]
    ) -> Path:
        """Create a YAML file with given content."""
        import yaml
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(content, f, default_flow_style=False)
        
        return path
    
    @staticmethod
    def create_csv_file(
        path: Path,
        dataframe: Optional[pd.DataFrame] = None,
        n_rows: int = 100,
        n_cols: int = 5
    ) -> Path:
        """Create a CSV file with data."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if dataframe is None:
            dataframe = DataFrameBuilder.build_classification_data(
                n_samples=n_rows,
                n_features=n_cols - 1  # -1 for target column
            )
        
        dataframe.to_csv(path, index=False)
        return path
    
    @staticmethod
    def create_sql_file(
        path: Path,
        query: Optional[str] = None
    ) -> Path:
        """Create a SQL file with query."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if query is None:
            query = """
            SELECT 
                user_id,
                feature_1,
                feature_2,
                target
            FROM training_data
            WHERE created_at >= '2024-01-01'
            """
        
        with open(path, 'w') as f:
            f.write(query)
        
        return path


class MockBuilder:
    """Builder for creating mock objects."""
    
    @staticmethod
    def build_mock_model(
        model_type: str = "classifier",
        predict_value: Any = None
    ):
        """Build a mock ML model."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        
        if model_type == "classifier":
            if predict_value is None:
                predict_value = [0, 1, 0, 1, 0]
            mock_model.predict.return_value = predict_value
            mock_model.predict_proba.return_value = [
                [0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7], [0.8, 0.2]
            ]
            mock_model.classes_ = [0, 1]
        elif model_type == "regressor":
            if predict_value is None:
                predict_value = [1.5, 2.3, 0.8, 3.2, 1.9]
            mock_model.predict.return_value = predict_value
        
        mock_model.fit.return_value = mock_model
        return mock_model
    
    @staticmethod
    def build_mock_adapter():
        """Build a mock data adapter."""
        from unittest.mock import MagicMock
        
        mock_adapter = MagicMock()
        mock_adapter.read.return_value = DataFrameBuilder.build_classification_data()
        mock_adapter.write.return_value = None
        
        return mock_adapter
    
    @staticmethod
    def build_mock_fetcher():
        """Build a mock feature fetcher."""
        from unittest.mock import MagicMock
        
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = DataFrameBuilder.build_classification_data()
        
        return mock_fetcher
    
    @staticmethod
    def build_mock_evaluator(metrics: Optional[Dict[str, float]] = None):
        """Build a mock evaluator."""
        from unittest.mock import MagicMock
        
        mock_evaluator = MagicMock()
        
        if metrics is None:
            metrics = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1': 0.85
            }
        
        mock_evaluator.evaluate.return_value = metrics
        
        return mock_evaluator


def random_string(length: int = 10) -> str:
    """Generate a random string for testing."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def random_int(min_val: int = 0, max_val: int = 100) -> int:
    """Generate a random integer for testing."""
    return random.randint(min_val, max_val)


def random_float(min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Generate a random float for testing."""
    return random.uniform(min_val, max_val)