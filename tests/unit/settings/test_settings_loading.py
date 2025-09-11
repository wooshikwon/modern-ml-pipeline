"""
Unit Tests for Settings System
Tests the core Settings loading, validation, and configuration functionality
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.settings.loader import Settings, load_settings, resolve_env_variables, _load_config, _load_recipe
from src.settings.config import Config, Environment, MLflow, DataSource, FeatureStore
from src.settings.recipe import Recipe, Model, Data, Evaluation, HyperparametersTuning


class TestEnvironmentVariableResolution:
    """Test environment variable resolution functionality."""
    
    def test_simple_string_substitution(self):
        """Test basic string environment variable substitution."""
        with patch.dict('os.environ', {'TEST_VAR': 'test_value'}):
            result = resolve_env_variables('${TEST_VAR}')
            assert result == 'test_value'
    
    def test_default_value_handling(self):
        """Test default values when environment variable doesn't exist."""
        result = resolve_env_variables('${NONEXISTENT:default_val}')
        assert result == 'default_val'
    
    def test_integer_conversion(self):
        """Test automatic conversion to integer."""
        with patch.dict('os.environ', {'PORT': '8080'}):
            result = resolve_env_variables('${PORT}')
            assert result == 8080
            assert isinstance(result, int)
    
    def test_float_conversion(self):
        """Test automatic conversion to float."""
        with patch.dict('os.environ', {'THRESHOLD': '0.85'}):
            result = resolve_env_variables('${THRESHOLD}')
            assert result == 0.85
            assert isinstance(result, float)
    
    def test_boolean_conversion(self):
        """Test automatic conversion to boolean."""
        with patch.dict('os.environ', {'DEBUG': 'true', 'PROD': 'false'}):
            assert resolve_env_variables('${DEBUG}') is True
            assert resolve_env_variables('${PROD}') is False
    
    def test_nested_dict_resolution(self):
        """Test environment variable resolution in nested dictionaries."""
        with patch.dict('os.environ', {'DB_HOST': 'localhost', 'DB_PORT': '5432'}):
            data = {
                'database': {
                    'host': '${DB_HOST}',
                    'port': '${DB_PORT}',
                    'name': '${DB_NAME:testdb}'
                }
            }
            result = resolve_env_variables(data)
            assert result['database']['host'] == 'localhost'
            assert result['database']['port'] == 5432
            assert result['database']['name'] == 'testdb'
    
    def test_list_resolution(self):
        """Test environment variable resolution in lists."""
        with patch.dict('os.environ', {'FEATURE1': 'age', 'FEATURE2': 'income'}):
            data = ['${FEATURE1}', '${FEATURE2}', 'static_feature']
            result = resolve_env_variables(data)
            assert result == ['age', 'income', 'static_feature']


class TestConfigLoading:
    """Test Config loading and validation."""
    
    def test_basic_config_loading(self, isolated_temp_directory):
        """Test loading a basic config file."""
        config_data = {
            'environment': {'name': 'test'},
            'data_source': {
                'name': 'test_db',
                'adapter_type': 'sql',
                'config': {'connection_uri': 'sqlite:///test.db'}
            },
            'feature_store': {'provider': 'none'}
        }
        
        config_file = isolated_temp_directory / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = _load_config(str(config_file))
        assert config.environment.name == 'test'
        assert config.data_source.adapter_type == 'sql'
        assert config.feature_store.provider == 'none'
    
    def test_config_with_env_variables(self, isolated_temp_directory):
        """Test config loading with environment variable substitution."""
        with patch.dict('os.environ', {'ENV_NAME': 'production', 'DB_URI': 'postgresql://prod:5432/db'}):
            config_data = {
                'environment': {'name': '${ENV_NAME}'},
                'data_source': {
                    'name': 'prod_db',
                    'adapter_type': 'sql',
                    'config': {'connection_uri': '${DB_URI}'}
                },
                'feature_store': {'provider': 'none'}
            }
            
            config_file = isolated_temp_directory / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            config = _load_config(str(config_file))
            assert config.environment.name == 'production'
            assert config.data_source.config['connection_uri'] == 'postgresql://prod:5432/db'
    
    def test_missing_config_file_error(self):
        """Test error handling for missing config files."""
        with pytest.raises(FileNotFoundError, match="Config 파일을 찾을 수 없습니다"):
            _load_config("/nonexistent/config.yaml")
    
    def test_invalid_config_structure(self, isolated_temp_directory):
        """Test error handling for invalid config structure."""
        config_file = isolated_temp_directory / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'invalid': 'structure'}, f)
        
        with pytest.raises(ValueError, match="Config 파싱 실패"):
            _load_config(str(config_file))


class TestRecipeLoading:
    """Test Recipe loading and validation."""
    
    def test_basic_recipe_loading(self, isolated_temp_directory):
        """Test loading a basic recipe file."""
        recipe_data = {
            'name': 'test_recipe',
            'description': 'Test recipe for unit testing',
            'task_choice': 'classification',
            'model': {
                'class_path': 'sklearn.ensemble.RandomForestClassifier',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'n_estimators': 100, 'random_state': 42}
                }
            },
            'data': {
                'loader': {'source_uri': 'test_data.csv'},
                'fetcher': {'type': 'pass_through'},
                'data_interface': {
                    'target_column': 'target',
                    'entity_columns': ['id']
                }
            },
            'evaluation': {
                'metrics': ['accuracy', 'f1'],
                'validation': {
                    'method': 'train_test_split',
                    'test_size': 0.2
                }
            }
        }
        
        recipe_file = isolated_temp_directory / "test_recipe.yaml"
        with open(recipe_file, 'w') as f:
            yaml.dump(recipe_data, f)
        
        recipe = _load_recipe(str(recipe_file))
        assert recipe.name == 'test_recipe'
        assert recipe.task_choice == 'classification'
        assert recipe.model.class_path == 'sklearn.ensemble.RandomForestClassifier'
        assert recipe.model.hyperparameters.tuning_enabled is False
        assert recipe.data.data_interface.target_column == 'target'
    
    def test_recipe_with_hyperparameter_tuning(self, isolated_temp_directory):
        """Test recipe with hyperparameter tuning enabled."""
        recipe_data = {
            'name': 'tuning_recipe',
            'task_choice': 'regression',
            'model': {
                'class_path': 'sklearn.ensemble.RandomForestRegressor',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': True,
                    'optimization_metric': 'mae',
                    'direction': 'minimize',
                    'n_trials': 50,
                    'fixed': {'random_state': 42},
                    'tunable': {
                        'n_estimators': {'type': 'int', 'range': [10, 200]},
                        'max_depth': {'type': 'int', 'range': [3, 15]}
                    }
                }
            },
            'data': {
                'loader': {'source_uri': 'regression_data.csv'},
                'fetcher': {'type': 'pass_through'},
                'data_interface': {
                    'target_column': 'price',
                    'entity_columns': ['property_id']
                }
            },
            'evaluation': {
                'metrics': ['mae', 'rmse'],
                'validation': {
                    'method': 'cross_validation',
                    'cv_folds': 5
                }
            }
        }
        
        recipe_file = isolated_temp_directory / "tuning_recipe.yaml"
        with open(recipe_file, 'w') as f:
            yaml.dump(recipe_data, f)
        
        recipe = _load_recipe(str(recipe_file))
        assert recipe.model.hyperparameters.tuning_enabled is True
        assert recipe.model.hyperparameters.optimization_metric == 'mae'
        assert recipe.model.hyperparameters.n_trials == 50
        assert 'n_estimators' in recipe.model.hyperparameters.tunable
    
    def test_missing_recipe_file_error(self):
        """Test error handling for missing recipe files."""
        with pytest.raises(FileNotFoundError, match="Recipe 파일을 찾을 수 없습니다"):
            _load_recipe("nonexistent_recipe")


class TestSettingsIntegration:
    """Test Settings integration and validation."""
    
    def test_complete_settings_loading(self, isolated_temp_directory):
        """Test loading complete Settings with Config + Recipe."""
        # Create config file
        config_data = {
            'environment': {'name': 'test'},
            'data_source': {
                'name': 'test_storage',
                'adapter_type': 'storage',
                'config': {}
            },
            'feature_store': {'provider': 'none'},
            'mlflow': {
                'tracking_uri': 'sqlite:///test_mlflow.db',
                'experiment_name': 'test_experiment'
            }
        }
        
        config_file = isolated_temp_directory / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create recipe file
        recipe_data = {
            'name': 'integration_test',
            'task_choice': 'classification',
            'model': {
                'class_path': 'sklearn.ensemble.RandomForestClassifier',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'n_estimators': 50, 'random_state': 42}
                }
            },
            'data': {
                'loader': {'source_uri': 'test_data.csv'},
                'fetcher': {'type': 'pass_through'},
                'data_interface': {
                    'target_column': 'label',
                    'entity_columns': ['id']
                }
            },
            'evaluation': {
                'metrics': ['accuracy'],
                'validation': {
                    'method': 'train_test_split',
                    'test_size': 0.25
                }
            }
        }
        
        recipe_file = isolated_temp_directory / "recipe.yaml"
        with open(recipe_file, 'w') as f:
            yaml.dump(recipe_data, f)
        
        # Load complete settings
        settings = load_settings(str(recipe_file), str(config_file))
        
        assert isinstance(settings, Settings)
        assert settings.config.environment.name == 'test'
        assert settings.recipe.name == 'integration_test'
        assert settings.recipe.task_choice == 'classification'
        assert settings.config.mlflow.experiment_name == 'test_experiment'
    
    def test_feature_store_validation_error(self, isolated_temp_directory):
        """Test validation error when recipe uses feature_store but config doesn't support it."""
        # Config without feast
        config_data = {
            'environment': {'name': 'test'},
            'data_source': {'name': 'test', 'adapter_type': 'storage', 'config': {}},
            'feature_store': {'provider': 'none'}
        }
        
        config_file = isolated_temp_directory / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Recipe that uses feature_store
        recipe_data = {
            'name': 'feature_store_test',
            'task_choice': 'classification',
            'model': {
                'class_path': 'sklearn.ensemble.RandomForestClassifier',
                'library': 'sklearn',
                'hyperparameters': {'tuning_enabled': False, 'values': {}}
            },
            'data': {
                'loader': {'source_uri': 'test_data.csv'},
                'fetcher': {'type': 'feature_store', 'feature_views': {}},
                'data_interface': {
                    'target_column': 'label',
                    'entity_columns': ['id']
                }
            },
            'evaluation': {
                'metrics': ['accuracy'],
                'validation': {'method': 'train_test_split', 'test_size': 0.2}
            }
        }
        
        recipe_file = isolated_temp_directory / "recipe.yaml"
        with open(recipe_file, 'w') as f:
            yaml.dump(recipe_data, f)
        
        with pytest.raises(ValueError, match="feature_store fetcher를 사용하지만"):
            load_settings(str(recipe_file), str(config_file))
    
    def test_settings_computed_fields(self, isolated_temp_directory):
        """Test that computed fields are properly added to Settings."""
        # Create minimal config and recipe
        config_data = {
            'environment': {'name': 'test'},
            'data_source': {'name': 'test', 'adapter_type': 'storage', 'config': {}},
            'feature_store': {'provider': 'none'}
        }
        
        recipe_data = {
            'name': 'computed_test',
            'task_choice': 'classification',
            'model': {
                'class_path': 'sklearn.ensemble.RandomForestClassifier',
                'library': 'sklearn',
                'hyperparameters': {'tuning_enabled': False, 'values': {}}
            },
            'data': {
                'loader': {'source_uri': 'test.csv'},
                'fetcher': {'type': 'pass_through'},
                'data_interface': {'target_column': 'y', 'entity_columns': ['id']}
            },
            'evaluation': {
                'metrics': ['accuracy'],
                'validation': {'method': 'train_test_split', 'test_size': 0.2}
            }
        }
        
        config_file = isolated_temp_directory / "config.yaml"
        recipe_file = isolated_temp_directory / "recipe.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        with open(recipe_file, 'w') as f:
            yaml.dump(recipe_data, f)
        
        settings = load_settings(str(recipe_file), str(config_file))
        
        # Check computed fields are added
        assert 'run_name' in settings.recipe.model.computed
        assert 'environment' in settings.recipe.model.computed
        assert settings.recipe.model.computed['environment'] == 'test'
        
        # Check run_name format
        run_name = settings.recipe.model.computed['run_name']
        assert 'recipe' in run_name
        assert len(run_name.split('_')) >= 2  # Should have timestamp


class TestSettingsUtilityMethods:
    """Test Settings utility methods."""
    
    def test_environment_name_getter(self, minimal_classification_settings):
        """Test getting environment name."""
        assert minimal_classification_settings.get_environment_name() == "test"
    
    def test_recipe_name_getter(self, minimal_classification_settings):
        """Test getting recipe name."""
        assert minimal_classification_settings.get_recipe_name() == "test_recipe"
    
    def test_to_dict_serialization(self, minimal_classification_settings):
        """Test Settings serialization to dictionary."""
        settings_dict = minimal_classification_settings.to_dict()
        
        assert 'config' in settings_dict
        assert 'recipe' in settings_dict
        assert settings_dict['config']['environment']['name'] == 'test'
        assert settings_dict['recipe']['task_choice'] == 'classification'
        
        # Test that the method returns proper dictionary structure
        assert isinstance(settings_dict, dict)
        assert isinstance(settings_dict['config'], dict)
        assert isinstance(settings_dict['recipe'], dict)