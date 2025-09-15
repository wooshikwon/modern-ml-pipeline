"""
Unit Tests for Settings System
Tests the core Settings loading, validation, and configuration functionality
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.settings import Settings, load_settings
from src.settings.factory import SettingsFactory
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
                },
                'split': {
                    'train': 0.6,
                    'validation': 0.2,
                    'test': 0.2,
                    'calibration': 0.0
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
    
    def test_recipe_with_hyperparameter_tuning(self, component_test_context):
        """Test recipe with hyperparameter tuning enabled"""
        with component_test_context.classification_stack() as ctx:
            factory = SettingsFactory()

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
                },
                'split': {
                    'train': 0.7,
                    'validation': 0.15,
                    'test': 0.15,
                    'calibration': 0.0
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
        
            recipe_file = ctx.temp_dir / "tuning_recipe.yaml"
            with open(recipe_file, 'w') as f:
                yaml.dump(recipe_data, f)

            recipe = factory._load_recipe(str(recipe_file))
            assert recipe.model.hyperparameters.tuning_enabled is True
            assert recipe.model.hyperparameters.optimization_metric == 'mae'
            assert recipe.model.hyperparameters.n_trials == 50
            assert 'n_estimators' in recipe.model.hyperparameters.tunable
    
    def test_missing_recipe_file_error(self):
        """Test error handling for missing recipe files."""
        with pytest.raises(FileNotFoundError, match="Recipe 파일을 찾을 수 없습니다"):
            _load_recipe("nonexistent_recipe")


class TestSettingsIntegration:
    """Test Settings integration and validation using Real Object Testing"""

    def test_complete_settings_loading(self, component_test_context, isolated_temp_directory):
        """Test loading complete Settings with Config + Recipe using context"""
        with component_test_context.classification_stack() as ctx:
            # Use existing settings from context and verify integration
            settings = ctx.settings

            assert isinstance(settings, Settings)
            assert settings.config.environment.name == 'test'
            assert settings.recipe.task_choice == 'classification'
            assert settings.recipe.model.class_path.endswith('RandomForestClassifier')

            # Test real loading with temporary files using isolated_temp_directory fixture
            temp_dir = isolated_temp_directory
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

            config_file = temp_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)

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
                    },
                    'split': {
                        'train': 0.6,
                        'validation': 0.2,
                        'test': 0.2,
                        'calibration': 0.0
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

            recipe_file = temp_dir / "recipe.yaml"
            with open(recipe_file, 'w') as f:
                yaml.dump(recipe_data, f)

            # Load complete settings with Real Object Testing
            loaded_settings = load_settings(str(recipe_file), str(config_file))

            assert isinstance(loaded_settings, Settings)
            assert loaded_settings.config.environment.name == 'test'
            assert loaded_settings.recipe.name == 'integration_test'
            assert loaded_settings.recipe.task_choice == 'classification'
            assert loaded_settings.config.mlflow.experiment_name == 'test_experiment'
    
    def test_feature_store_validation_error(self, component_test_context, isolated_temp_directory):
        """Test validation error when recipe uses feature_store but config doesn't support it"""
        with component_test_context.classification_stack() as ctx:
            temp_dir = isolated_temp_directory

            # Config without feast
            config_data = {
                'environment': {'name': 'test'},
                'data_source': {'name': 'test', 'adapter_type': 'storage', 'config': {}},
                'feature_store': {'provider': 'none'}
            }

            config_file = temp_dir / "config.yaml"
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
                    },
                    'split': {
                        'train': 0.6,
                        'validation': 0.2,
                        'test': 0.2,
                        'calibration': 0.0
                    }
                },
                'evaluation': {
                    'metrics': ['accuracy'],
                    'validation': {'method': 'train_test_split', 'test_size': 0.2}
                }
            }

            recipe_file = temp_dir / "recipe.yaml"
            with open(recipe_file, 'w') as f:
                yaml.dump(recipe_data, f)

            # Real Object Testing - should raise validation error
            with pytest.raises((ValueError, ValidationError)):
                load_settings(str(recipe_file), str(config_file))
    
    def test_settings_computed_fields(self, component_test_context, isolated_temp_directory):
        """Test that computed fields are properly added to Settings"""
        with component_test_context.classification_stack() as ctx:
            # Use context-provided settings to check computed fields
            settings = ctx.settings

            # Check computed fields exist
            assert hasattr(settings.recipe.model, 'computed')
            assert 'environment' in settings.recipe.model.computed
            assert settings.recipe.model.computed['environment'] == 'test'

            # Test with Real Object Testing for additional scenarios
            temp_dir = isolated_temp_directory
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
                    'data_interface': {'target_column': 'y', 'entity_columns': ['id']},
                    'split': {
                        'train': 0.6,
                        'validation': 0.2,
                        'test': 0.2,
                        'calibration': 0.0
                    }
                },
                'evaluation': {
                    'metrics': ['accuracy'],
                    'validation': {'method': 'train_test_split', 'test_size': 0.2}
                }
            }

            config_file = temp_dir / "config.yaml"
            recipe_file = temp_dir / "recipe.yaml"

            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            with open(recipe_file, 'w') as f:
                yaml.dump(recipe_data, f)

            loaded_settings = load_settings(str(recipe_file), str(config_file))

            # Check computed fields are added
            assert 'run_name' in loaded_settings.recipe.model.computed
            assert 'environment' in loaded_settings.recipe.model.computed
            assert loaded_settings.recipe.model.computed['environment'] == 'test'

            # Check run_name format
            run_name = loaded_settings.recipe.model.computed['run_name']
            assert isinstance(run_name, str)
            assert len(run_name) > 0


class TestSettingsUtilityMethods:
    """Test Settings utility methods using Real Object Testing"""

    def test_environment_name_getter(self, component_test_context):
        """Test getting environment name from context-provided settings"""
        with component_test_context.classification_stack() as ctx:
            assert ctx.settings.get_environment_name() == "test"

    def test_recipe_name_getter(self, component_test_context):
        """Test getting recipe name from context-provided settings"""
        with component_test_context.classification_stack() as ctx:
            # Context provides settings with generated recipe name
            recipe_name = ctx.settings.get_recipe_name()
            assert isinstance(recipe_name, str)
            assert len(recipe_name) > 0

    def test_to_dict_serialization(self, component_test_context):
        """Test Settings serialization to dictionary"""
        with component_test_context.classification_stack() as ctx:
            settings_dict = ctx.settings.to_dict()

            assert 'config' in settings_dict
            assert 'recipe' in settings_dict
            assert settings_dict['config']['environment']['name'] == 'test'
            assert settings_dict['recipe']['task_choice'] == 'classification'

            # Test that the method returns proper dictionary structure
            assert isinstance(settings_dict, dict)
            assert isinstance(settings_dict['config'], dict)
            assert isinstance(settings_dict['recipe'], dict)