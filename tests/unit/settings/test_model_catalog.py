"""
Unit Tests for Model Catalog Validation
Days 3-5: Model validation & specs tests
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import yaml

from src.settings.validator import (
    ModelSpec, ModelCatalog, TunableParameter, HyperparameterSpec, Validator
)


class TestModelCatalogLoading:
    """Model catalog loading and parsing tests"""
    
    def test_model_catalog_load_from_directory_success(self):
        """Test successful model catalog loading from directory structure"""
        # Mock directory structure
        mock_catalog_dir = MagicMock()
        mock_catalog_dir.exists.return_value = True
        
        # Mock task directories
        classification_dir = MagicMock()
        classification_dir.name = "Classification"
        classification_dir.is_dir.return_value = True
        
        regression_dir = MagicMock()
        regression_dir.name = "Regression"
        regression_dir.is_dir.return_value = True
        
        mock_catalog_dir.iterdir.return_value = [classification_dir, regression_dir]
        
        # Mock YAML files
        rf_file = MagicMock()
        rf_file.stem = "RandomForestClassifier"
        lr_file = MagicMock()
        lr_file.stem = "LogisticRegression"
        classification_dir.glob.return_value = [rf_file, lr_file]
        
        linear_file = MagicMock()
        linear_file.stem = "LinearRegression" 
        regression_dir.glob.return_value = [linear_file]
        
        # Mock ModelSpec creation
        with patch.object(ModelSpec, 'from_yaml') as mock_from_yaml:
            mock_spec1 = MagicMock()
            mock_spec2 = MagicMock()
            mock_spec3 = MagicMock()
            mock_from_yaml.side_effect = [mock_spec1, mock_spec2, mock_spec3]
            
            # Load catalog
            catalog = ModelCatalog.load_from_directory(mock_catalog_dir)
            
            # Verify structure
            assert len(catalog.models) == 2
            assert "Classification" in catalog.models
            assert "Regression" in catalog.models
            assert len(catalog.models["Classification"]) == 2
            assert len(catalog.models["Regression"]) == 1
            
            # Verify models loaded
            assert "RandomForestClassifier" in catalog.models["Classification"]
            assert "LogisticRegression" in catalog.models["Classification"]
            assert "LinearRegression" in catalog.models["Regression"]
    
    def test_model_catalog_handles_missing_directory(self):
        """Test model catalog handles missing catalog directory gracefully"""
        mock_catalog_dir = MagicMock()
        mock_catalog_dir.exists.return_value = False
        
        catalog = ModelCatalog.load_from_directory(mock_catalog_dir)
        
        # Should return empty catalog without crashing
        assert len(catalog.models) == 0
        assert catalog.list_tasks() == []
    
    def test_model_catalog_skips_deep_learning_directory(self):
        """Test model catalog skips DeepLearning directory as specified"""
        mock_catalog_dir = MagicMock()
        mock_catalog_dir.exists.return_value = True
        
        # Mock directories including DeepLearning
        classification_dir = MagicMock()
        classification_dir.name = "Classification"
        classification_dir.is_dir.return_value = True
        classification_dir.glob.return_value = []
        
        deep_learning_dir = MagicMock()
        deep_learning_dir.name = "DeepLearning"
        deep_learning_dir.is_dir.return_value = True
        
        mock_catalog_dir.iterdir.return_value = [classification_dir, deep_learning_dir]
        
        catalog = ModelCatalog.load_from_directory(mock_catalog_dir)
        
        # DeepLearning should be excluded
        assert "DeepLearning" not in catalog.models
        assert "Classification" in catalog.models


class TestModelSpecValidation:
    """Model specification validation tests"""
    
    def test_model_spec_from_yaml_success(self):
        """Test successful ModelSpec loading from YAML"""
        yaml_content = {
            'class_path': 'sklearn.ensemble.RandomForestClassifier',
            'library': 'sklearn',
            'description': 'Random Forest for classification',
            'supported_tasks': ['classification'],
            'hyperparameters': {
                'fixed': {'random_state': 42, 'n_jobs': -1},
                'tunable': {
                    'n_estimators': {
                        'type': 'int',
                        'range': [10, 200],
                        'default': 100,
                        'log': False
                    },
                    'max_depth': {
                        'type': 'int',
                        'range': [3, 20],
                        'default': None
                    }
                }
            }
        }
        
        mock_path = MagicMock()
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=yaml_content):
            
            spec = ModelSpec.from_yaml(mock_path)
            
            # Verify basic properties
            assert spec.class_path == 'sklearn.ensemble.RandomForestClassifier'
            assert spec.library == 'sklearn'
            assert spec.description == 'Random Forest for classification'
            assert spec.supported_tasks == ['classification']
            
            # Verify hyperparameters structure
            assert 'random_state' in spec.hyperparameters.fixed
            assert spec.hyperparameters.fixed['random_state'] == 42
            
            # Verify tunable parameters
            assert 'n_estimators' in spec.hyperparameters.tunable
            n_est_param = spec.hyperparameters.tunable['n_estimators']
            assert isinstance(n_est_param, TunableParameter)
            assert n_est_param.type == 'int'
            assert n_est_param.range == [10, 200]
            assert n_est_param.default == 100
    
    def test_model_spec_task_compatibility_check(self):
        """Test model spec task compatibility validation"""
        spec = ModelSpec(
            class_path='sklearn.ensemble.RandomForestClassifier',
            library='sklearn',
            supported_tasks=['classification', 'binary_classification']
        )
        
        # Compatible tasks
        assert spec.is_compatible_with_task('classification') is True
        assert spec.is_compatible_with_task('Classification') is True  # Case insensitive
        assert spec.is_compatible_with_task('binary_classification') is True
        
        # Incompatible task
        assert spec.is_compatible_with_task('regression') is False
        
        # Empty supported_tasks should allow all tasks
        spec_no_tasks = ModelSpec(
            class_path='custom.model.FlexibleModel',
            library='custom',
            supported_tasks=[]
        )
        assert spec_no_tasks.is_compatible_with_task('any_task') is True


class TestTunableParameterValidation:
    """Tunable parameter validation tests"""
    
    def test_tunable_parameter_int_range_validation(self):
        """Test integer parameter range validation"""
        # Valid integer parameter
        param = TunableParameter(
            type='int',
            range=[1, 100],
            default=10
        )
        assert param.type == 'int'
        assert param.range == [1, 100]
        assert param.default == 10
        
        # Invalid range (min >= max)
        with pytest.raises(ValueError, match="범위가 잘못되었습니다"):
            TunableParameter(
                type='int',
                range=[100, 1]  # Invalid: min >= max
            )
        
        # Invalid range format (not 2 elements)
        with pytest.raises(ValueError, match="\\[min, max\\] 형태여야 합니다"):
            TunableParameter(
                type='int',
                range=[1, 2, 3]  # Invalid: too many elements
            )
    
    def test_tunable_parameter_categorical_validation(self):
        """Test categorical parameter validation"""
        # Valid categorical parameter
        param = TunableParameter(
            type='categorical',
            range=['option1', 'option2', 'option3'],
            default='option1'
        )
        assert param.type == 'categorical'
        assert param.range == ['option1', 'option2', 'option3']
        
        # Invalid categorical (too few options)
        with pytest.raises(ValueError, match="2개 이상의 선택지가 필요합니다"):
            TunableParameter(
                type='categorical',
                range=['single_option']  # Invalid: only one option
            )


class TestHyperparameterSpecConversion:
    """Hyperparameter specification conversion tests"""
    
    def test_hyperparameter_spec_to_recipe_with_tuning_enabled(self):
        """Test conversion to recipe format with tuning enabled"""
        spec = HyperparameterSpec(
            fixed={'random_state': 42, 'n_jobs': -1},
            tunable={
                'n_estimators': TunableParameter(
                    type='int',
                    range=[10, 200], 
                    default=100,
                    log=False
                ),
                'learning_rate': TunableParameter(
                    type='float',
                    range=[0.01, 1.0],
                    log=True
                )
            }
        )
        
        result = spec.to_recipe_hyperparameters(enable_tuning=True)
        
        # Verify tuning enabled structure
        assert result['tuning_enabled'] is True
        assert result['fixed'] == {'random_state': 42, 'n_jobs': -1}
        
        # Verify tunable parameters structure
        assert 'n_estimators' in result['tunable']
        assert result['tunable']['n_estimators']['type'] == 'int'
        assert result['tunable']['n_estimators']['range'] == [10, 200]
        assert result['tunable']['n_estimators']['log'] is False
        
        assert 'learning_rate' in result['tunable']
        assert result['tunable']['learning_rate']['log'] is True
    
    def test_hyperparameter_spec_to_recipe_with_tuning_disabled(self):
        """Test conversion to recipe format with tuning disabled"""
        spec = HyperparameterSpec(
            fixed={'random_state': 42},
            tunable={
                'n_estimators': TunableParameter(
                    type='int',
                    range=[10, 200],
                    default=100
                ),
                'max_depth': TunableParameter(
                    type='categorical',
                    range=['auto', 'sqrt', 'log2'],
                    default='auto'
                )
            }
        )
        
        result = spec.to_recipe_hyperparameters(enable_tuning=False)
        
        # Verify tuning disabled structure
        assert result['tuning_enabled'] is False
        assert 'values' in result
        
        # Verify fixed parameters included
        assert result['values']['random_state'] == 42
        
        # Verify tunable defaults used
        assert result['values']['n_estimators'] == 100
        assert result['values']['max_depth'] == 'auto'