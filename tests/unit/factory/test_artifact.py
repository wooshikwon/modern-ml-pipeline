"""
Unit tests for the PyfuncWrapper artifact class.
Tests MLflow model wrapping and prediction functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.factory.artifact import PyfuncWrapper
from src.settings import Settings
from tests.helpers.builders import ConfigBuilder, RecipeBuilder


class TestPyfuncWrapperInitialization:
    """Test PyfuncWrapper initialization."""
    
    def test_pyfunc_wrapper_initialization(self):
        """Test PyfuncWrapper initialization with all components."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        
        model = Mock()
        preprocessor = Mock()
        fetcher = Mock()
        training_results = {"accuracy": 0.95}
        signature = Mock()
        data_schema = Mock()
        
        wrapper = PyfuncWrapper(
            settings=settings,
            trained_model=model,
            trained_preprocessor=preprocessor,
            trained_fetcher=fetcher,
            training_results=training_results,
            signature=signature,
            data_schema=data_schema
        )
        
        assert wrapper.settings == settings
        assert wrapper.trained_model == model
        assert wrapper.trained_preprocessor == preprocessor
        assert wrapper.trained_fetcher == fetcher
        assert wrapper.training_results == training_results
        assert wrapper.signature == signature
        assert wrapper.data_schema == data_schema
    
    def test_pyfunc_wrapper_initialization_minimal(self):
        """Test PyfuncWrapper initialization with minimal components."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        
        model = Mock()
        
        wrapper = PyfuncWrapper(
            settings=settings,
            trained_model=model,
            trained_preprocessor=None,
            trained_fetcher=None
        )
        
        assert wrapper.settings == settings
        assert wrapper.trained_model == model
        assert wrapper.trained_preprocessor is None
        assert wrapper.trained_fetcher is None
        assert wrapper.training_results == {}
        assert wrapper.signature is None
        assert wrapper.data_schema is None


class TestPyfuncWrapperProperties:
    """Test PyfuncWrapper property methods."""
    
    @pytest.fixture
    def wrapper(self):
        """Create a PyfuncWrapper instance with mock components."""
        settings = Mock(spec=Settings)
        recipe = RecipeBuilder.build()
        settings.recipe = recipe
        
        # Setup recipe structure (correct structure)
        settings.recipe.model.class_path = "sklearn.ensemble.RandomForestClassifier"
        settings.recipe.data.loader = Mock(source_uri="SELECT * FROM table")
        settings.recipe.data.fetcher = Mock()
        settings.recipe.data.fetcher.model_dump = Mock(
            return_value={"type": "feature_store", "features": ["f1", "f2"]}
        )
        
        wrapper = PyfuncWrapper(
            settings=settings,
            trained_model=Mock(),
            trained_preprocessor=Mock(),
            trained_fetcher=Mock(),
            training_results={
                "hyperparameter_optimization": {"best_params": {"n_estimators": 100}},
                "training_methodology": {"cv_folds": 5}
            }
        )
        return wrapper
    
    def test_model_class_path_property(self, wrapper):
        """Test model_class_path property."""
        assert wrapper.model_class_path == "sklearn.ensemble.RandomForestClassifier"
    
    def test_loader_sql_snapshot_property(self, wrapper):
        """Test loader_sql_snapshot property."""
        assert wrapper.loader_sql_snapshot == "SELECT * FROM table"
    
    def test_fetcher_config_snapshot_property(self, wrapper):
        """Test fetcher_config_snapshot property."""
        config = wrapper.fetcher_config_snapshot
        assert config["type"] == "feature_store"
        assert config["features"] == ["f1", "f2"]
    
    def test_fetcher_config_snapshot_empty(self):
        """Test fetcher_config_snapshot when fetcher is None."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        settings.recipe.data.fetcher = None
        
        wrapper = PyfuncWrapper(
            settings=settings,
            trained_model=Mock(),
            trained_preprocessor=None,
            trained_fetcher=None
        )
        
        assert wrapper.fetcher_config_snapshot == {}
    
    @patch('yaml.dump')
    @patch.object(RecipeBuilder.build().__class__, 'model_dump')
    def test_recipe_yaml_snapshot_property(self, mock_model_dump, mock_yaml_dump, wrapper):
        """Test recipe_yaml_snapshot property."""
        mock_model_dump.return_value = {"name": "test_recipe"}
        mock_yaml_dump.return_value = "name: test_recipe\n"
        
        snapshot = wrapper.recipe_yaml_snapshot
        
        assert snapshot == "name: test_recipe\n"
        mock_yaml_dump.assert_called_once_with({"name": "test_recipe"})
    
    def test_hyperparameter_optimization_property(self, wrapper):
        """Test hyperparameter_optimization property."""
        hyper_opt = wrapper.hyperparameter_optimization
        assert hyper_opt == {"best_params": {"n_estimators": 100}}
    
    def test_training_methodology_property(self, wrapper):
        """Test training_methodology property."""
        methodology = wrapper.training_methodology
        assert methodology == {"cv_folds": 5}


class TestPyfuncWrapperPredict:
    """Test PyfuncWrapper predict functionality."""
    
    @pytest.fixture
    def wrapper(self):
        """Create a PyfuncWrapper instance with mock components."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        
        model = Mock()
        preprocessor = Mock()
        fetcher = Mock()
        
        wrapper = PyfuncWrapper(
            settings=settings,
            trained_model=model,
            trained_preprocessor=preprocessor,
            trained_fetcher=fetcher
        )
        return wrapper
    
    def test_predict_batch_mode(self, wrapper):
        """Test predict in batch mode."""
        # Setup input data
        input_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        # Setup mock behaviors
        fetched_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'feature3': [7, 8, 9]
        })
        preprocessed_df = pd.DataFrame({
            'feature1_scaled': [0.1, 0.2, 0.3],
            'feature2_scaled': [0.4, 0.5, 0.6],
            'feature3_scaled': [0.7, 0.8, 0.9]
        })
        predictions = np.array([0, 1, 0])
        
        wrapper.trained_fetcher.fetch.return_value = fetched_df
        wrapper.trained_preprocessor.transform.return_value = preprocessed_df
        wrapper.trained_model.predict.return_value = predictions
        
        # Call predict
        result = wrapper.predict(context=None, model_input=input_df)
        
        # Assertions
        wrapper.trained_fetcher.fetch.assert_called_once_with(input_df, run_mode="batch")
        wrapper.trained_preprocessor.transform.assert_called_once_with(fetched_df)
        wrapper.trained_model.predict.assert_called_once_with(preprocessed_df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)
        assert list(result['prediction']) == [0, 1, 0]
    
    def test_predict_serving_mode(self, wrapper):
        """Test predict in serving mode."""
        input_df = pd.DataFrame({
            'feature1': [1],
            'feature2': [4]
        })
        
        fetched_df = pd.DataFrame({
            'feature1': [1],
            'feature2': [4],
            'feature3': [7]
        })
        preprocessed_df = pd.DataFrame({
            'feature1_scaled': [0.1],
            'feature2_scaled': [0.4],
            'feature3_scaled': [0.7]
        })
        predictions = np.array([1])
        
        wrapper.trained_fetcher.fetch.return_value = fetched_df
        wrapper.trained_preprocessor.transform.return_value = preprocessed_df
        wrapper.trained_model.predict.return_value = predictions
        
        # Call predict with serving mode
        result = wrapper.predict(
            context=None,
            model_input=input_df,
            params={"run_mode": "serving"}
        )
        
        wrapper.trained_fetcher.fetch.assert_called_once_with(input_df, run_mode="serving")
        assert result.shape == (1, 1)
        assert result['prediction'].iloc[0] == 1
    
    def test_predict_without_preprocessor(self):
        """Test predict when preprocessor is None."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        
        model = Mock()
        fetcher = Mock()
        
        wrapper = PyfuncWrapper(
            settings=settings,
            trained_model=model,
            trained_preprocessor=None,
            trained_fetcher=fetcher
        )
        
        input_df = pd.DataFrame({'feature1': [1]})
        fetched_df = pd.DataFrame({'feature1': [1], 'feature2': [2]})
        predictions = np.array([0])
        
        fetcher.fetch.return_value = fetched_df
        model.predict.return_value = predictions
        
        result = wrapper.predict(context=None, model_input=input_df)
        
        fetcher.fetch.assert_called_once_with(input_df, run_mode="batch")
        model.predict.assert_called_once_with(fetched_df)  # Should use fetched_df directly
        assert result['prediction'].iloc[0] == 0
    
    def test_predict_with_non_dataframe_input(self, wrapper):
        """Test predict with non-DataFrame input (should convert to DataFrame)."""
        # Dictionary input
        input_dict = {'feature1': [1, 2], 'feature2': [3, 4]}
        
        fetched_df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        preprocessed_df = pd.DataFrame({'f1': [0.1, 0.2], 'f2': [0.3, 0.4]})
        predictions = np.array([1, 0])
        
        wrapper.trained_fetcher.fetch.return_value = fetched_df
        wrapper.trained_preprocessor.transform.return_value = preprocessed_df
        wrapper.trained_model.predict.return_value = predictions
        
        result = wrapper.predict(context=None, model_input=input_dict)
        
        # Should have converted dict to DataFrame
        call_args = wrapper.trained_fetcher.fetch.call_args[0][0]
        assert isinstance(call_args, pd.DataFrame)
        assert result.shape == (2, 1)


class TestPyfuncWrapperSchemaValidation:
    """Test PyfuncWrapper schema validation."""
    
    def test_validate_input_schema_no_schema(self):
        """Test validation when no schema is provided."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        
        wrapper = PyfuncWrapper(
            settings=settings,
            trained_model=Mock(),
            trained_preprocessor=Mock(),
            trained_fetcher=Mock(),
            data_schema=None  # No schema
        )
        
        input_df = pd.DataFrame({'any': [1, 2, 3]})
        
        # Should not raise any error
        wrapper._validate_input_schema(input_df)