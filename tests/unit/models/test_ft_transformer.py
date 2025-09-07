"""
Unit tests for FT-Transformer models.
Tests FT-Transformer classifier and regressor with sklearn interface.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock, MagicMock

from src.models.custom.ft_transformer import (
    FTTransformerWrapperBase, 
    FTTransformerClassifier, 
    FTTransformerRegressor
)


class TestFTTransformerWrapperBaseInitialization:
    """Test FT-Transformer base wrapper initialization."""
    
    def test_ft_transformer_base_default_initialization(self):
        """Test default initialization."""
        model = FTTransformerWrapperBase()
        
        # Assert default state
        assert model.model is None
        assert model.hyperparams == {}
        assert model._internal_preprocessor is None
        assert model.handles_own_preprocessing is True
    
    def test_ft_transformer_base_custom_hyperparams(self):
        """Test initialization with custom hyperparameters."""
        hyperparams = {
            'd_block': 64,
            'n_blocks': 4,
            'attention_n_heads': 8,
            'attention_dropout': 0.2,
            'ffn_dropout': 0.3
        }
        
        model = FTTransformerWrapperBase(**hyperparams)
        
        assert model.hyperparams == hyperparams
        assert model.model is None
        assert model._internal_preprocessor is None


class TestFTTransformerWrapperBasePreprocessing:
    """Test FT-Transformer preprocessing logic."""
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_initialize_and_fit_mixed_features(self, mock_ft_transformer_class):
        """Test initialization with mixed numerical and categorical features."""
        # Setup
        model = FTTransformerWrapperBase(d_block=32, n_blocks=2)
        
        # Create mixed data
        X = pd.DataFrame({
            'num_feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num_feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'cat_feature1': ['A', 'B', 'A', 'C', 'B'],
            'cat_feature2': ['X', 'Y', 'X', 'Z', 'Y']
        })
        y = pd.Series([0, 1, 0, 1, 0])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute
        model._initialize_and_fit(X, y, d_out=2)
        
        # Assertions
        assert model._internal_preprocessor is not None
        assert model.model == mock_ft_instance
        
        # Check FT-Transformer was called with correct parameters
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['n_cont_features'] == 2  # 2 numerical features
        assert len(call_args['cat_cardinalities']) == 2  # 2 categorical features
        assert call_args['d_out'] == 2
        assert call_args['d_block'] == 32
        assert call_args['n_blocks'] == 2
        
        # Check cardinalities include unknown handling (+1)
        # cat_feature1: A, B, C (3 categories + 1 unknown = 4)
        # cat_feature2: X, Y, Z (3 categories + 1 unknown = 4)
        assert call_args['cat_cardinalities'] == [4, 4]
        
        # Check model was fitted
        mock_ft_instance.fit.assert_called_once()
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_initialize_and_fit_numerical_only(self, mock_ft_transformer_class):
        """Test initialization with only numerical features."""
        model = FTTransformerWrapperBase()
        
        # Numerical data only
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'feature3': [7.0, 8.0, 9.0]
        })
        y = pd.Series([10.0, 20.0, 30.0])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute
        model._initialize_and_fit(X, y, d_out=1)
        
        # Check parameters
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['n_cont_features'] == 3
        assert call_args['cat_cardinalities'] == []  # No categorical features
        assert call_args['d_out'] == 1
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_initialize_and_fit_categorical_only(self, mock_ft_transformer_class):
        """Test initialization with only categorical features."""
        model = FTTransformerWrapperBase()
        
        # Categorical data only
        X = pd.DataFrame({
            'category1': ['A', 'B', 'C', 'A'],
            'category2': ['X', 'Y', 'X', 'Z']
        })
        y = pd.Series([0, 1, 1, 0])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute
        model._initialize_and_fit(X, y, d_out=2)
        
        # Check parameters
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['n_cont_features'] == 0  # No numerical features
        assert len(call_args['cat_cardinalities']) == 2
        assert call_args['d_out'] == 2
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_initialize_and_fit_hyperparameter_handling(self, mock_ft_transformer_class):
        """Test proper hyperparameter handling."""
        # Custom hyperparameters
        hyperparams = {
            'd_block': 128,
            'n_blocks': 6,
            'attention_n_heads': 16,
            'attention_dropout': 0.3,
            'ffn_d_hidden_multiplier': 8,
            'ffn_dropout': 0.2,
            'residual_dropout': 0.1,
            'extra_param': 'custom_value'  # Extra parameter
        }
        
        model = FTTransformerWrapperBase(**hyperparams)
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute
        model._initialize_and_fit(X, y, d_out=2)
        
        # Check all hyperparameters were passed
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['d_block'] == 128
        assert call_args['n_blocks'] == 6
        assert call_args['attention_n_heads'] == 16
        assert call_args['attention_dropout'] == 0.3
        assert call_args['ffn_d_hidden_multiplier'] == 8
        assert call_args['ffn_dropout'] == 0.2
        assert call_args['residual_dropout'] == 0.1
        assert call_args['extra_param'] == 'custom_value'
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_initialize_and_fit_n_heads_alias(self, mock_ft_transformer_class):
        """Test n_heads alias for attention_n_heads."""
        model = FTTransformerWrapperBase(n_heads=4)
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute
        model._initialize_and_fit(X, y, d_out=2)
        
        # Check n_heads was mapped to attention_n_heads
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['attention_n_heads'] == 4


class TestFTTransformerWrapperBasePredict:
    """Test FT-Transformer base wrapper prediction."""
    
    def test_predict_not_fitted_error(self):
        """Test prediction error when model is not fitted."""
        model = FTTransformerWrapperBase()
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(RuntimeError, match="모델이 학습되지 않았습니다"):
            model.predict(X)
    
    def test_predict_success_with_predict_method(self):
        """Test successful prediction using predict method."""
        # Setup fitted model
        model = FTTransformerWrapperBase()
        
        # Mock internal components
        mock_preprocessor = Mock()
        mock_preprocessor.transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        model._internal_preprocessor = mock_preprocessor
        
        mock_ft_model = Mock()
        mock_ft_model.predict.return_value = np.array([0.1, 0.8])
        # No predict_proba method
        del mock_ft_model.predict_proba
        model.model = mock_ft_model
        
        # Test data
        X_test = pd.DataFrame({'feature1': [5, 6]}, index=[10, 11])
        
        # Execute
        predictions = model.predict(X_test)
        
        # Assertions
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape == (2, 1)
        assert list(predictions.columns) == ['prediction']
        assert list(predictions.index) == [10, 11]
        np.testing.assert_array_equal(predictions['prediction'].values, [0.1, 0.8])
        
        # Check method calls
        mock_preprocessor.transform.assert_called_once()
        mock_ft_model.predict.assert_called_once()
    
    def test_predict_success_with_predict_proba_binary(self):
        """Test successful prediction using predict_proba for binary classification."""
        # Setup fitted model
        model = FTTransformerWrapperBase()
        
        # Mock internal components
        mock_preprocessor = Mock()
        mock_preprocessor.transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        model._internal_preprocessor = mock_preprocessor
        
        mock_ft_model = Mock()
        # Binary classification probabilities
        mock_ft_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])
        model.model = mock_ft_model
        
        # Test data
        X_test = pd.DataFrame({'feature1': [5, 6]}, index=[0, 1])
        
        # Execute
        predictions = model.predict(X_test)
        
        # Assertions
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape == (2, 1)
        # Should use class 1 probabilities for binary classification
        np.testing.assert_array_equal(predictions['prediction'].values, [0.3, 0.8])
        
        # Check method calls
        mock_preprocessor.transform.assert_called_once()
        mock_ft_model.predict_proba.assert_called_once()
    
    def test_predict_success_with_predict_proba_multiclass(self):
        """Test successful prediction using predict_proba for multiclass classification."""
        # Setup fitted model
        model = FTTransformerWrapperBase()
        
        # Mock internal components
        mock_preprocessor = Mock()
        mock_preprocessor.transform.return_value = np.array([[1.0], [2.0], [3.0]])
        model._internal_preprocessor = mock_preprocessor
        
        mock_ft_model = Mock()
        # Multiclass probabilities (3 classes)
        mock_ft_model.predict_proba.return_value = np.array([
            [0.1, 0.8, 0.1],  # Class 1 has highest prob
            [0.3, 0.2, 0.5],  # Class 2 has highest prob  
            [0.6, 0.3, 0.1]   # Class 0 has highest prob
        ])
        model.model = mock_ft_model
        
        # Test data
        X_test = pd.DataFrame({'feature1': [5, 6, 7]})
        
        # Execute
        predictions = model.predict(X_test)
        
        # Assertions
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape == (3, 1)
        # Should use argmax for multiclass
        np.testing.assert_array_equal(predictions['prediction'].values, [1, 2, 0])
    
    def test_predict_with_preprocessing_error(self):
        """Test prediction when preprocessing fails."""
        # Setup fitted model
        model = FTTransformerWrapperBase()
        
        # Mock internal components with error
        mock_preprocessor = Mock()
        mock_preprocessor.transform.side_effect = ValueError("Preprocessing failed")
        model._internal_preprocessor = mock_preprocessor
        model.model = Mock()
        
        X_test = pd.DataFrame({'feature1': [5, 6]})
        
        with pytest.raises(ValueError, match="Preprocessing failed"):
            model.predict(X_test)


class TestFTTransformerClassifier:
    """Test FT-Transformer classifier."""
    
    def test_ft_transformer_classifier_initialization(self):
        """Test FTTransformerClassifier initialization."""
        hyperparams = {'d_block': 64, 'n_blocks': 3}
        classifier = FTTransformerClassifier(**hyperparams)
        
        assert classifier.hyperparams == hyperparams
        assert classifier.model is None
        assert classifier.handles_own_preprocessing is True
        assert isinstance(classifier, FTTransformerWrapperBase)
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_ft_transformer_classifier_binary_fit(self, mock_ft_transformer_class):
        """Test FTTransformerClassifier binary classification fit."""
        classifier = FTTransformerClassifier(d_block=32)
        
        # Binary classification data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        y = pd.Series([0, 1, 0, 1, 0])  # 2 unique classes
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute
        result = classifier.fit(X, y)
        
        # Assertions
        assert result is classifier  # Returns self
        assert classifier.model == mock_ft_instance
        
        # Check d_out was set to number of unique classes
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['d_out'] == 2  # Binary classification
        assert call_args['n_cont_features'] == 1
        assert len(call_args['cat_cardinalities']) == 1
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_ft_transformer_classifier_multiclass_fit(self, mock_ft_transformer_class):
        """Test FTTransformerClassifier multiclass classification fit."""
        classifier = FTTransformerClassifier()
        
        # Multiclass classification data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [10, 20, 30, 40, 50, 60]
        })
        y = pd.Series([0, 1, 2, 0, 1, 2])  # 3 unique classes
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute
        result = classifier.fit(X, y)
        
        # Assertions
        assert result is classifier
        
        # Check d_out was set to number of unique classes
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['d_out'] == 3  # Multiclass
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_ft_transformer_classifier_fit_predict_pipeline(self, mock_ft_transformer_class):
        """Test complete FTTransformerClassifier fit-predict pipeline."""
        classifier = FTTransformerClassifier(d_block=16, n_blocks=2)
        
        # Training data
        X_train = pd.DataFrame({
            'numeric': [1, 2, 3, 4],
            'category': ['A', 'B', 'A', 'B']
        })
        y_train = pd.Series([0, 1, 0, 1])
        
        # Test data
        X_test = pd.DataFrame({
            'numeric': [2.5, 3.5],
            'category': ['A', 'B']
        }, index=[10, 11])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_instance.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute complete pipeline
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        
        # Assertions
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape == (2, 1)
        assert list(predictions.index) == [10, 11]
        # Binary classification should use class 1 probabilities
        np.testing.assert_array_equal(predictions['prediction'].values, [0.2, 0.7])


class TestFTTransformerRegressor:
    """Test FT-Transformer regressor."""
    
    def test_ft_transformer_regressor_initialization(self):
        """Test FTTransformerRegressor initialization."""
        hyperparams = {'d_block': 128, 'n_blocks': 4}
        regressor = FTTransformerRegressor(**hyperparams)
        
        assert regressor.hyperparams == hyperparams
        assert regressor.model is None
        assert regressor.handles_own_preprocessing is True
        assert isinstance(regressor, FTTransformerWrapperBase)
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_ft_transformer_regressor_fit(self, mock_ft_transformer_class):
        """Test FTTransformerRegressor fit."""
        regressor = FTTransformerRegressor(d_block=64)
        
        # Regression data
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'category': ['A', 'B', 'C', 'A', 'B']
        })
        y = pd.Series([1.5, 2.8, 4.1, 5.2, 6.9])  # Continuous target
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute
        result = regressor.fit(X, y)
        
        # Assertions
        assert result is regressor  # Returns self
        assert regressor.model == mock_ft_instance
        
        # Check d_out was set to 1 for regression
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['d_out'] == 1  # Regression always has d_out=1
        assert call_args['n_cont_features'] == 2
        assert len(call_args['cat_cardinalities']) == 1
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_ft_transformer_regressor_fit_predict_pipeline(self, mock_ft_transformer_class):
        """Test complete FTTransformerRegressor fit-predict pipeline."""
        regressor = FTTransformerRegressor(d_block=32, n_blocks=3)
        
        # Training data
        X_train = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 4, 6, 8, 10]
        })
        y_train = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        
        # Test data
        X_test = pd.DataFrame({
            'x1': [2.5, 3.5, 4.5],
            'x2': [5, 7, 9]
        }, index=[100, 101, 102])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_instance.predict.return_value = np.array([2.75, 3.85, 4.95])
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Execute complete pipeline
        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_test)
        
        # Assertions
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape == (3, 1)
        assert list(predictions.index) == [100, 101, 102]
        np.testing.assert_array_equal(predictions['prediction'].values, [2.75, 3.85, 4.95])


class TestFTTransformerIntegration:
    """Test FT-Transformer integration scenarios."""
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_classifier_vs_regressor_differences(self, mock_ft_transformer_class):
        """Test key differences between classifier and regressor."""
        # Same data for both
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'category': ['A', 'B', 'A', 'B']
        })
        
        # Classification target
        y_class = pd.Series([0, 1, 0, 1])
        # Regression target  
        y_reg = pd.Series([1.5, 2.8, 1.2, 2.9])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Test classifier
        classifier = FTTransformerClassifier()
        classifier.fit(X, y_class)
        
        # Test regressor
        regressor = FTTransformerRegressor()
        regressor.fit(X, y_reg)
        
        # Check d_out differences
        classifier_call = mock_ft_transformer_class.call_args_list[0][1]
        regressor_call = mock_ft_transformer_class.call_args_list[1][1]
        
        assert classifier_call['d_out'] == 2  # Number of unique classes
        assert regressor_call['d_out'] == 1   # Always 1 for regression
        
        # Other parameters should be the same
        assert classifier_call['n_cont_features'] == regressor_call['n_cont_features']
        assert classifier_call['cat_cardinalities'] == regressor_call['cat_cardinalities']
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_complex_feature_types(self, mock_ft_transformer_class):
        """Test with complex mix of feature types."""
        # Complex data with various types
        X = pd.DataFrame({
            'float_feature': [1.1, 2.2, 3.3, 4.4, 5.5],
            'int_feature': [10, 20, 30, 40, 50],
            'string_category': ['red', 'blue', 'green', 'red', 'blue'],
            'boolean_category': [True, False, True, False, True],
            'numeric_category': pd.Categorical([1, 2, 3, 1, 2])
        })
        y = pd.Series([0, 1, 2, 0, 1])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Test classifier
        classifier = FTTransformerClassifier()
        classifier.fit(X, y)
        
        # Check feature type detection
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['n_cont_features'] == 2  # float_feature, int_feature
        assert len(call_args['cat_cardinalities']) == 3  # string, boolean, numeric categories
        assert call_args['d_out'] == 3  # 3 unique classes


class TestFTTransformerEdgeCases:
    """Test FT-Transformer edge cases."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        classifier = FTTransformerClassifier()
        
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=int)
        
        # Should raise an error with empty data
        with pytest.raises(Exception):  # Various errors possible
            classifier.fit(X_empty, y_empty)
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_single_sample(self, mock_ft_transformer_class):
        """Test with single sample."""
        classifier = FTTransformerClassifier()
        
        X_single = pd.DataFrame({'feature': [1.0]})
        y_single = pd.Series([1])
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Should handle single sample
        result = classifier.fit(X_single, y_single)
        assert result is classifier
        
        # Check parameters
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['d_out'] == 1  # Single unique class
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_single_class_classification(self, mock_ft_transformer_class):
        """Test classification with single class."""
        classifier = FTTransformerClassifier()
        
        X = pd.DataFrame({'feature': [1, 2, 3, 4]})
        y = pd.Series([1, 1, 1, 1])  # All same class
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Should handle single class
        result = classifier.fit(X, y)
        assert result is classifier
        
        # Check d_out
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['d_out'] == 1  # Single unique class
    
    def test_inconsistent_features_predict(self):
        """Test prediction with inconsistent features."""
        # Setup fitted model with mock preprocessor
        classifier = FTTransformerClassifier()
        
        mock_preprocessor = Mock()
        mock_preprocessor.transform.side_effect = ValueError("Feature mismatch")
        classifier._internal_preprocessor = mock_preprocessor
        classifier.model = Mock()
        
        X_test = pd.DataFrame({'wrong_feature': [1, 2]})
        
        # Should raise error for inconsistent features
        with pytest.raises(ValueError, match="Feature mismatch"):
            classifier.predict(X_test)
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_extreme_hyperparameters(self, mock_ft_transformer_class):
        """Test with extreme hyperparameters."""
        # Very large model
        large_hyperparams = {
            'd_block': 1024,
            'n_blocks': 20,
            'attention_n_heads': 32,
            'ffn_d_hidden_multiplier': 16
        }
        
        # Very small model
        small_hyperparams = {
            'd_block': 4,
            'n_blocks': 1,
            'attention_n_heads': 1,
            'ffn_d_hidden_multiplier': 1
        }
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Test both extremes
        for hyperparams in [large_hyperparams, small_hyperparams]:
            regressor = FTTransformerRegressor(**hyperparams)
            
            X = pd.DataFrame({'feature': [1, 2, 3]})
            y = pd.Series([1.0, 2.0, 3.0])
            
            result = regressor.fit(X, y)
            assert result is regressor


class TestFTTransformerPerformance:
    """Test FT-Transformer performance characteristics."""
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_large_dataset(self, mock_ft_transformer_class):
        """Test with large dataset."""
        regressor = FTTransformerRegressor(d_block=32)
        
        # Large dataset
        n_samples = 10000
        X_large = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        y_large = pd.Series(np.random.randn(n_samples))
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Should handle large dataset without issues
        result = regressor.fit(X_large, y_large)
        assert result is regressor
        
        # Check model was fitted with large data
        mock_ft_instance.fit.assert_called_once()
        call_args = mock_ft_instance.fit.call_args[0]
        assert call_args[0].shape[0] == n_samples  # X_transformed
        assert len(call_args[1]) == n_samples      # y
    
    @patch('rtdl_revisiting_models.FTTransformer')
    def test_many_categorical_features(self, mock_ft_transformer_class):
        """Test with many categorical features."""
        classifier = FTTransformerClassifier()
        
        # Many categorical features
        n_categories = 20
        X_many_cats = pd.DataFrame()
        for i in range(n_categories):
            X_many_cats[f'cat_{i}'] = np.random.choice(['A', 'B', 'C'], 100)
        
        y = pd.Series(np.random.choice([0, 1], 100))
        
        # Mock FT-Transformer
        mock_ft_instance = Mock()
        mock_ft_transformer_class.return_value = mock_ft_instance
        
        # Should handle many categorical features
        result = classifier.fit(X_many_cats, y)
        assert result is classifier
        
        # Check cardinalities
        call_args = mock_ft_transformer_class.call_args[1]
        assert call_args['n_cont_features'] == 0
        assert len(call_args['cat_cardinalities']) == n_categories
        assert all(card == 4 for card in call_args['cat_cardinalities'])  # 3 categories + 1 unknown