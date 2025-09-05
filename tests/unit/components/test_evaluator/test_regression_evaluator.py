"""
Unit tests for RegressionEvaluator.
Tests regression evaluation functionality with sklearn metrics.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.components.evaluator.modules.regression_evaluator import RegressionEvaluator
from src.interface.base_evaluator import BaseEvaluator
from src.settings.recipe import DataInterface


class TestRegressionEvaluatorInitialization:
    """Test RegressionEvaluator initialization."""
    
    def test_regression_evaluator_inherits_base_evaluator(self):
        """Test that RegressionEvaluator properly inherits from BaseEvaluator."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="price",
            feature_columns=["size", "location", "age"]
        )
        
        # Act
        evaluator = RegressionEvaluator(data_interface)
        
        # Assert
        assert isinstance(evaluator, BaseEvaluator)
        assert isinstance(evaluator, RegressionEvaluator)
    
    def test_init_stores_settings_and_task_type(self):
        """Test that initialization properly stores settings and task type."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="sales",
            feature_columns=["advertising", "price", "season"]
        )
        
        # Act
        evaluator = RegressionEvaluator(data_interface)
        
        # Assert
        assert evaluator.settings == data_interface
        assert evaluator.task_type == "regression"
        assert evaluator.settings.target_column == "sales"
        assert evaluator.settings.feature_columns == ["advertising", "price", "season"]
    
    def test_init_minimal_configuration(self):
        """Test initialization with minimal required configuration."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="y"
        )
        
        # Act
        evaluator = RegressionEvaluator(data_interface)
        
        # Assert
        assert evaluator.task_type == "regression"
        assert evaluator.settings.target_column == "y"
        assert evaluator.settings.feature_columns is None  # Optional field


class TestRegressionEvaluatorEvaluate:
    """Test RegressionEvaluator evaluate method."""
    
    def test_evaluate_perfect_predictions_success(self):
        """Test successful regression evaluation with perfect predictions."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="target",
            feature_columns=["feature1", "feature2"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        # Mock model with perfect predictions
        mock_model = Mock()
        perfect_predictions = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mock_model.predict.return_value = perfect_predictions
        
        # Test data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])  # Same as predictions
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        expected_metrics = {
            "r2_score": 1.0,  # Perfect R² for perfect predictions
            "mean_squared_error": 0.0  # No error for perfect predictions
        }
        assert result == expected_metrics
    
    def test_evaluate_realistic_predictions(self):
        """Test evaluation with realistic predictions having some error."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="house_price",
            feature_columns=["sqft", "bedrooms", "age"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        # Mock model with some prediction errors
        mock_model = Mock()
        predictions = np.array([105.0, 195.0, 310.0, 405.0, 485.0])  # Close but not perfect
        mock_model.predict.return_value = predictions
        
        # Test data
        X = pd.DataFrame({
            'sqft': [1000, 1500, 2000, 2500, 3000],
            'bedrooms': [2, 3, 4, 4, 5],
            'age': [10, 5, 15, 8, 3]
        })
        y = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])  # True values
        
        # Act
        with patch('sklearn.metrics.r2_score', return_value=0.95) as mock_r2, \
             patch('sklearn.metrics.mean_squared_error', return_value=125.0) as mock_mse:
            
            result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        expected_metrics = {
            "r2_score": 0.95,
            "mean_squared_error": 125.0
        }
        assert result == expected_metrics
        
        # Verify sklearn functions were called correctly
        mock_r2.assert_called_once_with(y, predictions)
        mock_mse.assert_called_once_with(y, predictions)
    
    def test_evaluate_with_source_df_parameter(self):
        """Test evaluation with optional source_df parameter."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="target",
            feature_columns=["x"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.1, 2.2])
        
        X = pd.DataFrame({'x': [1, 2]})
        y = pd.Series([1.0, 2.0])
        source_df = pd.DataFrame({
            'id': [1, 2],
            'x': [1, 2],
            'target': [1.0, 2.0],
            'metadata': ['A', 'B']
        })
        
        # Act
        with patch('sklearn.metrics.r2_score', return_value=0.98), \
             patch('sklearn.metrics.mean_squared_error', return_value=0.025):
            
            result = evaluator.evaluate(mock_model, X, y, source_df=source_df)
        
        # Assert - should work normally even with source_df provided
        assert result["r2_score"] == 0.98
        assert result["mean_squared_error"] == 0.025
        assert len(result) == 2
    
    def test_evaluate_model_predict_called_correctly(self):
        """Test that model.predict is called with correct parameters."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="y",
            feature_columns=["x1", "x2"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.5, 2.5, 3.5])
        
        X_test = pd.DataFrame({
            'x1': [1.0, 2.0, 3.0],
            'x2': [0.5, 1.5, 2.5]
        })
        y_test = pd.Series([1.4, 2.6, 3.2])
        
        # Act
        with patch('sklearn.metrics.r2_score'), \
             patch('sklearn.metrics.mean_squared_error'):
            
            evaluator.evaluate(mock_model, X_test, y_test)
        
        # Assert
        mock_model.predict.assert_called_once()
        # Verify X_test was passed to predict
        call_args = mock_model.predict.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, X_test)


class TestRegressionEvaluatorMetrics:
    """Test RegressionEvaluator metric calculations."""
    
    def test_evaluate_negative_r2_score(self):
        """Test evaluation with negative R² score (worse than mean prediction)."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        # Very bad predictions - much worse than just predicting the mean
        bad_predictions = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        mock_model.predict.return_value = bad_predictions
        
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - R² should be very negative for such bad predictions
        assert result["r2_score"] < 0
        assert result["mean_squared_error"] > 0  # High MSE for bad predictions
    
    def test_evaluate_with_zero_variance_target(self):
        """Test evaluation when target has zero variance (constant values)."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="constant_target",
            feature_columns=["x"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        # Model predicts the constant value perfectly
        mock_model.predict.return_value = np.array([5.0, 5.0, 5.0, 5.0])
        
        X = pd.DataFrame({'x': [1, 2, 3, 4]})
        y = pd.Series([5.0, 5.0, 5.0, 5.0])  # Constant target
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert result["mean_squared_error"] == 0.0  # Perfect predictions
        # R² is undefined for constant target, but sklearn handles this
        assert "r2_score" in result
    
    def test_evaluate_with_large_errors(self):
        """Test evaluation with large prediction errors."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        # Predictions way off from true values
        mock_model.predict.return_value = np.array([1000.0, 2000.0, 3000.0])
        
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1.0, 2.0, 3.0])  # Much smaller true values
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert result["mean_squared_error"] > 1000000  # Very high MSE
        assert result["r2_score"] < 0  # Much worse than mean prediction


class TestRegressionEvaluatorErrorHandling:
    """Test RegressionEvaluator error handling."""
    
    def test_evaluate_with_model_predict_error(self):
        """Test evaluation when model.predict raises an error."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.side_effect = RuntimeError("Model failed")
        
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1.5, 2.5, 3.5])
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Model failed"):
            evaluator.evaluate(mock_model, X, y)
    
    def test_evaluate_with_mismatched_data_shapes(self):
        """Test evaluation with mismatched X and y shapes."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0])  # 2 predictions
        
        X = pd.DataFrame({'feature': [1, 2]})
        y = pd.Series([1.0, 2.0, 3.0])  # 3 true values - mismatch!
        
        # Act & Assert - sklearn should handle this and raise appropriate error
        with pytest.raises(ValueError):
            evaluator.evaluate(mock_model, X, y)
    
    def test_evaluate_with_nan_predictions(self):
        """Test evaluation when model predictions contain NaN values."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, np.nan, 3.0])
        
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1.1, 2.2, 3.3])
        
        # Act & Assert - sklearn metrics should handle NaN appropriately
        result = evaluator.evaluate(mock_model, X, y)
        
        # The exact behavior depends on sklearn's handling of NaN
        # But the function should not crash
        assert isinstance(result, dict)
        assert "r2_score" in result
        assert "mean_squared_error" in result


class TestRegressionEvaluatorIntegration:
    """Test RegressionEvaluator integration scenarios."""
    
    def test_evaluate_house_price_prediction_scenario(self):
        """Test evaluation with realistic house price prediction scenario."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="price",
            feature_columns=["sqft", "bedrooms", "bathrooms", "age"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        # Realistic house price prediction
        mock_model = Mock()
        # Model predictions with some realistic error
        predictions = np.array([485000, 320000, 750000, 425000, 680000])
        mock_model.predict.return_value = predictions
        
        X = pd.DataFrame({
            'sqft': [2000, 1200, 3500, 1800, 2800],
            'bedrooms': [3, 2, 5, 3, 4],
            'bathrooms': [2, 1, 4, 2, 3],
            'age': [10, 25, 5, 15, 8]
        })
        y = pd.Series([500000, 300000, 800000, 450000, 650000])  # True prices
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert "r2_score" in result
        assert "mean_squared_error" in result
        
        # Reasonable R² for house price prediction (should be positive)
        assert result["r2_score"] > 0.7, "R² should be reasonably high for good model"
        
        # MSE should be reasonable for house price scale
        assert result["mean_squared_error"] < 10000000000, "MSE should not be astronomically high"
    
    def test_evaluate_with_single_sample(self):
        """Test evaluation with single data sample."""
        # Arrange
        data_interface = DataInterface(
            task_type="regression",
            target_column="target",
            feature_columns=["x"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([2.5])
        
        X = pd.DataFrame({'x': [1]})
        y = pd.Series([2.0])
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert result["mean_squared_error"] == 0.25  # (2.5 - 2.0)² = 0.25
        # R² is undefined for single sample, but sklearn handles this
        assert "r2_score" in result


class TestRegressionEvaluatorSelfRegistration:
    """Test RegressionEvaluator self-registration mechanism."""
    
    def test_regression_evaluator_self_registration(self):
        """Test that RegressionEvaluator registers itself in EvaluatorRegistry."""
        # Act - Import triggers self-registration
        from src.components.evaluator.modules import regression_evaluator
        from src.components.evaluator.registry import EvaluatorRegistry
        
        # Assert
        assert "regression" in EvaluatorRegistry.evaluators
        assert EvaluatorRegistry.evaluators["regression"] == RegressionEvaluator
        
        # Verify can create instance through registry
        data_interface = DataInterface(
            task_type="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        instance = EvaluatorRegistry.create("regression", data_interface)
        assert isinstance(instance, RegressionEvaluator)