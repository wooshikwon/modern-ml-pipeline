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
from tests.helpers.builders import DataFrameBuilder, RecipeBuilder


def DataInterface(
    entity_columns=None,
    task_choice=None,
    target_column=None,
    feature_columns=None,
    treatment_column=None,
):
    overrides = {}
    if entity_columns is not None:
        overrides['data.data_interface.entity_columns'] = entity_columns
    if target_column is not None:
        overrides['data.data_interface.target_column'] = target_column
    if feature_columns is not None:
        overrides['data.data_interface.feature_columns'] = feature_columns
    if treatment_column is not None:
        overrides['data.data_interface.treatment_column'] = treatment_column
    task = task_type or 'regression'
    recipe = RecipeBuilder.build(task_choice=task, **overrides)
    return recipe.data.data_interface


class TestRegressionEvaluatorInitialization:
    """Test RegressionEvaluator initialization."""
    
    def test_regression_evaluator_inherits_base_evaluator(self):
        """Test that RegressionEvaluator properly inherits from BaseEvaluator."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["property_id"],
            task_choice="regression",
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
            entity_columns=["product_id"],
            task_choice="regression",
            target_column="sales",
            feature_columns=["advertising", "price", "season"]
        )
        
        # Act
        evaluator = RegressionEvaluator(data_interface)
        
        # Assert
        assert evaluator.settings == data_interface
        assert evaluator.task_choice == "regression"
        assert evaluator.settings.target_column == "sales"
        assert evaluator.settings.feature_columns == ["advertising", "price", "season"]
    
    def test_init_minimal_configuration(self):
        """Test initialization with minimal required configuration."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="y"
        )
        
        # Act
        evaluator = RegressionEvaluator(data_interface)
        
        # Assert
        assert evaluator.task_choice == "regression"
        assert evaluator.settings.target_column == "y"
        assert evaluator.settings.feature_columns is None  # Optional field


class TestRegressionEvaluatorEvaluate:
    """Test RegressionEvaluator evaluate method."""
    
    def test_evaluate_perfect_predictions_success(self):
        """Test successful regression evaluation with perfect predictions."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="target",
            feature_columns=["feature1", "feature2"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        # Mock model with perfect predictions
        mock_model = Mock()
        perfect_predictions = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mock_model.predict.return_value = perfect_predictions
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=5,
            n_features=2,
            add_entity_column=False
        )
        X = df[['feature_0', 'feature_1']]
        X.columns = ['feature1', 'feature2']
        # For perfect predictions test, we'll use the model's predictions as y
        y = pd.Series(perfect_predictions)  # Will match predictions exactly
        
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
            entity_columns=["house_id"],
            task_choice="regression",
            target_column="house_price",
            feature_columns=["sqft", "bedrooms", "age"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        # Mock model with some prediction errors
        mock_model = Mock()
        predictions = np.array([105.0, 195.0, 310.0, 405.0, 485.0])  # Close but not perfect
        mock_model.predict.return_value = predictions
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=5,
            n_features=3,
            add_entity_column=False
        )
        X = df[['feature_0', 'feature_1', 'feature_2']]
        X.columns = ['sqft', 'bedrooms', 'age']
        y = df['target']  # Use generated target values
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - Check that metrics are calculated
        # With generated data, exact values will vary
        assert "r2_score" in result
        assert "mean_squared_error" in result
        # Check that MSE is positive
        assert result["mean_squared_error"] >= 0
        # R2 score can be negative if predictions are worse than mean
        assert isinstance(result["r2_score"], float)
    
    def test_evaluate_with_source_df_parameter(self):
        """Test evaluation with optional source_df parameter."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="target",
            feature_columns=["x"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.1, 2.2])
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=2,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['x']
        y = df['target']
        source_df = X.copy()
        source_df['id'] = [1, 2]
        source_df['target'] = y
        source_df['metadata'] = ['A', 'B']
        
        # Act
        result = evaluator.evaluate(mock_model, X, y, source_df=source_df)
        
        # Assert - should work normally even with source_df provided
        # With generated data, exact values will vary
        assert "r2_score" in result
        assert "mean_squared_error" in result
        # Check that metrics are calculated
        assert result["mean_squared_error"] >= 0
        # R2 score can be negative if predictions are worse than mean
        assert isinstance(result["r2_score"], float)
    
    def test_evaluate_model_predict_called_correctly(self):
        """Test that model.predict is called with correct parameters."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="y",
            feature_columns=["x1", "x2"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.5, 2.5, 3.5])
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=3,
            n_features=2,
            add_entity_column=False
        )
        X_test = df[['feature_0', 'feature_1']]
        X_test.columns = ['x1', 'x2']
        y_test = df['target']
        
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
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        # Very bad predictions - much worse than just predicting the mean
        bad_predictions = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        mock_model.predict.return_value = bad_predictions
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=5,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['feature']
        y = df['target'] * 100 + 200  # Scale to larger values for test
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - R² should be very negative for such bad predictions
        assert result["r2_score"] < 0
        assert result["mean_squared_error"] > 0  # High MSE for bad predictions
    
    def test_evaluate_with_zero_variance_target(self):
        """Test evaluation when target has zero variance (constant values)."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="constant_target",
            feature_columns=["x"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        # Model predicts the constant value perfectly
        mock_model.predict.return_value = np.array([5.0, 5.0, 5.0, 5.0])
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=4,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['x']
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
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        # Predictions way off from true values
        mock_model.predict.return_value = np.array([1000.0, 2000.0, 3000.0])
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=3,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['feature']
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
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.side_effect = RuntimeError("Model failed")
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=3,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['feature']
        y = df['target']
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Model failed"):
            evaluator.evaluate(mock_model, X, y)
    
    def test_evaluate_with_mismatched_data_shapes(self):
        """Test evaluation with mismatched X and y shapes."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0])  # 2 predictions
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=2,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['feature']
        y = pd.Series([1.0, 2.0, 3.0])  # 3 true values - mismatch!
        
        # Act & Assert - sklearn should handle this and raise appropriate error
        with pytest.raises(ValueError):
            evaluator.evaluate(mock_model, X, y)
    
    def test_evaluate_with_nan_predictions(self):
        """Test evaluation when model predictions contain NaN values."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, np.nan, 3.0])
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=3,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['feature']
        y = df['target']
        
        # Act & Assert - sklearn will raise an error for NaN predictions
        with pytest.raises(ValueError, match="Input (y_pred|contains NaN)"):
            evaluator.evaluate(mock_model, X, y)


class TestRegressionEvaluatorIntegration:
    """Test RegressionEvaluator integration scenarios."""
    
    def test_evaluate_house_price_prediction_scenario(self):
        """Test evaluation with realistic house price prediction scenario."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["house_id"],
            task_choice="regression",
            target_column="price",
            feature_columns=["sqft", "bedrooms", "bathrooms", "age"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        # Realistic house price prediction
        mock_model = Mock()
        # Model predictions with some realistic error
        predictions = np.array([485000, 320000, 750000, 425000, 680000])
        mock_model.predict.return_value = predictions
        
        # Use DataFrameBuilder for realistic house price data
        df = DataFrameBuilder.build_regression_data(
            n_samples=5,
            n_features=4,
            add_entity_column=False
        )
        X = df[['feature_0', 'feature_1', 'feature_2', 'feature_3']]
        X.columns = ['sqft', 'bedrooms', 'bathrooms', 'age']
        # Scale features to realistic ranges
        X = X.copy()
        X['sqft'] = (X['sqft'] - X['sqft'].min()) / (X['sqft'].max() - X['sqft'].min()) * 2300 + 1200
        X['bedrooms'] = ((X['bedrooms'] - X['bedrooms'].min()) / (X['bedrooms'].max() - X['bedrooms'].min()) * 3 + 2).astype(int)
        X['bathrooms'] = ((X['bathrooms'] - X['bathrooms'].min()) / (X['bathrooms'].max() - X['bathrooms'].min()) * 3 + 1).astype(int)
        X['age'] = ((X['age'] - X['age'].min()) / (X['age'].max() - X['age'].min()) * 20 + 5).astype(int)
        # Generate realistic house prices based on features
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
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="target",
            feature_columns=["x"]
        )
        
        evaluator = RegressionEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([2.5])
        
        # Use DataFrameBuilder for single sample
        df = DataFrameBuilder.build_regression_data(
            n_samples=1,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['x']
        y = df['target']
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - With generated data, exact values will vary
        assert "mean_squared_error" in result
        assert result["mean_squared_error"] >= 0  # MSE should be non-negative
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
            entity_columns=["user_id"],
            task_choice="regression",
            target_column="target",
            feature_columns=["feature"]
        )
        instance = EvaluatorRegistry.create("regression", data_interface)
        assert isinstance(instance, RegressionEvaluator)