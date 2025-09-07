"""
Unit tests for TimeSeriesEvaluator.
Tests timeseries evaluation functionality with regression metrics plus MAPE/SMAPE.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.components.evaluator.modules.timeseries_evaluator import TimeSeriesEvaluator
from src.interface.base_evaluator import BaseEvaluator
from src.settings.recipe import DataInterface
from tests.helpers.builders import DataFrameBuilder, RecipeBuilder


def DataInterface(
    entity_columns=None,
    task_choice=None,
    target_column=None,
    feature_columns=None,
    timestamp_column=None,
):
    overrides = {}
    if entity_columns is not None:
        overrides['data.data_interface.entity_columns'] = entity_columns
    if target_column is not None:
        overrides['data.data_interface.target_column'] = target_column
    if feature_columns is not None:
        overrides['data.data_interface.feature_columns'] = feature_columns
    if timestamp_column is not None:
        overrides['data.data_interface.timestamp_column'] = timestamp_column
    task = task_type or 'timeseries'
    recipe = RecipeBuilder.build(task_choice=task, **overrides)
    return recipe.data.data_interface


class TestTimeSeriesEvaluatorInitialization:
    """Test TimeSeriesEvaluator initialization."""
    
    def test_timeseries_evaluator_inherits_base_evaluator(self):
        """Test that TimeSeriesEvaluator properly inherits from BaseEvaluator."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["store_id"],
            task_choice="timeseries",
            target_column="sales",
            feature_columns=["temperature", "promotion"],
            timestamp_column="date"
        )
        
        # Act
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Assert
        assert isinstance(evaluator, BaseEvaluator)
        assert isinstance(evaluator, TimeSeriesEvaluator)
    
    def test_init_stores_settings_and_task_type(self):
        """Test that initialization properly stores settings and task type."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["product_id"],
            task_choice="timeseries",
            target_column="demand",
            feature_columns=["price", "inventory"],
            timestamp_column="timestamp"
        )
        
        # Act
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Assert
        assert evaluator.settings == data_interface
        assert evaluator.task_choice == "timeseries"
        assert evaluator.settings.target_column == "demand"
        assert evaluator.settings.feature_columns == ["price", "inventory"]
        assert evaluator.settings.timestamp_column == "timestamp"
    
    def test_init_minimal_configuration(self):
        """Test initialization with minimal required configuration."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="y",
            timestamp_column="date"
        )
        
        # Act
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Assert
        assert evaluator.task_choice == "timeseries"
        assert evaluator.settings.target_column == "y"
        assert evaluator.settings.timestamp_column == "date"
        assert evaluator.settings.feature_columns is None  # Optional field


class TestTimeSeriesEvaluatorEvaluate:
    """Test TimeSeriesEvaluator evaluate method."""
    
    def test_evaluate_perfect_predictions_success(self):
        """Test successful timeseries evaluation with perfect predictions."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            timestamp_column="date"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
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
            "mean_squared_error": 0.0,  # No error for perfect predictions
            "mape": 0.0,  # Perfect MAPE
            "smape": 0.0  # Perfect SMAPE
        }
        assert result == expected_metrics
    
    def test_evaluate_realistic_predictions(self):
        """Test evaluation with realistic predictions having some error."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["house_id"],
            task_choice="timeseries",
            target_column="electricity_usage",
            feature_columns=["temperature", "hour_of_day", "day_of_week"],
            timestamp_column="timestamp"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Mock model with some prediction errors
        mock_model = Mock()
        predictions = np.array([95.0, 185.0, 305.0, 415.0, 495.0])  # Close but not perfect
        mock_model.predict.return_value = predictions
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=5,
            n_features=3,
            add_entity_column=False
        )
        X = df[['feature_0', 'feature_1', 'feature_2']]
        X.columns = ['temperature', 'hour_of_day', 'day_of_week']
        y = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])  # Actual usage values
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - Check that all metrics are calculated
        assert "r2_score" in result
        assert "mean_squared_error" in result
        assert "mape" in result
        assert "smape" in result
        # Check that MSE is positive
        assert result["mean_squared_error"] >= 0
        # R2 score can be negative if predictions are worse than mean
        assert isinstance(result["r2_score"], float)
        # MAPE and SMAPE should be positive
        assert result["mape"] >= 0
        assert result["smape"] >= 0
    
    def test_evaluate_with_source_df_parameter(self):
        """Test evaluation with optional source_df parameter."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            feature_columns=["x"],
            timestamp_column="date"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([10.1, 20.2])
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_regression_data(
            n_samples=2,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['x']
        y = pd.Series([10.0, 20.0])  # Known values for consistent MAPE/SMAPE calculation
        source_df = X.copy()
        source_df['id'] = [1, 2]
        source_df['target'] = y
        source_df['metadata'] = ['A', 'B']
        
        # Act
        result = evaluator.evaluate(mock_model, X, y, source_df=source_df)
        
        # Assert - should work normally even with source_df provided
        assert "r2_score" in result
        assert "mean_squared_error" in result
        assert "mape" in result
        assert "smape" in result
        # Check that metrics are calculated
        assert result["mean_squared_error"] >= 0
        assert result["mape"] >= 0
        assert result["smape"] >= 0
    
    def test_evaluate_model_predict_called_correctly(self):
        """Test that model.predict is called with correct parameters."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="y",
            feature_columns=["x1", "x2"],
            timestamp_column="timestamp"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
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
        y_test = pd.Series([1.0, 2.0, 3.0])
        
        # Act
        with patch('sklearn.metrics.r2_score'), \
             patch('sklearn.metrics.mean_squared_error'), \
             patch.object(evaluator, '_calculate_mape'), \
             patch.object(evaluator, '_calculate_smape'):
            
            evaluator.evaluate(mock_model, X_test, y_test)
        
        # Assert
        mock_model.predict.assert_called_once()
        # Verify X_test was passed to predict
        call_args = mock_model.predict.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, X_test)


class TestTimeSeriesEvaluatorMetrics:
    """Test TimeSeriesEvaluator metric calculations."""
    
    def test_evaluate_negative_r2_score(self):
        """Test evaluation with negative R² score (worse than mean prediction)."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            feature_columns=["feature"],
            timestamp_column="date"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
        mock_model = Mock()
        # Very bad predictions - much worse than just predicting the mean
        bad_predictions = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        mock_model.predict.return_value = bad_predictions
        
        # Test with positive target values to make MAPE/SMAPE calculations meaningful
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])  # Large positive values
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - R² should be very negative for such bad predictions
        assert result["r2_score"] < 0
        assert result["mean_squared_error"] > 0  # High MSE for bad predictions
        assert result["mape"] > 90  # Very high MAPE (close to 100% error)
        assert result["smape"] > 90  # Very high SMAPE
    
    def test_calculate_mape_normal_values(self):
        """Test MAPE calculation with normal positive values."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            timestamp_column="date"
        )
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Test data with known MAPE result
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([90.0, 210.0, 270.0, 420.0])
        # Expected MAPE: (|100-90|/100 + |200-210|/200 + |300-270|/300 + |400-420|/400) / 4 * 100
        # = (10/100 + 10/200 + 30/300 + 20/400) / 4 * 100
        # = (0.1 + 0.05 + 0.1 + 0.05) / 4 * 100 = 0.3 / 4 * 100 = 7.5
        
        # Act
        result = evaluator._calculate_mape(y_true, y_pred)
        
        # Assert
        expected_mape = 7.5
        assert abs(result - expected_mape) < 0.001  # Close to expected value
    
    def test_calculate_mape_with_zero_values(self):
        """Test MAPE calculation when true values contain zeros."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            timestamp_column="date"
        )
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Test data with some zero true values
        y_true = np.array([0.0, 200.0, 0.0, 400.0])  # Contains zeros
        y_pred = np.array([10.0, 210.0, 5.0, 420.0])
        
        # Act
        result = evaluator._calculate_mape(y_true, y_pred)
        
        # Assert - Should handle zeros by skipping them in calculation
        # MAPE = (|200-210|/200 + |400-420|/400) / 2 * 100 = (10/200 + 20/400) / 2 * 100 = 3.75
        expected_mape = 3.75
        assert abs(result - expected_mape) < 0.001
    
    def test_calculate_smape_normal_values(self):
        """Test SMAPE calculation with normal positive values."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            timestamp_column="date"
        )
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Test data with known SMAPE result
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # Expected SMAPE: (|100-110|/(100+110) + |200-180|/(200+180)) / 2 * 100
        # = (10/210 + 20/380) / 2 * 100 = (0.0476 + 0.0526) / 2 * 100 = 5.01
        
        # Act
        result = evaluator._calculate_smape(y_true, y_pred)
        
        # Assert
        expected_smape = 5.01
        assert abs(result - expected_smape) < 0.1  # Close to expected value
    
    def test_calculate_smape_with_zero_values(self):
        """Test SMAPE calculation when both true and predicted values are zero."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            timestamp_column="date"
        )
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Test data with zeros
        y_true = np.array([0.0, 100.0, 0.0])
        y_pred = np.array([0.0, 90.0, 5.0])
        
        # Act
        result = evaluator._calculate_smape(y_true, y_pred)
        
        # Assert - Should handle zeros appropriately
        # For (0,0) pair, should be 0% error
        # For (100,90), SMAPE = |100-90|/(100+90) * 100 = 10/190 * 100 = 5.26
        # For (0,5), SMAPE = |0-5|/(0+5) * 100 = 5/5 * 100 = 100
        # Average: (0 + 5.26 + 100) / 3 = 35.09
        assert result > 30 and result < 40  # Should be around 35


class TestTimeSeriesEvaluatorErrorHandling:
    """Test TimeSeriesEvaluator error handling."""
    
    def test_evaluate_with_model_predict_error(self):
        """Test evaluation when model.predict raises an error."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            feature_columns=["feature"],
            timestamp_column="date"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
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
            task_choice="timeseries",
            target_column="target",
            feature_columns=["feature"],
            timestamp_column="date"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
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
            task_choice="timeseries",
            target_column="target",
            feature_columns=["feature"],
            timestamp_column="date"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
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
    
    def test_calculate_mape_all_zeros(self):
        """Test MAPE calculation when all true values are zero."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            timestamp_column="date"
        )
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Test data with all zeros in true values
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        # Act
        result = evaluator._calculate_mape(y_true, y_pred)
        
        # Assert - Should return 0 when no non-zero values to calculate from
        assert result == 0.0


class TestTimeSeriesEvaluatorIntegration:
    """Test TimeSeriesEvaluator integration scenarios."""
    
    def test_evaluate_electricity_demand_forecasting_scenario(self):
        """Test evaluation with realistic electricity demand forecasting scenario."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["household_id"],
            task_choice="timeseries",
            target_column="electricity_usage",
            feature_columns=["temperature", "hour", "day_type", "appliance_count"],
            timestamp_column="timestamp"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
        # Realistic electricity usage prediction
        mock_model = Mock()
        # Model predictions with some realistic error (kWh usage)
        predictions = np.array([12.5, 8.2, 15.7, 22.1, 18.3])
        mock_model.predict.return_value = predictions
        
        # Use DataFrameBuilder for realistic electricity data
        df = DataFrameBuilder.build_regression_data(
            n_samples=5,
            n_features=4,
            add_entity_column=False
        )
        X = df[['feature_0', 'feature_1', 'feature_2', 'feature_3']]
        X.columns = ['temperature', 'hour', 'day_type', 'appliance_count']
        
        # Realistic electricity usage values (kWh)
        y = pd.Series([12.0, 8.5, 16.0, 21.5, 19.0])  # True usage values
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert "r2_score" in result
        assert "mean_squared_error" in result
        assert "mape" in result
        assert "smape" in result
        
        # Reasonable metrics for electricity forecasting
        assert result["r2_score"] > 0.8, "R² should be high for good forecasting model"
        assert result["mape"] < 10.0, "MAPE should be low for good forecasting"
        assert result["smape"] < 10.0, "SMAPE should be low for good forecasting"
        assert result["mean_squared_error"] < 5.0, "MSE should be reasonable for electricity usage scale"
    
    def test_evaluate_with_single_sample(self):
        """Test evaluation with single data sample."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            feature_columns=["x"],
            timestamp_column="date"
        )
        
        evaluator = TimeSeriesEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([10.5])
        
        # Use DataFrameBuilder for single sample
        df = DataFrameBuilder.build_regression_data(
            n_samples=1,
            n_features=1,
            add_entity_column=False
        )
        X = df[['feature_0']]
        X.columns = ['x']
        y = pd.Series([10.0])  # Known value
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - All metrics should be calculated for single sample
        assert "mean_squared_error" in result
        assert "mape" in result
        assert "smape" in result
        assert result["mean_squared_error"] >= 0  # MSE should be non-negative
        assert result["mape"] >= 0  # MAPE should be non-negative
        assert result["smape"] >= 0  # SMAPE should be non-negative
        # R² is undefined for single sample, but sklearn handles this
        assert "r2_score" in result


class TestTimeSeriesEvaluatorSelfRegistration:
    """Test TimeSeriesEvaluator self-registration mechanism."""
    
    def test_timeseries_evaluator_self_registration(self):
        """Test that TimeSeriesEvaluator registers itself in EvaluatorRegistry."""
        # Act - Import triggers self-registration
        from src.components.evaluator.modules import timeseries_evaluator
        from src.components.evaluator.registry import EvaluatorRegistry
        
        # Assert
        assert "timeseries" in EvaluatorRegistry.evaluators
        assert EvaluatorRegistry.evaluators["timeseries"] == TimeSeriesEvaluator
        
        # Verify can create instance through registry
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="timeseries",
            target_column="target",
            feature_columns=["feature"],
            timestamp_column="timestamp"
        )
        instance = EvaluatorRegistry.create("timeseries", data_interface)
        assert isinstance(instance, TimeSeriesEvaluator)