"""
Unit tests for ClassificationEvaluator.
Tests classification evaluation functionality with sklearn metrics.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.components.evaluator.modules.classification_evaluator import ClassificationEvaluator
from src.interface.base_evaluator import BaseEvaluator
from src.settings.recipe import DataInterface


class TestClassificationEvaluatorInitialization:
    """Test ClassificationEvaluator initialization."""
    
    def test_classification_evaluator_inherits_base_evaluator(self):
        """Test that ClassificationEvaluator properly inherits from BaseEvaluator."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="target",
            feature_columns=["feature1", "feature2"]
        )
        
        # Act
        evaluator = ClassificationEvaluator(data_interface)
        
        # Assert
        assert isinstance(evaluator, BaseEvaluator)
        assert isinstance(evaluator, ClassificationEvaluator)
    
    def test_init_stores_settings_and_task_type(self):
        """Test that initialization properly stores settings and task type."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="label",
            feature_columns=["x1", "x2", "x3"]
        )
        
        # Act
        evaluator = ClassificationEvaluator(data_interface)
        
        # Assert
        assert evaluator.settings == data_interface
        assert evaluator.task_type == "classification"
        assert evaluator.settings.target_column == "label"
        assert evaluator.settings.feature_columns == ["x1", "x2", "x3"]
    
    def test_init_with_minimal_settings(self):
        """Test initialization with minimal settings for classification."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="category",
            feature_columns=["f1", "f2"]
        )
        
        # Act
        evaluator = ClassificationEvaluator(data_interface)
        
        # Assert
        assert evaluator.settings.task_type == "classification"
        assert evaluator.settings.target_column == "category"


class TestClassificationEvaluatorEvaluate:
    """Test ClassificationEvaluator evaluate method."""
    
    def test_evaluate_binary_classification_success(self):
        """Test successful binary classification evaluation with class-wise metrics."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="target",
            feature_columns=["feature1", "feature2"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 1, 0, 1])
        
        # Test data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y = pd.Series([0, 1, 1, 0, 1])  # Perfect predictions
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - should have accuracy and class-wise metrics
        assert "accuracy" in result
        assert result["accuracy"] == 1.0
        
        # Class 0 metrics
        assert "class_0_precision" in result
        assert "class_0_recall" in result  
        assert "class_0_f1" in result
        assert "class_0_support" in result
        
        # Class 1 metrics
        assert "class_1_precision" in result
        assert "class_1_recall" in result
        assert "class_1_f1" in result
        assert "class_1_support" in result
        
        # Perfect predictions should yield perfect class-wise metrics
        assert result["class_0_precision"] == 1.0
        assert result["class_0_recall"] == 1.0
        assert result["class_0_f1"] == 1.0
        assert result["class_1_precision"] == 1.0
        assert result["class_1_recall"] == 1.0
        assert result["class_1_f1"] == 1.0
        
        # Support counts
        assert result["class_0_support"] == 2  # Two class 0 samples
        assert result["class_1_support"] == 3  # Three class 1 samples
    
    def test_evaluate_multiclass_classification_success(self):
        """Test successful multiclass classification evaluation with class-wise metrics."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="category",
            feature_columns=["f1", "f2", "f3"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        # Mock model
        mock_model = Mock()
        predictions = np.array([0, 1, 2, 1, 0, 2])
        mock_model.predict.return_value = predictions
        
        # Test data
        X = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5, 6],
            'f2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'f3': [10, 20, 30, 40, 50, 60]
        })
        y = pd.Series([0, 1, 2, 2, 0, 1])  # Some misclassifications
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - should have accuracy and class-wise metrics for all classes
        assert "accuracy" in result
        assert result["accuracy"] == 4/6  # 4 correct out of 6: [0,1,2,2,0,1] vs [0,1,2,1,0,2]
        
        # Should have metrics for all 3 classes (0, 1, 2)
        for class_label in [0, 1, 2]:
            assert f"class_{class_label}_precision" in result
            assert f"class_{class_label}_recall" in result
            assert f"class_{class_label}_f1" in result
            assert f"class_{class_label}_support" in result
        
        # Support counts should match actual class distribution
        assert result["class_0_support"] == 2  # Two class 0 samples
        assert result["class_1_support"] == 2  # Two class 1 samples  
        assert result["class_2_support"] == 2  # Two class 2 samples
        
        # All metrics should be between 0 and 1
        for key, value in result.items():
            if "support" not in key:
                assert 0 <= value <= 1, f"{key} should be between 0 and 1"
    
    def test_evaluate_with_source_df_parameter(self):
        """Test evaluation with optional source_df parameter."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="target",
            feature_columns=["feature1"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0])
        
        X = pd.DataFrame({'feature1': [1, 2]})
        y = pd.Series([1, 0])
        source_df = pd.DataFrame({
            'id': [1, 2],
            'feature1': [1, 2],
            'target': [1, 0],
            'metadata': ['A', 'B']
        })
        
        # Act
        result = evaluator.evaluate(mock_model, X, y, source_df=source_df)
        
        # Assert - should work normally even with source_df provided
        assert result["accuracy"] == 1.0
        assert "class_0_precision" in result
        assert "class_1_precision" in result
        assert "class_0_support" in result
        assert "class_1_support" in result
        assert len([k for k in result.keys() if k.startswith("class_")]) == 8  # 4 metrics x 2 classes
    
    def test_evaluate_model_predict_called_correctly(self):
        """Test that model.predict is called with correct parameters."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="label",
            feature_columns=["x", "y"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 1])
        
        X_test = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [0.5, 1.5, 2.5]
        })
        y_test = pd.Series([0, 1, 1])
        
        # Act
        result = evaluator.evaluate(mock_model, X_test, y_test)
        
        # Assert
        mock_model.predict.assert_called_once()
        # Verify X_test was passed to predict
        call_args = mock_model.predict.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, X_test)
        
        # Verify result structure
        assert "accuracy" in result
        assert "class_0_precision" in result
        assert "class_1_precision" in result


class TestClassificationEvaluatorMetrics:
    """Test ClassificationEvaluator metric calculations."""
    
    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        mock_model = Mock()
        perfect_predictions = np.array([1, 0, 1, 0, 1])
        mock_model.predict.return_value = perfect_predictions
        
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series([1, 0, 1, 0, 1])  # Same as predictions
        
        # Act - Use real sklearn metrics for perfect case
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert result["accuracy"] == 1.0
        # Perfect predictions should yield perfect class-wise metrics
        assert result["class_0_precision"] == 1.0
        assert result["class_0_recall"] == 1.0
        assert result["class_0_f1"] == 1.0
        assert result["class_1_precision"] == 1.0
        assert result["class_1_recall"] == 1.0
        assert result["class_1_f1"] == 1.0
        
        # Support should match actual distribution
        assert result["class_0_support"] == 2  # Two class 0 samples
        assert result["class_1_support"] == 3  # Three class 1 samples
    
    def test_evaluate_worst_case_predictions(self):
        """Test evaluation with worst case predictions."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        mock_model = Mock()
        # Completely wrong predictions
        worst_predictions = np.array([0, 1, 0, 1, 0])
        mock_model.predict.return_value = worst_predictions
        
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series([1, 0, 1, 0, 1])  # Opposite of predictions
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - accuracy should be 0 for completely wrong predictions
        assert result["accuracy"] == 0.0
        
        # Should have class-wise metrics for both classes
        assert "class_0_precision" in result
        assert "class_0_recall" in result
        assert "class_0_f1" in result
        assert "class_1_precision" in result
        assert "class_1_recall" in result
        assert "class_1_f1" in result
        
        # Support should still be correct
        assert result["class_0_support"] == 2
        assert result["class_1_support"] == 3
    
    def test_evaluate_consistent_multiclass_results(self):
        """Test that evaluation produces consistent class-wise results for multiclass."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="category",
            feature_columns=["x"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 0, 1, 2])
        
        X = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6]})
        y = pd.Series([0, 1, 2, 0, 1, 2])  # Perfect predictions
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - perfect predictions should yield perfect metrics for all classes
        assert result["accuracy"] == 1.0
        
        for class_label in [0, 1, 2]:
            assert result[f"class_{class_label}_precision"] == 1.0
            assert result[f"class_{class_label}_recall"] == 1.0
            assert result[f"class_{class_label}_f1"] == 1.0
            assert result[f"class_{class_label}_support"] == 2  # Each class has 2 samples
        
        # Should have metrics for exactly 3 classes
        precision_keys = [k for k in result.keys() if "precision" in k]
        assert len(precision_keys) == 3


class TestClassificationEvaluatorErrorHandling:
    """Test ClassificationEvaluator error handling."""
    
    def test_evaluate_with_model_predict_error(self):
        """Test evaluation when model.predict raises an error."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.side_effect = ValueError("Model prediction failed")
        
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        
        # Act & Assert
        with pytest.raises(ValueError, match="Model prediction failed"):
            evaluator.evaluate(mock_model, X, y)
    
    def test_evaluate_with_mismatched_data_shapes(self):
        """Test evaluation with mismatched X and y shapes."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="target",
            feature_columns=["feature"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1])  # 2 predictions
        
        X = pd.DataFrame({'feature': [1, 2]})
        y = pd.Series([0, 1, 0])  # 3 true labels - mismatch!
        
        # Act & Assert - sklearn should handle this and raise appropriate error
        with pytest.raises(ValueError):
            evaluator.evaluate(mock_model, X, y)


class TestClassificationEvaluatorIntegration:
    """Test ClassificationEvaluator integration scenarios."""
    
    def test_evaluate_with_real_data_scenario(self):
        """Test evaluation with realistic data scenario."""
        # Arrange
        data_interface = DataInterface(
            task_type="classification",
            target_column="spam",
            feature_columns=["word_count", "exclamation_count", "caps_ratio"]
        )
        
        evaluator = ClassificationEvaluator(data_interface)
        
        # Realistic spam detection scenario
        mock_model = Mock()
        # Simulate some correct and some incorrect predictions
        mock_model.predict.return_value = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        
        X = pd.DataFrame({
            'word_count': [50, 10, 100, 75, 8, 15, 120, 5],
            'exclamation_count': [5, 0, 8, 3, 0, 1, 10, 0],
            'caps_ratio': [0.3, 0.05, 0.6, 0.2, 0.0, 0.1, 0.8, 0.02]
        })
        y = pd.Series([1, 0, 1, 0, 0, 0, 1, 0])  # 7 correct out of 8
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert result["accuracy"] == 0.875  # 7/8 correct
        
        # Should have class-wise metrics for both classes
        assert "class_0_precision" in result
        assert "class_0_recall" in result
        assert "class_0_f1" in result
        assert "class_1_precision" in result
        assert "class_1_recall" in result
        assert "class_1_f1" in result
        
        # Support should match actual class distribution
        assert "class_0_support" in result
        assert "class_1_support" in result
        
        # All metrics (except support) should be between 0 and 1
        for metric_name, metric_value in result.items():
            if "support" not in metric_name:
                assert 0 <= metric_value <= 1, f"{metric_name} should be between 0 and 1"


class TestClassificationEvaluatorSelfRegistration:
    """Test ClassificationEvaluator self-registration mechanism."""
    
    def test_classification_evaluator_self_registration(self):
        """Test that ClassificationEvaluator registers itself in EvaluatorRegistry."""
        # Act - Import triggers self-registration
        from src.components.evaluator.modules import classification_evaluator
        from src.components.evaluator.registry import EvaluatorRegistry
        
        # Assert
        assert "classification" in EvaluatorRegistry.evaluators
        assert EvaluatorRegistry.evaluators["classification"] == ClassificationEvaluator
        
        # Verify can create instance through registry
        data_interface = DataInterface(
            task_type="classification",
            target_column="target",
            feature_columns=["feature"]
        )
        instance = EvaluatorRegistry.create("classification", data_interface)
        assert isinstance(instance, ClassificationEvaluator)