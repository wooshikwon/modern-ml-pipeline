"""
Unit tests for CausalEvaluator.
Tests causal evaluation functionality with uplift/causal inference metrics.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.components.evaluator.modules.causal_evaluator import CausalEvaluator
from src.interface.base_evaluator import BaseEvaluator
from src.settings.recipe import DataInterface


class TestCausalEvaluatorInitialization:
    """Test CausalEvaluator initialization."""
    
    def test_causal_evaluator_inherits_base_evaluator(self):
        """Test that CausalEvaluator properly inherits from BaseEvaluator."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="outcome",
            feature_columns=["treatment", "feature1", "feature2"]
        )
        
        # Act
        evaluator = CausalEvaluator(data_interface)
        
        # Assert
        assert isinstance(evaluator, BaseEvaluator)
        assert isinstance(evaluator, CausalEvaluator)
    
    def test_init_stores_settings_and_task_type(self):
        """Test that initialization properly stores settings and task type."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="conversion",
            feature_columns=["treatment", "age", "gender", "location"]
        )
        
        # Act
        evaluator = CausalEvaluator(data_interface)
        
        # Assert
        assert evaluator.settings == data_interface
        assert evaluator.task_type == "causal"
        assert evaluator.settings.target_column == "conversion"
        assert evaluator.settings.feature_columns == ["treatment", "age", "gender", "location"]
    
    def test_init_minimal_configuration(self):
        """Test initialization with minimal required configuration."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="outcome"
        )
        
        # Act
        evaluator = CausalEvaluator(data_interface)
        
        # Assert
        assert evaluator.task_type == "causal"
        assert evaluator.settings.target_column == "outcome"
        assert evaluator.settings.feature_columns is None  # Optional field


class TestCausalEvaluatorEvaluate:
    """Test CausalEvaluator evaluate method."""
    
    def test_evaluate_returns_placeholder_metric(self):
        """Test that evaluate method returns the expected placeholder uplift metric."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="outcome",
            feature_columns=["treatment", "feature1", "feature2"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        # Mock causal model
        mock_model = Mock()
        
        # Test data for causal inference
        X = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        })
        y = pd.Series([0, 1, 0, 1, 1, 1])  # Outcomes
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - Check for placeholder metric
        expected_metrics = {
            "uplift_auc": 0.6
        }
        assert result == expected_metrics
    
    def test_evaluate_with_source_df_parameter(self):
        """Test evaluation with optional source_df parameter."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["customer_id"],
            task_type="causal",
            target_column="revenue_lift",
            feature_columns=["treatment", "customer_value"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        X = pd.DataFrame({
            'treatment': [0, 1, 0, 1],
            'customer_value': [100, 200, 150, 250]
        })
        y = pd.Series([50, 150, 75, 200])
        source_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'treatment': [0, 1, 0, 1],
            'customer_value': [100, 200, 150, 250],
            'revenue_lift': [50, 150, 75, 200],
            'metadata': ['A', 'B', 'C', 'D']
        })
        
        # Act
        result = evaluator.evaluate(mock_model, X, y, source_df=source_df)
        
        # Assert - should work normally even with source_df provided
        assert result["uplift_auc"] == 0.6
        assert len(result) == 1
    
    def test_evaluate_different_treatment_scenarios(self):
        """Test evaluation with different treatment assignment scenarios."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="click_through_rate",
            feature_columns=["treatment", "segment"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        # Test different treatment scenarios
        treatment_scenarios = [
            # Balanced treatment assignment
            (pd.DataFrame({
                'treatment': [0, 1, 0, 1, 0, 1],
                'segment': ['A', 'A', 'B', 'B', 'C', 'C']
            }), pd.Series([0.1, 0.3, 0.2, 0.4, 0.15, 0.35])),
            
            # Unbalanced treatment assignment  
            (pd.DataFrame({
                'treatment': [0, 0, 0, 1, 1, 0],
                'segment': ['A', 'B', 'C', 'A', 'B', 'C']
            }), pd.Series([0.1, 0.2, 0.15, 0.4, 0.5, 0.18])),
        ]
        
        for X, y in treatment_scenarios:
            mock_model = Mock()
            
            # Act
            result = evaluator.evaluate(mock_model, X, y)
            
            # Assert - should return consistent placeholder metric
            assert result == {"uplift_auc": 0.6}
    
    def test_evaluate_model_parameter_passed_correctly(self):
        """Test that model parameter is passed to evaluate method correctly."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="outcome",
            feature_columns=["treatment", "covariate"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        # Create a mock model with specific attributes to verify it's used
        mock_model = Mock()
        mock_model.model_type = "uplift_tree"
        mock_model.treatment_col = "treatment"
        
        X = pd.DataFrame({
            'treatment': [0, 1, 0, 1],
            'covariate': [1.0, 2.0, 3.0, 4.0]
        })
        y = pd.Series([0, 1, 0, 1])
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - placeholder implementation doesn't use model, but should not error
        assert result == {"uplift_auc": 0.6}
        # Model object should be accessible (even if not used in placeholder)
        assert mock_model.model_type == "uplift_tree"


class TestCausalEvaluatorMetrics:
    """Test CausalEvaluator metric calculations."""
    
    def test_evaluate_uplift_auc_metric_range(self):
        """Test that uplift_auc metric is in valid range."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="conversion",
            feature_columns=["treatment", "feature"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        X = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1],
            'feature': [1, 2, 3, 4, 5, 6]
        })
        y = pd.Series([0, 1, 1, 1, 0, 1])
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - uplift_auc should be in valid range [0, 1]
        assert 0 <= result["uplift_auc"] <= 1
        assert result["uplift_auc"] == 0.6  # Placeholder value
    
    def test_evaluate_consistent_metric_output(self):
        """Test that metric output is consistent across different calls."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="revenue",
            feature_columns=["treatment", "segment", "value"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        X = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1, 0, 1],
            'segment': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
            'value': [100, 200, 150, 250, 300, 400, 175, 275]
        })
        y = pd.Series([50, 120, 80, 140, 160, 220, 90, 150])
        
        # Act - multiple evaluations
        result1 = evaluator.evaluate(mock_model, X, y)
        result2 = evaluator.evaluate(mock_model, X, y)
        result3 = evaluator.evaluate(mock_model, X, y)
        
        # Assert - should return consistent results
        assert result1 == result2 == result3
        assert all(res["uplift_auc"] == 0.6 for res in [result1, result2, result3])


class TestCausalEvaluatorEdgeCases:
    """Test CausalEvaluator edge cases."""
    
    def test_evaluate_single_treatment_group(self):
        """Test evaluation when all samples belong to same treatment group."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="outcome",
            feature_columns=["treatment", "feature"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        # All control group (treatment = 0)
        X = pd.DataFrame({
            'treatment': [0, 0, 0, 0],
            'feature': [1, 2, 3, 4]
        })
        y = pd.Series([0, 1, 0, 1])
        
        # Act - placeholder implementation should still work
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert result == {"uplift_auc": 0.6}
    
    def test_evaluate_no_variation_in_outcome(self):
        """Test evaluation when outcome has no variation."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="constant_outcome",
            feature_columns=["treatment", "covariate"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        X = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0, 1],
            'covariate': [1, 2, 3, 4, 5, 6]
        })
        y = pd.Series([1, 1, 1, 1, 1, 1])  # No variation
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - placeholder should handle this case
        assert result == {"uplift_auc": 0.6}
    
    def test_evaluate_single_sample(self):
        """Test evaluation with single data sample."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="outcome",
            feature_columns=["treatment"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        X = pd.DataFrame({'treatment': [1]})
        y = pd.Series([1])
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - should handle single sample case
        assert result == {"uplift_auc": 0.6}


class TestCausalEvaluatorIntegration:
    """Test CausalEvaluator integration scenarios."""
    
    def test_evaluate_marketing_campaign_scenario(self):
        """Test evaluation with realistic marketing campaign uplift scenario."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["customer_id"],
            task_type="causal",
            target_column="purchase_probability",
            feature_columns=["email_campaign", "age", "previous_purchases", "engagement_score"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        # Realistic marketing campaign data
        mock_model = Mock()
        
        X = pd.DataFrame({
            'email_campaign': [0, 1, 0, 1, 0, 1, 0, 1],  # Treatment assignment
            'age': [25, 35, 45, 55, 30, 40, 50, 60],
            'previous_purchases': [2, 5, 1, 8, 3, 6, 0, 4],
            'engagement_score': [0.3, 0.7, 0.2, 0.9, 0.5, 0.8, 0.1, 0.6]
        })
        y = pd.Series([0.1, 0.4, 0.05, 0.6, 0.2, 0.5, 0.02, 0.35])  # Purchase probabilities
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert "uplift_auc" in result
        assert result["uplift_auc"] == 0.6  # Placeholder value
        
        # Should handle realistic causal inference data structure
        assert len(X) == len(y)
        assert 'email_campaign' in X.columns  # Treatment column present
    
    def test_evaluate_ab_test_scenario(self):
        """Test evaluation with A/B test scenario."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="conversion_rate",
            feature_columns=["variant", "user_segment", "device_type"]
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        # A/B test data
        X = pd.DataFrame({
            'variant': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # A/B variant
            'user_segment': ['new', 'returning', 'new', 'returning', 'new', 
                           'returning', 'new', 'returning', 'new', 'returning'],
            'device_type': ['mobile', 'desktop', 'mobile', 'desktop', 'tablet',
                          'mobile', 'desktop', 'tablet', 'mobile', 'desktop']
        })
        y = pd.Series([0.05, 0.12, 0.04, 0.15, 0.06, 0.10, 0.08, 0.18, 0.03, 0.14])
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert result == {"uplift_auc": 0.6}
        
        # Verify data structure is appropriate for causal inference
        assert X['variant'].nunique() == 2  # Binary treatment
        assert len(X) == 10
        assert len(y) == 10


class TestCausalEvaluatorSelfRegistration:
    """Test CausalEvaluator self-registration mechanism."""
    
    def test_causal_evaluator_self_registration(self):
        """Test that CausalEvaluator registers itself in EvaluatorRegistry."""
        # Act - Import triggers self-registration
        from src.components.evaluator.modules import causal_evaluator
        from src.components.evaluator.registry import EvaluatorRegistry
        
        # Assert
        assert "causal" in EvaluatorRegistry.evaluators
        assert EvaluatorRegistry.evaluators["causal"] == CausalEvaluator
        
        # Verify can create instance through registry
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="causal",
            target_column="outcome",
            feature_columns=["treatment", "covariate"]
        )
        instance = EvaluatorRegistry.create("causal", data_interface)
        assert isinstance(instance, CausalEvaluator)