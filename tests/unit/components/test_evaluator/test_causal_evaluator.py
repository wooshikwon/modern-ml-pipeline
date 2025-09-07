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
    task = task_type or 'causal'
    recipe = RecipeBuilder.build(task_choice=task, **overrides)
    return recipe.data.data_interface


class TestCausalEvaluatorInitialization:
    """Test CausalEvaluator initialization."""
    
    def test_causal_evaluator_inherits_base_evaluator(self):
        """Test that CausalEvaluator properly inherits from BaseEvaluator."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
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
            task_choice="causal",
            target_column="conversion",
            feature_columns=["treatment", "age", "gender", "location"]
        )
        
        # Act
        evaluator = CausalEvaluator(data_interface)
        
        # Assert
        assert evaluator.settings == data_interface
        assert evaluator.task_choice == "causal"
        assert evaluator.settings.target_column == "conversion"
        assert evaluator.settings.feature_columns == ["treatment", "age", "gender", "location"]
    
    def test_init_minimal_configuration(self):
        """Test initialization with minimal required configuration."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="outcome"
        )
        
        # Act
        evaluator = CausalEvaluator(data_interface)
        
        # Assert
        assert evaluator.task_choice == "causal"
        assert evaluator.settings.target_column == "outcome"
        assert evaluator.settings.feature_columns is None  # Optional field


class TestCausalEvaluatorEvaluate:
    """Test CausalEvaluator evaluate method."""
    
    def test_evaluate_returns_placeholder_metric(self):
        """Test that evaluate method returns the expected placeholder uplift metric."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="outcome",
            feature_columns=["feature1", "feature2"],
            treatment_column="treatment"  # Add treatment_column for causal task
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        # Mock causal model
        mock_model = Mock()
        
        # Use DataFrameBuilder for causal inference test data
        df = DataFrameBuilder.build_causal_data(
            n_samples=6,
            treatment_effect=2.0,
            n_features=2,
            add_entity_column=False
        )
        X = df[['treatment', 'feature_0', 'feature_1']]
        X.columns = ['treatment', 'feature1', 'feature2']  # Rename for test compatibility
        y = df['outcome_binary']  # Use binary outcome for this test
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - Check for expected metrics
        assert "ate" in result  # Average Treatment Effect
        assert "ate_std" in result  # ATE standard deviation
        assert "treatment_effect_significance" in result  # Significance test
        assert "uplift_auc" in result  # Uplift AUC (should be 0.0 for mock model)
        
        # Check ATE is calculated and in reasonable range
        # With DataFrameBuilder data, exact values will vary
        assert isinstance(result["ate"], float)
        assert -10 <= result["ate"] <= 10  # Reasonable range for ATE
    
    def test_evaluate_with_source_df_parameter(self):
        """Test evaluation with optional source_df parameter."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["customer_id"],
            task_choice="causal",
            target_column="revenue_lift",
            feature_columns=["customer_value"],
            treatment_column="treatment"  # Add treatment_column
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_causal_data(
            n_samples=4,
            treatment_effect=100.0,  # Strong treatment effect for revenue
            n_features=1,
            add_entity_column=False
        )
        X = df[['treatment', 'feature_0']]
        X.columns = ['treatment', 'customer_value']  # Rename for test compatibility
        y = df['outcome']  # Use continuous outcome for revenue
        
        # Create source_df with same data plus metadata
        source_df = X.copy()
        source_df['id'] = list(range(1, len(X) + 1))
        source_df['revenue_lift'] = y
        source_df['metadata'] = ['A', 'B', 'C', 'D']
        
        # Act
        result = evaluator.evaluate(mock_model, X, y, source_df=source_df)
        
        # Assert - should work normally even with source_df provided
        assert "ate" in result
        assert "uplift_auc" in result
        assert result["uplift_auc"] == 0.0  # Mock model returns 0.0
        # Check ATE is calculated and reasonable
        # With DataFrameBuilder data, exact values will vary based on generated data
        assert isinstance(result["ate"], float)
        assert "ate_std" in result
    
    def test_evaluate_different_treatment_scenarios(self):
        """Test evaluation with different treatment assignment scenarios."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="click_through_rate",
            feature_columns=["segment"],
            treatment_column="treatment"
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        # Test different treatment scenarios using DataFrameBuilder
        treatment_scenarios = []
        
        # Balanced treatment assignment
        df1 = DataFrameBuilder.build_causal_data(
            n_samples=6,
            treatment_effect=0.2,
            n_features=1,
            add_entity_column=False
        )
        X1 = df1[['treatment', 'feature_0']]
        X1.columns = ['treatment', 'segment']
        # Convert feature to categorical segments
        X1 = X1.copy()
        X1['segment'] = ['A', 'A', 'B', 'B', 'C', 'C']
        y1 = df1['outcome']
        treatment_scenarios.append((X1, y1))
        
        # Unbalanced treatment assignment
        df2 = DataFrameBuilder.build_causal_data(
            n_samples=6,
            treatment_effect=0.3,
            confounding_strength=1.5,  # Higher confounding for unbalanced assignment
            n_features=1,
            add_entity_column=False
        )
        X2 = df2[['treatment', 'feature_0']]
        X2.columns = ['treatment', 'segment']
        X2 = X2.copy()
        X2['segment'] = ['A', 'B', 'C', 'A', 'B', 'C']
        y2 = df2['outcome']
        treatment_scenarios.append((X2, y2))
        
        for X, y in treatment_scenarios:
            mock_model = Mock()
            
            # Act
            result = evaluator.evaluate(mock_model, X, y)
            
            # Assert - should return consistent placeholder metric
            # Assert - should return expected metrics structure
        assert "ate" in result
        assert "uplift_auc" in result
    
    def test_evaluate_model_parameter_passed_correctly(self):
        """Test that model parameter is passed to evaluate method correctly."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="outcome",
            feature_columns=["treatment", "covariate"],
            treatment_column="treatment"
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        # Create a mock model with specific attributes to verify it's used
        mock_model = Mock()
        mock_model.model_type = "uplift_tree"
        mock_model.treatment_col = "treatment"
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_causal_data(
            n_samples=4,
            treatment_effect=1.0,
            n_features=1,
            add_entity_column=False
        )
        X = df[['treatment', 'feature_0']]
        X.columns = ['treatment', 'covariate']
        y = df['outcome_binary']
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - placeholder implementation doesn't use model, but should not error
        # Assert - should return expected metrics structure
        assert "ate" in result
        assert "uplift_auc" in result
        # Model object should be accessible (even if not used in placeholder)
        assert mock_model.model_type == "uplift_tree"


class TestCausalEvaluatorMetrics:
    """Test CausalEvaluator metric calculations."""
    
    def test_evaluate_uplift_auc_metric_range(self):
        """Test that uplift_auc metric is in valid range."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="conversion",
            feature_columns=["treatment", "feature"],
            treatment_column="treatment"
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_causal_data(
            n_samples=6,
            treatment_effect=1.0,
            n_features=1,
            add_entity_column=False
        )
        X = df[['treatment', 'feature_0']]
        X.columns = ['treatment', 'feature']
        y = df['outcome_binary']
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - uplift_auc should be in valid range [0, 1]
        assert 0 <= result["uplift_auc"] <= 1
        assert result["uplift_auc"] == 0.0  # Mock model without predict_uplift returns 0.0
    
    def test_evaluate_consistent_metric_output(self):
        """Test that metric output is consistent across different calls."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="revenue",
            feature_columns=["treatment", "segment", "value"],
            treatment_column="treatment"
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_causal_data(
            n_samples=8,
            treatment_effect=70.0,  # Revenue lift effect
            n_features=2,
            add_entity_column=False
        )
        X = df[['treatment', 'feature_0', 'feature_1']]
        X.columns = ['treatment', 'segment', 'value']
        # Replace segment with categorical values
        X = X.copy()
        X['segment'] = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
        y = df['outcome']
        
        # Act - multiple evaluations
        result1 = evaluator.evaluate(mock_model, X, y)
        result2 = evaluator.evaluate(mock_model, X, y)
        result3 = evaluator.evaluate(mock_model, X, y)
        
        # Assert - should return consistent results
        assert result1 == result2 == result3
        assert all(res["uplift_auc"] == 0.0 for res in [result1, result2, result3])  # Mock model returns 0.0


class TestCausalEvaluatorEdgeCases:
    """Test CausalEvaluator edge cases."""
    
    def test_evaluate_single_treatment_group(self):
        """Test evaluation when all samples belong to same treatment group."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="outcome",
            feature_columns=["treatment", "feature"],
            treatment_column="treatment"
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        # Use DataFrameBuilder but force all to control group
        df = DataFrameBuilder.build_causal_data(
            n_samples=4,
            treatment_effect=1.0,
            n_features=1,
            add_entity_column=False
        )
        X = df[['treatment', 'feature_0']]
        X.columns = ['treatment', 'feature']
        # Force all to control group (treatment = 0)
        X = X.copy()
        X['treatment'] = 0
        y = df['outcome_binary']
        
        # Act - placeholder implementation should still work
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        # Assert - should return expected metrics structure
        assert "ate" in result
        assert "uplift_auc" in result
    
    def test_evaluate_no_variation_in_outcome(self):
        """Test evaluation when outcome has no variation."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="constant_outcome",
            feature_columns=["treatment", "covariate"],
            treatment_column="treatment"
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        # Use DataFrameBuilder for test data
        df = DataFrameBuilder.build_causal_data(
            n_samples=6,
            treatment_effect=0.0,  # No effect since outcome is constant
            n_features=1,
            add_entity_column=False
        )
        X = df[['treatment', 'feature_0']]
        X.columns = ['treatment', 'covariate']
        # Force all outcomes to be the same (no variation)
        y = pd.Series([1, 1, 1, 1, 1, 1])
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - placeholder should handle this case
        # Assert - should return expected metrics structure
        assert "ate" in result
        assert "uplift_auc" in result
    
    def test_evaluate_single_sample(self):
        """Test evaluation with single data sample."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="outcome",
            feature_columns=["treatment"],
            treatment_column="treatment"
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        # Use DataFrameBuilder for single sample
        df = DataFrameBuilder.build_causal_data(
            n_samples=1,
            treatment_effect=1.0,
            n_features=0,  # No additional features
            add_entity_column=False
        )
        X = df[['treatment']]
        y = df['outcome_binary']
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert - should handle single sample case
        # Assert - should return expected metrics structure
        assert "ate" in result
        assert "uplift_auc" in result


class TestCausalEvaluatorIntegration:
    """Test CausalEvaluator integration scenarios."""
    
    def test_evaluate_marketing_campaign_scenario(self):
        """Test evaluation with realistic marketing campaign uplift scenario."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["customer_id"],
            task_choice="causal",
            target_column="purchase_probability",
            feature_columns=["email_campaign", "age", "previous_purchases", "engagement_score"],
            treatment_column="email_campaign"
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        # Realistic marketing campaign data
        mock_model = Mock()
        
        # Use DataFrameBuilder for realistic marketing data
        df = DataFrameBuilder.build_causal_data(
            n_samples=8,
            treatment_effect=0.3,  # 30% uplift from email campaign
            n_features=3,  # age, previous_purchases, engagement_score
            confounding_strength=0.7,  # Realistic confounding
            add_entity_column=False
        )
        X = df[['treatment', 'feature_0', 'feature_1', 'feature_2']]
        X.columns = ['email_campaign', 'age', 'previous_purchases', 'engagement_score']
        # Scale features to realistic ranges
        X = X.copy()
        X['age'] = (X['age'] - X['age'].min()) / (X['age'].max() - X['age'].min()) * 40 + 25  # 25-65
        X['previous_purchases'] = ((X['previous_purchases'] - X['previous_purchases'].min()) / 
                                  (X['previous_purchases'].max() - X['previous_purchases'].min()) * 8).astype(int)  # 0-8
        X['engagement_score'] = ((X['engagement_score'] - X['engagement_score'].min()) / 
                                (X['engagement_score'].max() - X['engagement_score'].min()))  # 0-1
        y = df['outcome']  # Use continuous outcome for purchase probability
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        assert "uplift_auc" in result
        assert result["uplift_auc"] == 0.0  # Mock model without predict_uplift returns 0.0
        
        # Should handle realistic causal inference data structure
        assert len(X) == len(y)
        assert 'email_campaign' in X.columns  # Treatment column present
    
    def test_evaluate_ab_test_scenario(self):
        """Test evaluation with A/B test scenario."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_choice="causal",
            target_column="conversion_rate",
            feature_columns=["variant", "user_segment", "device_type"],
            treatment_column="variant"
        )
        
        evaluator = CausalEvaluator(data_interface)
        
        mock_model = Mock()
        
        # A/B test data
        # Use DataFrameBuilder for A/B test data
        df = DataFrameBuilder.build_causal_data(
            n_samples=10,
            treatment_effect=0.08,  # 8% uplift from variant B
            n_features=2,  # Will map to user_segment and device_type
            confounding_strength=0.3,  # Some selection bias in A/B test
            add_entity_column=False
        )
        X = df[['treatment', 'feature_0', 'feature_1']]
        X.columns = ['variant', 'user_segment', 'device_type']
        # Map features to categorical values
        X = X.copy()
        X['user_segment'] = ['new', 'returning', 'new', 'returning', 'new', 
                            'returning', 'new', 'returning', 'new', 'returning']
        X['device_type'] = ['mobile', 'desktop', 'mobile', 'desktop', 'tablet',
                           'mobile', 'desktop', 'tablet', 'mobile', 'desktop']
        y = df['outcome']  # Conversion rates
        
        # Act
        result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        # Assert - should return expected metrics structure
        assert "ate" in result
        assert "uplift_auc" in result
        
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
            task_choice="causal",
            target_column="outcome",
            feature_columns=["treatment", "covariate"],
            treatment_column="treatment"
        )
        instance = EvaluatorRegistry.create("causal", data_interface)
        assert isinstance(instance, CausalEvaluator)