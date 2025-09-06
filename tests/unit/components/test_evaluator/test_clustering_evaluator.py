"""
Unit tests for ClusteringEvaluator.
Tests clustering evaluation functionality with sklearn metrics.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.components.evaluator.modules.clustering_evaluator import ClusteringEvaluator
from src.interface.base_evaluator import BaseEvaluator
from src.settings.recipe import DataInterface
from tests.helpers.builders import RecipeBuilder, DataFrameBuilder


class TestClusteringEvaluatorInitialization:
    """Test ClusteringEvaluator initialization."""
    
    def test_clustering_evaluator_inherits_base_evaluator(self):
        """Test that ClusteringEvaluator properly inherits from BaseEvaluator."""
        # Arrange - Use RecipeBuilder for consistency
        recipe = RecipeBuilder.build(task_type="clustering")
        data_interface = recipe.data.data_interface
        
        # Act
        evaluator = ClusteringEvaluator(data_interface)
        
        # Assert
        assert isinstance(evaluator, BaseEvaluator)
        assert isinstance(evaluator, ClusteringEvaluator)
    
    def test_init_stores_settings_and_task_type(self):
        """Test that initialization properly stores settings and task type."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["feature1", "feature2", "feature3"]}
        )
        data_interface = recipe.data.data_interface
        
        # Act
        evaluator = ClusteringEvaluator(data_interface)
        
        # Assert
        assert evaluator.settings == data_interface
        assert evaluator.task_type == "clustering"
        assert evaluator.settings.feature_columns == ["feature1", "feature2", "feature3"]
        # target_column has dummy value for clustering but it's ignored by the evaluator
    
    def test_init_minimal_configuration(self):
        """Test initialization with minimal required configuration."""
        # Arrange - Use RecipeBuilder with minimal config
        recipe = RecipeBuilder.build(task_type="clustering")
        data_interface = recipe.data.data_interface
        
        # Act
        evaluator = ClusteringEvaluator(data_interface)
        
        # Assert
        assert evaluator.task_type == "clustering"
        assert evaluator.settings.feature_columns is None
        # target_column has dummy value for clustering but it's ignored


class TestClusteringEvaluatorEvaluate:
    """Test ClusteringEvaluator evaluate method."""
    
    def test_evaluate_kmeans_clustering_success(self):
        """Test successful clustering evaluation with K-means."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["x", "y"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        # Mock model with labels attribute
        mock_model = Mock()
        mock_model.labels_ = np.array([0, 0, 1, 1, 2, 2])
        
        # Test data - Use DataFrameBuilder for well-separated clusters
        X = DataFrameBuilder.build_clustering_data(
            n_samples=6,
            n_clusters=3,
            n_features=2,
            separation='well_separated',
            add_entity_column=False
        )
        # Remove the cluster_label column if present (it's for validation, not input)
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        y = None  # Clustering doesn't use target labels
        
        # Act - patch the actual import location in the module
        with patch('src.components.evaluator.modules.clustering_evaluator.silhouette_score', return_value=0.85) as mock_silhouette:
            result = evaluator.evaluate(mock_model, X, y)
        
        # Assert
        expected_metrics = {
            "silhouette_score": 0.85
        }
        assert result == expected_metrics
        
        # Verify sklearn function was called correctly
        mock_silhouette.assert_called_once_with(X, mock_model.labels_)
    
    def test_evaluate_with_different_cluster_sizes(self):
        """Test evaluation with different numbers of clusters."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["feature1", "feature2", "feature3"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        # Test different cluster scenarios
        cluster_scenarios = [
            (np.array([0, 1, 0, 1]), 0.6),  # 2 clusters
            (np.array([0, 1, 2, 0, 1, 2]), 0.75),  # 3 clusters
            (np.array([0, 1, 2, 3, 0, 1, 2, 3]), 0.82),  # 4 clusters
        ]
        
        for labels, expected_score in cluster_scenarios:
            mock_model = Mock()
            mock_model.labels_ = labels
            
            X = DataFrameBuilder.build_clustering_data(
                n_samples=len(labels),
                n_clusters=len(np.unique(labels)),
                n_features=3,
                separation='overlapping',
                add_entity_column=False
            )
            if 'cluster_label' in X.columns:
                X = X.drop(columns=['cluster_label'])
            
            # Act
            with patch('src.components.evaluator.modules.clustering_evaluator.silhouette_score', return_value=expected_score):
                result = evaluator.evaluate(mock_model, X, None)
            
            # Assert
            assert result["silhouette_score"] == expected_score
    
    def test_evaluate_with_source_df_parameter(self):
        """Test evaluation with optional source_df parameter."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["x", "y"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.labels_ = np.array([0, 1, 0, 1])
        
        # Use DataFrameBuilder for test data
        X = DataFrameBuilder.build_clustering_data(
            n_samples=4,
            n_clusters=2,
            n_features=2,
            separation='overlapping',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        
        # Create source_df with same data plus metadata
        source_df = X.copy()
        source_df['id'] = list(range(1, len(X) + 1))
        source_df['metadata'] = ['A', 'B', 'C', 'D']
        
        # Act
        with patch('src.components.evaluator.modules.clustering_evaluator.silhouette_score', return_value=0.7):
            result = evaluator.evaluate(mock_model, X, None, source_df=source_df)
        
        # Assert - should work normally even with source_df provided
        assert result["silhouette_score"] == 0.7
        assert len(result) == 1
    
    def test_evaluate_model_labels_accessed_correctly(self):
        """Test that model.labels_ is accessed correctly."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["x", "y"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        cluster_labels = np.array([0, 1, 2, 0, 1, 2])
        mock_model.labels_ = cluster_labels
        
        # Use DataFrameBuilder for random test data
        X = DataFrameBuilder.build_clustering_data(
            n_samples=6,
            n_clusters=3,
            n_features=2,
            separation='random',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        
        # Act
        with patch('src.components.evaluator.modules.clustering_evaluator.silhouette_score', return_value=0.8) as mock_silhouette:
            evaluator.evaluate(mock_model, X, None)
        
        # Assert - verify that the correct labels were passed to silhouette_score
        mock_silhouette.assert_called_once_with(X, cluster_labels)


class TestClusteringEvaluatorMetrics:
    """Test ClusteringEvaluator metric calculations."""
    
    def test_evaluate_perfect_clustering_scenario(self):
        """Test evaluation with well-separated clusters (high silhouette score)."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["x", "y"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        # Well-separated clusters with more samples per cluster
        mock_model.labels_ = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        # Use DataFrameBuilder for well-separated clusters
        X = DataFrameBuilder.build_clustering_data(
            n_samples=12,
            n_clusters=3,
            n_features=2,
            separation='well_separated',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        
        # Act - Use real sklearn metrics for perfect case
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert - should get high silhouette score for well-separated clusters
        assert result["silhouette_score"] > 0.7
    
    def test_evaluate_poor_clustering_scenario(self):
        """Test evaluation with poorly separated clusters."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["feature"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        # Random cluster assignment - poor clustering with more samples
        mock_model.labels_ = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        # Use DataFrameBuilder for overlapping/poor clustering scenario
        X = DataFrameBuilder.build_clustering_data(
            n_samples=8,
            n_clusters=2,
            n_features=1,
            separation='overlapping',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        
        # Act
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert - silhouette score should be low for poor clustering
        assert -1 <= result["silhouette_score"] <= 1  # Valid range
        # For overlapping data with random assignment, score will be low
        assert result["silhouette_score"] < 0.5
    
    def test_evaluate_single_cluster_scenario(self):
        """Test evaluation when all points belong to single cluster."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["x", "y"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        # All points in single cluster
        mock_model.labels_ = np.array([0, 0, 0, 0, 0])
        
        # Use DataFrameBuilder for random data
        X = DataFrameBuilder.build_clustering_data(
            n_samples=5,
            n_clusters=1,  # Will generate random data effectively
            n_features=2,
            separation='random',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        
        # Act & Assert - Single cluster should raise ValueError in silhouette_score
        with pytest.raises(ValueError):
            evaluator.evaluate(mock_model, X, None)
    
    def test_evaluate_with_outlier_detection(self):
        """Test evaluation with outlier labels (-1)."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["x", "y"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        # DBSCAN-style labels with outliers (-1) and more samples
        mock_model.labels_ = np.array([0, 0, 0, 1, 1, 1, -1, -1])
        
        # Use DataFrameBuilder for test data with outliers
        X = DataFrameBuilder.build_clustering_data(
            n_samples=8,
            n_clusters=2,
            n_features=2,
            separation='overlapping',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        
        # Act
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert - should handle outlier labels appropriately
        assert "silhouette_score" in result
        assert -1 <= result["silhouette_score"] <= 1


class TestClusteringEvaluatorErrorHandling:
    """Test ClusteringEvaluator error handling."""
    
    def test_evaluate_with_missing_labels_attribute(self):
        """Test evaluation when model doesn't have labels_ attribute."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["feature"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        # Model without labels_ attribute - use spec to limit attributes
        mock_model = Mock(spec=[])  # Empty spec means no attributes
        
        # Use DataFrameBuilder for test data
        X = DataFrameBuilder.build_clustering_data(
            n_samples=3,
            n_clusters=2,
            n_features=1,
            separation='random',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        
        # Act & Assert
        with pytest.raises(AttributeError):
            evaluator.evaluate(mock_model, X, None)
    
    def test_evaluate_with_mismatched_data_shapes(self):
        """Test evaluation with mismatched X and labels shapes."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["x", "y"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.labels_ = np.array([0, 1, 0])  # 3 labels
        
        # Use DataFrameBuilder for test data with only 2 points
        X = DataFrameBuilder.build_clustering_data(
            n_samples=2,  # Only 2 data points - mismatch!
            n_clusters=2,
            n_features=2,
            separation='random',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        
        # Act & Assert - sklearn should handle this and raise appropriate error
        with pytest.raises(ValueError):
            evaluator.evaluate(mock_model, X, None)
    
    def test_evaluate_with_invalid_labels_type(self):
        """Test evaluation when labels_ is not numpy array."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["feature"]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.labels_ = [0, 1, 0, 1]  # List instead of numpy array
        
        # Use DataFrameBuilder for test data
        X = DataFrameBuilder.build_clustering_data(
            n_samples=4,
            n_clusters=2,
            n_features=1,
            separation='random',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        
        # Act - sklearn should handle conversion
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert - should work despite type difference
        assert "silhouette_score" in result
        assert isinstance(result["silhouette_score"], float)


class TestClusteringEvaluatorIntegration:
    """Test ClusteringEvaluator integration scenarios."""
    
    def test_evaluate_customer_segmentation_scenario(self):
        """Test evaluation with realistic customer segmentation scenario."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{
                'data.data_interface.entity_columns': ["customer_id"],
                'data.data_interface.feature_columns': ["age", "income", "spending_score"]
            }
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        # Realistic customer segmentation
        mock_model = Mock()
        # 3 customer segments with more balanced samples
        mock_model.labels_ = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        # Use DataFrameBuilder for realistic customer segmentation data
        X = DataFrameBuilder.build_clustering_data(
            n_samples=12,
            n_clusters=3,
            n_features=3,  # age, income, spending_score
            separation='well_separated',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        # Rename columns for realistic customer data
        X.columns = ['age', 'income', 'spending_score']
        
        # Act
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert
        assert "silhouette_score" in result
        
        # Reasonable silhouette score for customer segmentation
        assert -1 <= result["silhouette_score"] <= 1
        # For well-defined customer segments, should have positive score
        assert result["silhouette_score"] > 0
    
    def test_evaluate_with_high_dimensional_data(self):
        """Test evaluation with high-dimensional feature space."""
        # Arrange - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': [f"feature_{i}" for i in range(10)]}
        )
        data_interface = recipe.data.data_interface
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        # More samples to ensure proper clustering evaluation
        mock_model.labels_ = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
        
        # High-dimensional data (10 features) with more samples
        X = DataFrameBuilder.build_clustering_data(
            n_samples=12,
            n_clusters=3,
            n_features=10,
            separation='overlapping',
            add_entity_column=False
        )
        if 'cluster_label' in X.columns:
            X = X.drop(columns=['cluster_label'])
        X.columns = [f"feature_{i}" for i in range(10)]
        
        # Act
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert
        assert "silhouette_score" in result
        assert isinstance(result["silhouette_score"], float)
        assert -1 <= result["silhouette_score"] <= 1


class TestClusteringEvaluatorSelfRegistration:
    """Test ClusteringEvaluator self-registration mechanism."""
    
    def test_clustering_evaluator_self_registration(self):
        """Test that ClusteringEvaluator registers itself in EvaluatorRegistry."""
        # Act - Import triggers self-registration
        from src.components.evaluator.modules import clustering_evaluator
        from src.components.evaluator.registry import EvaluatorRegistry
        
        # Assert
        assert "clustering" in EvaluatorRegistry.evaluators
        assert EvaluatorRegistry.evaluators["clustering"] == ClusteringEvaluator
        
        # Verify can create instance through registry - Use RecipeBuilder
        recipe = RecipeBuilder.build(
            task_type="clustering",
            **{'data.data_interface.feature_columns': ["feature1", "feature2"]}
        )
        data_interface = recipe.data.data_interface
        instance = EvaluatorRegistry.create("clustering", data_interface)
        assert isinstance(instance, ClusteringEvaluator)