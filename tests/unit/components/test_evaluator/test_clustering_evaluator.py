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


class TestClusteringEvaluatorInitialization:
    """Test ClusteringEvaluator initialization."""
    
    def test_clustering_evaluator_inherits_base_evaluator(self):
        """Test that ClusteringEvaluator properly inherits from BaseEvaluator."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            target_column=None,
            feature_columns=["x", "y", "z"]
        )
        
        # Act
        evaluator = ClusteringEvaluator(data_interface)
        
        # Assert
        assert isinstance(evaluator, BaseEvaluator)
        assert isinstance(evaluator, ClusteringEvaluator)
    
    def test_init_stores_settings_and_task_type(self):
        """Test that initialization properly stores settings and task type."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            target_column=None,
            feature_columns=["feature1", "feature2", "feature3"]
        )
        
        # Act
        evaluator = ClusteringEvaluator(data_interface)
        
        # Assert
        assert evaluator.settings == data_interface
        assert evaluator.task_type == "clustering"
        assert evaluator.settings.feature_columns == ["feature1", "feature2", "feature3"]
        assert evaluator.settings.target_column is None  # Clustering has no target
    
    def test_init_minimal_configuration(self):
        """Test initialization with minimal required configuration."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering"
        )
        
        # Act
        evaluator = ClusteringEvaluator(data_interface)
        
        # Assert
        assert evaluator.task_type == "clustering"
        assert evaluator.settings.feature_columns is None
        assert evaluator.settings.target_column is None


class TestClusteringEvaluatorEvaluate:
    """Test ClusteringEvaluator evaluate method."""
    
    def test_evaluate_kmeans_clustering_success(self):
        """Test successful clustering evaluation with K-means."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["x", "y"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        # Mock model with labels attribute
        mock_model = Mock()
        mock_model.labels_ = np.array([0, 0, 1, 1, 2, 2])
        
        # Test data - 2D clustering scenario
        X = pd.DataFrame({
            'x': [1.0, 1.2, 5.0, 5.1, 9.0, 9.2],
            'y': [1.0, 1.1, 5.0, 5.2, 9.0, 9.1]
        })
        y = None  # Clustering doesn't use target labels
        
        # Act
        with patch('sklearn.metrics.silhouette_score', return_value=0.85) as mock_silhouette:
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
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["feature1", "feature2", "feature3"]
        )
        
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
            
            X = pd.DataFrame(np.random.rand(len(labels), 3))
            
            # Act
            with patch('sklearn.metrics.silhouette_score', return_value=expected_score):
                result = evaluator.evaluate(mock_model, X, None)
            
            # Assert
            assert result["silhouette_score"] == expected_score
    
    def test_evaluate_with_source_df_parameter(self):
        """Test evaluation with optional source_df parameter."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["x", "y"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.labels_ = np.array([0, 1, 0, 1])
        
        X = pd.DataFrame({
            'x': [1, 2, 1, 2],
            'y': [1, 2, 1, 2]
        })
        source_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'x': [1, 2, 1, 2],
            'y': [1, 2, 1, 2],
            'metadata': ['A', 'B', 'C', 'D']
        })
        
        # Act
        with patch('sklearn.metrics.silhouette_score', return_value=0.7):
            result = evaluator.evaluate(mock_model, X, None, source_df=source_df)
        
        # Assert - should work normally even with source_df provided
        assert result["silhouette_score"] == 0.7
        assert len(result) == 1
    
    def test_evaluate_model_labels_accessed_correctly(self):
        """Test that model.labels_ is accessed correctly."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["x", "y"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        cluster_labels = np.array([0, 1, 2, 0, 1, 2])
        mock_model.labels_ = cluster_labels
        
        X = pd.DataFrame(np.random.rand(6, 2))
        
        # Act
        with patch('sklearn.metrics.silhouette_score', return_value=0.8) as mock_silhouette:
            evaluator.evaluate(mock_model, X, None)
        
        # Assert - verify that the correct labels were passed to silhouette_score
        mock_silhouette.assert_called_once_with(X, cluster_labels)


class TestClusteringEvaluatorMetrics:
    """Test ClusteringEvaluator metric calculations."""
    
    def test_evaluate_perfect_clustering_scenario(self):
        """Test evaluation with well-separated clusters (high silhouette score)."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["x", "y"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        # Well-separated clusters
        mock_model.labels_ = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        # Well-separated data points
        X = pd.DataFrame({
            'x': [1, 1.1, 0.9, 5, 5.1, 4.9, 9, 9.1, 8.9],
            'y': [1, 1.1, 0.9, 5, 5.1, 4.9, 9, 9.1, 8.9]
        })
        
        # Act - Use real sklearn metrics for perfect case
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert - should get high silhouette score for well-separated clusters
        assert result["silhouette_score"] > 0.7
    
    def test_evaluate_poor_clustering_scenario(self):
        """Test evaluation with poorly separated clusters."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["feature"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        # Random cluster assignment - poor clustering
        mock_model.labels_ = np.array([0, 1, 0, 1, 0, 1])
        
        # Overlapping data points
        X = pd.DataFrame({
            'feature': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # Very close points
        })
        
        # Act
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert - silhouette score should be low for poor clustering
        assert -1 <= result["silhouette_score"] <= 1  # Valid range
        # For overlapping data with random assignment, score will be low
        assert result["silhouette_score"] < 0.5
    
    def test_evaluate_single_cluster_scenario(self):
        """Test evaluation when all points belong to single cluster."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["x", "y"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        # All points in single cluster
        mock_model.labels_ = np.array([0, 0, 0, 0, 0])
        
        X = pd.DataFrame(np.random.rand(5, 2))
        
        # Act & Assert - Single cluster should raise ValueError in silhouette_score
        with pytest.raises(ValueError):
            evaluator.evaluate(mock_model, X, None)
    
    def test_evaluate_with_outlier_detection(self):
        """Test evaluation with outlier labels (-1)."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["x", "y"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        # DBSCAN-style labels with outliers (-1)
        mock_model.labels_ = np.array([0, 0, 1, 1, -1, -1])
        
        X = pd.DataFrame(np.random.rand(6, 2))
        
        # Act
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert - should handle outlier labels appropriately
        assert "silhouette_score" in result
        assert -1 <= result["silhouette_score"] <= 1


class TestClusteringEvaluatorErrorHandling:
    """Test ClusteringEvaluator error handling."""
    
    def test_evaluate_with_missing_labels_attribute(self):
        """Test evaluation when model doesn't have labels_ attribute."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["feature"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        # Model without labels_ attribute
        mock_model = Mock()
        del mock_model.labels_  # Remove the attribute
        
        X = pd.DataFrame({'feature': [1, 2, 3]})
        
        # Act & Assert
        with pytest.raises(AttributeError):
            evaluator.evaluate(mock_model, X, None)
    
    def test_evaluate_with_mismatched_data_shapes(self):
        """Test evaluation with mismatched X and labels shapes."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["x", "y"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.labels_ = np.array([0, 1, 0])  # 3 labels
        
        X = pd.DataFrame({
            'x': [1, 2],  # Only 2 data points - mismatch!
            'y': [1, 2]
        })
        
        # Act & Assert - sklearn should handle this and raise appropriate error
        with pytest.raises(ValueError):
            evaluator.evaluate(mock_model, X, None)
    
    def test_evaluate_with_invalid_labels_type(self):
        """Test evaluation when labels_ is not numpy array."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["feature"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.labels_ = [0, 1, 0, 1]  # List instead of numpy array
        
        X = pd.DataFrame({'feature': [1, 2, 3, 4]})
        
        # Act - sklearn should handle conversion
        result = evaluator.evaluate(mock_model, X, None)
        
        # Assert - should work despite type difference
        assert "silhouette_score" in result
        assert isinstance(result["silhouette_score"], float)


class TestClusteringEvaluatorIntegration:
    """Test ClusteringEvaluator integration scenarios."""
    
    def test_evaluate_customer_segmentation_scenario(self):
        """Test evaluation with realistic customer segmentation scenario."""
        # Arrange
        data_interface = DataInterface(
            entity_columns=["customer_id"],
            task_type="clustering",
            feature_columns=["age", "income", "spending_score"]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        # Realistic customer segmentation
        mock_model = Mock()
        # 3 customer segments
        mock_model.labels_ = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        
        X = pd.DataFrame({
            'age': [25, 30, 45, 50, 35, 40, 28, 47, 38, 32],
            'income': [30000, 35000, 60000, 65000, 45000, 50000, 32000, 62000, 48000, 37000],
            'spending_score': [80, 85, 40, 35, 60, 55, 82, 38, 58, 80]
        })
        
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
        # Arrange
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=[f"feature_{i}" for i in range(10)]
        )
        
        evaluator = ClusteringEvaluator(data_interface)
        
        mock_model = Mock()
        mock_model.labels_ = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        
        # High-dimensional data (10 features)
        X = pd.DataFrame(np.random.rand(8, 10))
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
        
        # Verify can create instance through registry
        data_interface = DataInterface(
            entity_columns=["user_id"],
            task_type="clustering",
            feature_columns=["feature1", "feature2"]
        )
        instance = EvaluatorRegistry.create("clustering", data_interface)
        assert isinstance(instance, ClusteringEvaluator)