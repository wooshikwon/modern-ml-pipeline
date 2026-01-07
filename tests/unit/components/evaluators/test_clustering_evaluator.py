"""
Clustering Evaluator Unit Tests - No Mock Hell Approach
Real clustering metrics calculation
Following comprehensive testing strategy document principles
"""

import numpy as np
from sklearn.cluster import Birch, KMeans
from sklearn.mixture import GaussianMixture

from src.components.evaluator.modules.clustering_evaluator import ClusteringEvaluator
from src.components.evaluator.base import BaseEvaluator


class TestClusteringEvaluator:
    """Test ClusteringEvaluator with real clustering models and metrics."""

    def test_clustering_evaluator_initialization(self, settings_builder):
        """Test ClusteringEvaluator initialization."""
        # Given: Valid settings for clustering
        settings = settings_builder.with_task("clustering").build()

        # When: Creating ClusteringEvaluator
        evaluator = ClusteringEvaluator(settings)

        # Then: Evaluator is properly initialized
        assert isinstance(evaluator, ClusteringEvaluator)
        assert isinstance(evaluator, BaseEvaluator)
        assert evaluator.task_choice == "clustering"

    def test_evaluate_with_kmeans_clustering(self, settings_builder):
        """Test evaluation with KMeans clustering."""
        # Given: Data for clustering
        np.random.seed(42)
        # Create 3 clear clusters
        cluster1 = np.random.randn(30, 2) + [0, 0]
        cluster2 = np.random.randn(30, 2) + [5, 5]
        cluster3 = np.random.randn(30, 2) + [10, 0]
        X = np.vstack([cluster1, cluster2, cluster3])

        # Train KMeans model
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)

        settings = settings_builder.with_task("clustering").build()
        evaluator = ClusteringEvaluator(settings)

        # When: Evaluating clustering model
        # Note: Clustering typically doesn't use y (unsupervised)
        metrics = evaluator.evaluate(model, X, y=None)

        # Then: Clustering metrics are calculated
        assert isinstance(metrics, dict)

        # Common clustering metrics
        possible_metrics = [
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
            "inertia",
        ]
        assert any(m in metrics for m in possible_metrics)

        # Silhouette score should be good for well-separated clusters
        if "silhouette_score" in metrics:
            assert -1 <= metrics["silhouette_score"] <= 1
            assert metrics["silhouette_score"] > 0.3  # Reasonably good clustering

    def test_evaluate_with_gaussian_mixture(self, settings_builder):
        """Test evaluation with Gaussian Mixture Model clustering."""
        # Given: Data suitable for GMM
        np.random.seed(42)
        # Create two Gaussian distributions
        cluster1 = np.random.randn(50, 2) * 0.5 + [0, 0]
        cluster2 = np.random.randn(50, 2) * 0.5 + [3, 3]
        X = np.vstack([cluster1, cluster2])

        # Train GaussianMixture model
        model = GaussianMixture(n_components=2, random_state=42)
        model.fit(X)

        settings = settings_builder.with_task("clustering").build()
        evaluator = ClusteringEvaluator(settings)

        # When: Evaluating GMM model
        metrics = evaluator.evaluate(model, X, y=None)

        # Then: Metrics are calculated correctly
        assert isinstance(metrics, dict)
        assert "n_clusters" in metrics
        assert metrics["n_clusters"] == 2
        assert "silhouette_score" in metrics
        assert "bic" in metrics  # GMM-specific
        assert "aic" in metrics  # GMM-specific
        assert metrics["silhouette_score"] > 0.3  # Should have good separation

    def test_evaluate_with_birch_clustering(self, settings_builder):
        """Test evaluation with Birch clustering."""
        # Given: Dataset for Birch clustering
        np.random.seed(42)
        # Create three small clusters
        cluster1 = np.random.randn(20, 3) * 0.3 + [0, 0, 0]
        cluster2 = np.random.randn(20, 3) * 0.3 + [2, 2, 0]
        cluster3 = np.random.randn(20, 3) * 0.3 + [1, 1, 3]
        X = np.vstack([cluster1, cluster2, cluster3])

        # Train Birch model
        model = Birch(n_clusters=3, threshold=0.5)
        model.fit(X)

        settings = settings_builder.with_task("clustering").build()
        evaluator = ClusteringEvaluator(settings)

        # When: Evaluating Birch model
        metrics = evaluator.evaluate(model, X, y=None)

        # Then: Metrics are valid
        assert isinstance(metrics, dict)
        assert "n_clusters" in metrics
        assert "silhouette_score" in metrics
        assert metrics["n_clusters"] <= 3  # Birch might merge clusters
        assert metrics["silhouette_score"] > 0  # Should have some separation
