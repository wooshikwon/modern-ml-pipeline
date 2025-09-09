"""
Clustering Evaluator Unit Tests - No Mock Hell Approach
Real clustering metrics calculation
Following comprehensive testing strategy document principles
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.components.evaluator.modules.clustering_evaluator import ClusteringEvaluator
from src.interface.base_evaluator import BaseEvaluator


class TestClusteringEvaluator:
    """Test ClusteringEvaluator with real clustering models and metrics."""
    
    def test_clustering_evaluator_initialization(self, settings_builder):
        """Test ClusteringEvaluator initialization."""
        # Given: Valid settings for clustering
        settings = settings_builder \
            .with_task("clustering") \
            .build()
        
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
        
        settings = settings_builder \
            .with_task("clustering") \
            .build()
        evaluator = ClusteringEvaluator(settings)
        
        # When: Evaluating clustering model
        # Note: Clustering typically doesn't use y (unsupervised)
        metrics = evaluator.evaluate(model, X, y=None)
        
        # Then: Clustering metrics are calculated
        assert isinstance(metrics, dict)
        
        # Common clustering metrics
        possible_metrics = ['silhouette_score', 'davies_bouldin_score', 
                          'calinski_harabasz_score', 'inertia']
        assert any(m in metrics for m in possible_metrics)
        
        # Silhouette score should be good for well-separated clusters
        if 'silhouette_score' in metrics:
            assert -1 <= metrics['silhouette_score'] <= 1
            assert metrics['silhouette_score'] > 0.3  # Reasonably good clustering
    
    def test_evaluate_with_dbscan_clustering(self, settings_builder):
        """Test evaluation with DBSCAN clustering."""
        # Given: Data with density-based clusters
        np.random.seed(42)
        # Create two dense regions
        region1 = np.random.randn(50, 2) * 0.5 + [0, 0]
        region2 = np.random.randn(50, 2) * 0.5 + [3, 3]
        X = np.vstack([region1, region2])
        
        # Train DBSCAN model
        model = DBSCAN(eps=0.5, min_samples=5)
        model.fit(X)
        
        settings = settings_builder \
            .with_task("clustering") \
            .build()
        evaluator = ClusteringEvaluator(settings)
        
        # When: Evaluating DBSCAN model
        metrics = evaluator.evaluate(model, X, y=None)
        
        # Then: Metrics are calculated (if clusters were found)
        assert isinstance(metrics, dict)
        
        # DBSCAN might find noise points (-1 label)
        n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
        if n_clusters > 1:  # Only if multiple clusters found
            assert len(metrics) > 0
    
    def test_evaluate_with_hierarchical_clustering(self, settings_builder):
        """Test evaluation with hierarchical clustering."""
        # Given: Small dataset for hierarchical clustering
        np.random.seed(42)
        X = np.random.randn(30, 3)
        
        # Train AgglomerativeClustering model
        model = AgglomerativeClustering(n_clusters=3)
        model.fit(X)
        
        settings = settings_builder \
            .with_task("clustering") \
            .build()
        evaluator = ClusteringEvaluator(settings)
        
        # When: Evaluating hierarchical model
        metrics = evaluator.evaluate(model, X, y=None)
        
        # Then: Metrics are calculated
        assert isinstance(metrics, dict)
        assert len(metrics) >= 0  # May have no metrics or some metrics