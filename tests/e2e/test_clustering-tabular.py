"""
E2E Test: Clustering with Tabular Data
Tests complete clustering pipeline with KMeans model and tabular data handler.
"""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import mlflow
from types import SimpleNamespace

from src.settings import Settings
from src.settings.config import Config, Environment, MLflow as MLflowConfig, DataSource, FeatureStore
from src.settings.recipe import Recipe
from src.pipelines.train_pipeline import run_train_pipeline


class TestClusteringTabularE2E:
    """End-to-end test for clustering with tabular data."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for clustering E2E test."""
        workspace = tempfile.mkdtemp()
        data_dir = os.path.join(workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate clustering dataset
        np.random.seed(42)
        n_samples = 300
        
        # Create 3 distinct clusters
        cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 100)
        cluster2 = np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 0.5]], 100)
        cluster3 = np.random.multivariate_normal([0, -2], [[0.5, 0], [0, 0.5]], 100)
        
        data = np.vstack([cluster1, cluster2, cluster3])
        
        df = pd.DataFrame({
            'x': data[:, 0],
            'y': data[:, 1],
            'feature_3': np.random.normal(0, 1, n_samples)
        })
        
        train_path = os.path.join(data_dir, "train.csv")
        df.to_csv(train_path, index=False)
        
        yield {
            'workspace': workspace,
            'data_dir': data_dir,
            'train_path': train_path,
            'df': df
        }
        
        shutil.rmtree(workspace)
    
    @pytest.fixture
    def clustering_settings(self, temp_workspace):
        """Create settings for clustering E2E test."""
        config = Config(
            environment=Environment(name="e2e_test"),
            mlflow=MLflowConfig(
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
                experiment_name="e2e_clustering_test"
            ),
            data_source=DataSource(
                name="file_storage",
                adapter_type="storage",
                config={"base_path": temp_workspace['data_dir']}
            ),
            feature_store=FeatureStore(provider="none")
        )
        
        recipe = Recipe(
            name="e2e_clustering_recipe",
            task_choice="clustering",
            data={
                "data_interface": {
                    "target_column": None,  # No target for clustering
                    "drop_columns": []
                },
                "feature_view": {
                    "name": "clustering_features",
                    "entities": [],
                    "features": ["x", "y", "feature_3"],
                    "source": {
                        "path": "train.csv",
                        "timestamp_column": None
                    }
                }
            },
            loader={
                "name": "csv_loader",
                "batch_size": 100,
                "shuffle": False  # Keep order for clustering
            },
            model={
                "class_path": "sklearn.cluster.KMeans",
                "init_args": {
                    "n_clusters": 3,
                    "random_state": 42
                },
                "compile_args": {},
                "fit_args": {}
            },
            fetcher={"type": "pass_through"},
            preprocessor={
                "steps": [
                    {
                        "name": "scaler",
                        "params": {
                            "method": "standard",
                            "features": ["x", "y", "feature_3"]
                        }
                    }
                ]
            },
            trainer={
                "validation_split": 0.0,  # No validation split for clustering
                "stratify": False,
                "random_state": 42
            }
        )
        
        return Settings(config=config, recipe=recipe)
    
    def test_complete_clustering_pipeline_e2e(self, clustering_settings, temp_workspace):
        """Test complete clustering pipeline."""
        print("🚀 Starting E2E Clustering Pipeline...")
        
        # Set MLflow experiment
        mlflow.set_experiment(clustering_settings.config.mlflow.experiment_name)
        
        # Run training pipeline
        train_result = run_train_pipeline(clustering_settings)
        
        # Validate training results
        assert hasattr(train_result, 'run_id'), "Training should return run_id"
        assert hasattr(train_result, 'model_uri'), "Training should return model_uri"
        
        print(f"✅ Clustering training completed. Run ID: {train_result.run_id}")
        
        # Load and test model
        model = mlflow.pyfunc.load_model(train_result.model_uri)
        
        # Test clustering predictions
        test_data = pd.DataFrame({
            'x': [2.0, -2.0, 0.0],
            'y': [2.0, 2.0, -2.0],
            'feature_3': [0.0, 0.0, 0.0]
        })
        
        predictions = model.predict(test_data)
        assert len(predictions) == 3, "Should predict clusters for all samples"
        assert all(isinstance(p, (int, np.integer)) for p in predictions), "Cluster predictions should be integers"
        
        print(f"✅ Clustering predictions: {predictions}")
        print("✅ E2E Clustering Pipeline completed successfully!")
        
        return {
            'train_result': train_result,
            'model': model,
            'predictions': predictions
        }