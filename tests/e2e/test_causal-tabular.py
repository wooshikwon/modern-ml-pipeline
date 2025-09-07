"""
E2E Test: Causal Analysis with Tabular Data
Tests complete causal pipeline with LinearRegression model and tabular data handler.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import mlflow
from src.settings import Settings
from src.settings.config import Config, Environment, MLflow as MLflowConfig, DataSource, FeatureStore
from src.settings.recipe import Recipe
from src.pipelines.train_pipeline import run_train_pipeline


class TestCausalTabularE2E:
    """End-to-end test for causal analysis with tabular data."""
    
    @pytest.fixture
    def temp_workspace(self):
        workspace = tempfile.mkdtemp()
        data_dir = os.path.join(workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate causal dataset
        np.random.seed(42)
        n_samples = 200
        
        df = pd.DataFrame({
            'treatment': np.random.binomial(1, 0.5, n_samples),
            'confounder_1': np.random.normal(0, 1, n_samples),
            'confounder_2': np.random.normal(0, 1, n_samples),
            'outcome': np.random.normal(0, 1, n_samples)
        })
        
        train_path = os.path.join(data_dir, "train.csv")
        df.to_csv(train_path, index=False)
        
        yield {'workspace': workspace, 'data_dir': data_dir, 'train_path': train_path}
        shutil.rmtree(workspace)
    
    @pytest.fixture
    def causal_settings(self, temp_workspace):
        config = Config(
            environment=Environment(name="e2e_test"),
            mlflow=MLflowConfig(
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
                experiment_name="e2e_causal_test"
            ),
            data_source=DataSource(
                name="file_storage",
                adapter_type="storage",
                config={"base_path": temp_workspace['data_dir']}
            ),
            feature_store=FeatureStore(provider="none")
        )
        
        recipe = Recipe(
            name="e2e_causal_recipe",
            task_choice="causal",
            data={
                "data_interface": {
                    "target_column": "outcome",
                    "drop_columns": []
                },
                "feature_view": {
                    "name": "causal_features",
                    "entities": [],
                    "features": ["treatment", "confounder_1", "confounder_2"],
                    "source": {"path": "train.csv", "timestamp_column": None}
                }
            },
            loader={"name": "csv_loader", "batch_size": 50, "shuffle": True},
            model={
                "class_path": "sklearn.linear_model.LinearRegression",
                "init_args": {},
                "compile_args": {},
                "fit_args": {}
            },
            fetcher={"type": "pass_through"},
            preprocessor={"steps": []},
            trainer={"validation_split": 0.2, "stratify": False, "random_state": 42}
        )
        
        return Settings(config=config, recipe=recipe)
    
    def test_complete_causal_pipeline_e2e(self, causal_settings, temp_workspace):
        """Test complete causal analysis pipeline."""
        print("🚀 Starting E2E Causal Pipeline...")
        
        mlflow.set_experiment(causal_settings.config.mlflow.experiment_name)
        train_result = run_train_pipeline(causal_settings)
        
        assert hasattr(train_result, 'run_id'), "Training should return run_id"
        assert hasattr(train_result, 'model_uri'), "Training should return model_uri"
        
        model = mlflow.pyfunc.load_model(train_result.model_uri)
        
        test_data = pd.DataFrame({
            'treatment': [0, 1, 0],
            'confounder_1': [0.5, -0.5, 0.0],
            'confounder_2': [1.0, 0.0, -1.0]
        })
        
        predictions = model.predict(test_data)
        assert len(predictions) == 3, "Should predict for all samples"
        
        print("✅ E2E Causal Pipeline completed successfully!")
        return {'train_result': train_result, 'predictions': predictions}