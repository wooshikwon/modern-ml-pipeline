"""
E2E Test: Deep Learning Classification
Tests complete deep learning pipeline with neural network models.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import mlflow
from types import SimpleNamespace

from src.settings import Settings
from src.settings.config import Config, Environment, MLflow as MLflowConfig, DataSource, FeatureStore, Output, OutputTarget
from src.settings.recipe import Recipe, Model, Data, Loader, Fetcher, DataInterface, Evaluation, ValidationConfig, HyperparametersTuning
from src.pipelines.train_pipeline import run_train_pipeline


class TestDeeplearningClassificationE2E:
    """End-to-end test for deep learning classification."""
    
    @pytest.fixture
    def temp_workspace(self):
        workspace = tempfile.mkdtemp()
        data_dir = os.path.join(workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate synthetic tabular data suitable for deep learning
        np.random.seed(42)
        n_samples = 500
        n_features = 20
        
        # Create high-dimensional feature space
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Create non-linear target
        weights = np.random.normal(0, 1, n_features)
        linear_combination = X @ weights
        non_linear = np.tanh(linear_combination) + 0.1 * np.sin(linear_combination)
        target = (non_linear > 0).astype(int)
        
        # Create feature names
        feature_names = [f'feature_{i:02d}' for i in range(n_features)]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = target
        
        train_path = os.path.join(data_dir, "train.csv")
        df.to_csv(train_path, index=False)
        
        yield {'workspace': workspace, 'data_dir': data_dir, 'train_path': train_path, 'features': feature_names}
        shutil.rmtree(workspace)
    
    @pytest.fixture
    def deeplearning_settings(self, temp_workspace):
        """Create settings for deep learning classification E2E test."""
        config = Config(
            environment=Environment(name="e2e_test"),
            mlflow=MLflowConfig(
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
                experiment_name="e2e_deeplearning_test"
            ),
            data_source=DataSource(
                name="file_storage",
                adapter_type="storage",
                config={"base_path": temp_workspace['data_dir']}
            ),
            feature_store=FeatureStore(provider="none"),
            output=Output(
                inference=OutputTarget(
                    name="e2e_deeplearning_output",
                    enabled=True,
                    adapter_type="storage",
                    config={"base_path": temp_workspace['workspace']}
                ),
                preprocessed=OutputTarget(
                    name="e2e_deeplearning_preprocessed",
                    enabled=False,
                    adapter_type="storage",
                    config={}
                )
            )
        )
        
        recipe = Recipe(
            name="e2e_deeplearning_recipe",
            task_choice="classification",
            model=Model(
                class_path="sklearn.neural_network.MLPClassifier",
                library="sklearn",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=False,
                    values={
                        "hidden_layer_sizes": (100, 50),
                        "max_iter": 200,
                        "random_state": 42
                    }
                ),
                computed={"run_name": "e2e_deeplearning_test_run"}
            ),
            data=Data(
                loader=Loader(source_uri=temp_workspace['train_path']),
                fetcher=Fetcher(type="pass_through"),
                data_interface=DataInterface(
                    target_column="target",
                    entity_columns=[],
                    feature_columns=temp_workspace['features']
                )
            ),
            evaluation=Evaluation(
                metrics=["accuracy", "precision", "recall", "f1"],
                validation=ValidationConfig(
                    method="train_test_split",
                    test_size=0.2,
                    random_state=42
                )
            )
        )
        
        return Settings(config=config, recipe=recipe)
    
    def test_complete_deeplearning_pipeline_e2e(self, deeplearning_settings, temp_workspace):
        """Test complete deep learning classification pipeline."""
        print("🚀 Starting E2E Deep Learning Classification Pipeline...")
        
        mlflow.set_experiment(deeplearning_settings.config.mlflow.experiment_name)
        train_result = run_train_pipeline(deeplearning_settings)
        
        assert hasattr(train_result, 'run_id'), "Training should return run_id"
        assert hasattr(train_result, 'model_uri'), "Training should return model_uri"
        
        model = mlflow.pyfunc.load_model(train_result.model_uri)
        
        # Test with sample data
        test_data = pd.DataFrame(
            np.random.normal(0, 1, (5, len(temp_workspace['features']))),
            columns=temp_workspace['features']
        )
        
        predictions = model.predict(test_data)
        assert len(predictions) == 5, "Should predict for all samples"
        assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary"
        
        print("✅ E2E Deep Learning Classification Pipeline completed successfully!")
        return {'train_result': train_result, 'predictions': predictions}