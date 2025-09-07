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
import uuid
from src.settings import Settings
from src.settings.config import Config, Environment, MLflow as MLflowConfig, DataSource, FeatureStore, Output, OutputTarget
from src.settings.recipe import Recipe, Model, Data, Loader, Fetcher, DataInterface, Evaluation, ValidationConfig, HyperparametersTuning
from src.pipelines.train_pipeline import run_train_pipeline


class TestCausalTabularE2E:
    """End-to-end test for causal analysis with tabular data."""
    
    @pytest.fixture

    
    def temp_workspace(self, isolated_mlflow):
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
    def causal_settings(self, temp_workspace, isolated_mlflow, unique_experiment_name):
        config = Config(
            environment=Environment(name="e2e_test"),
            mlflow=MLflowConfig(
                tracking_uri=isolated_mlflow,
                experiment_name=f"e2e_causal_test_{unique_experiment_name}"
            ),
            data_source=DataSource(
                name="file_storage",
                adapter_type="storage",
                config={"base_path": temp_workspace['data_dir']}
            ),
            feature_store=FeatureStore(provider="none"),
            output=Output(
                inference=OutputTarget(
                    name="e2e_test_output",
                    enabled=True,
                    adapter_type="storage",
                    config={"base_path": temp_workspace['workspace']}
                ),
                preprocessed=OutputTarget(
                    name="e2e_test_preprocessed",
                    enabled=False,
                    adapter_type="storage",
                    config={}
                )
            )
        )
        
        recipe = Recipe(
            name="e2e_causal_recipe",
            task_choice="causal",
            model=Model(
                class_path="sklearn.linear_model.LinearRegression",
                library="sklearn",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=False,
                    values={}
                ),
                computed={"run_name": "e2e_causal_test_run"}
            ),
            data=Data(
                loader=Loader(source_uri=temp_workspace['train_path']),
                fetcher=Fetcher(type="pass_through"),
                data_interface=DataInterface(
                    target_column="outcome",
                    treatment_column="treatment",  # causal task에서 필수
                    entity_columns=[],
                    feature_columns=None  # null이면 모든 컬럼 사용 (target, entity 제외)
                )
            ),
            evaluation=Evaluation(
                metrics=["mae", "rmse", "r2"],
                validation=ValidationConfig(
                    method="train_test_split",
                    test_size=0.2,
                    random_state=42
                )
            )
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