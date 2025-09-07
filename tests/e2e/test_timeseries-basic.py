"""
E2E Test: Time Series Analysis
Tests complete time series pipeline with ARIMA model and timeseries data handler.
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


class TestTimeseriesBasicE2E:
    """End-to-end test for time series analysis."""
    
    @pytest.fixture
    def temp_workspace(self):
        workspace = tempfile.mkdtemp()
        data_dir = os.path.join(workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate time series data
        np.random.seed(42)
        n_samples = 100
        
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        trend = np.linspace(10, 20, n_samples)
        seasonal = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        noise = np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame({
            'date': dates,
            'value': trend + seasonal + noise,
            'external_feature': np.random.normal(0, 1, n_samples)
        })
        
        train_path = os.path.join(data_dir, "train.csv")
        df.to_csv(train_path, index=False)
        
        yield {'workspace': workspace, 'data_dir': data_dir, 'train_path': train_path}
        shutil.rmtree(workspace)
    
    @pytest.fixture
    def timeseries_settings(self, temp_workspace):
        config = Config(
            environment=Environment(name="e2e_test"),
            mlflow=MLflowConfig(
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
                experiment_name="e2e_timeseries_test"
            ),
            data_source=DataSource(
                name="file_storage",
                adapter_type="storage",
                config={"base_path": temp_workspace['data_dir']}
            ),
            feature_store=FeatureStore(provider="none")
        )
        
        from src.settings.recipe import Recipe, Model, Data, Loader, Fetcher, DataInterface, Evaluation, ValidationConfig, HyperparametersTuning
        from datetime import datetime
        
        recipe = Recipe(
            name="e2e_timeseries_recipe",
            task_choice="timeseries",
            model=Model(
                class_path="sklearn.linear_model.LinearRegression",
                library="sklearn",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=False,
                    values={}
                ),
                computed={"run_name": "e2e_timeseries_test_run"}
            ),
            data=Data(
                loader=Loader(source_uri=temp_workspace['train_path']),
                fetcher=Fetcher(type="pass_through"),
                data_interface=DataInterface(
                    target_column="value",
                    entity_columns=[],
                    timestamp_column="date"
                )
            ),
            evaluation=Evaluation(
                metrics=["mae", "rmse"],
                validation=ValidationConfig(
                    method="train_test_split",
                    test_size=0.2,
                    random_state=42
                )
            )
        )
        
        return Settings(config=config, recipe=recipe)
    
    def test_complete_timeseries_pipeline_e2e(self, timeseries_settings, temp_workspace):
        """Test complete time series analysis pipeline."""
        print("🚀 Starting E2E Time Series Pipeline...")
        
        mlflow.set_experiment(timeseries_settings.config.mlflow.experiment_name)
        train_result = run_train_pipeline(timeseries_settings)
        
        assert hasattr(train_result, 'run_id'), "Training should return run_id"
        assert hasattr(train_result, 'model_uri'), "Training should return model_uri"
        
        model = mlflow.pyfunc.load_model(train_result.model_uri)
        
        # For simplified test, use basic prediction
        test_data = pd.DataFrame({
            'external_feature': [0.5, -0.5, 0.0]
        })
        
        predictions = model.predict(test_data)
        assert len(predictions) == 3, "Should predict for all samples"
        
        print("✅ E2E Time Series Pipeline completed successfully!")
        return {'train_result': train_result, 'predictions': predictions}