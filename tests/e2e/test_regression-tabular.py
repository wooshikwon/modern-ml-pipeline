"""
E2E Test: Regression with Tabular Data
Tests complete regression pipeline with RandomForestRegressor model and tabular data handler.
"""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import mlflow
import mlflow.pyfunc
from types import SimpleNamespace
from sklearn.metrics import mean_squared_error, r2_score

from src.settings import Settings
from src.settings.config import Config, Environment, MLflow as MLflowConfig, DataSource, FeatureStore
from src.settings.recipe import Recipe
from src.pipelines.train_pipeline import run_train_pipeline
from src.pipelines.inference_pipeline import run_inference_pipeline
from src.factory import Factory


class TestRegressionTabularE2E:
    """End-to-end test for regression with tabular data."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for E2E test."""
        workspace = tempfile.mkdtemp()
        data_dir = os.path.join(workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate realistic regression dataset (house prices)
        np.random.seed(42)
        n_samples = 500
        
        # Generate features with realistic patterns
        square_footage = np.random.normal(2000, 800, n_samples)
        square_footage = np.clip(square_footage, 500, 5000)  # Reasonable range
        
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.25, 0.35, 0.25, 0.05])
        bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                                   p=[0.15, 0.1, 0.25, 0.15, 0.2, 0.1, 0.05])
        age = np.random.randint(0, 50, n_samples)
        location_score = np.random.normal(7, 2, n_samples)
        location_score = np.clip(location_score, 1, 10)
        
        # Create realistic target (house price) with noise
        price_base = (
            square_footage * 100 +  # $100 per sq ft base
            bedrooms * 15000 +      # $15k per bedroom
            bathrooms * 8000 +      # $8k per bathroom
            (50 - age) * 2000 +     # Depreciation
            location_score * 25000  # Location premium
        )
        
        # Add realistic noise
        noise = np.random.normal(0, price_base * 0.1, n_samples)
        house_price = price_base + noise
        house_price = np.clip(house_price, 50000, 2000000)  # Reasonable price range
        
        df = pd.DataFrame({
            'square_footage': square_footage,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age_years': age,
            'location_score': location_score,
            'house_price': house_price
        })
        
        # Split into train and test
        train_df = df.iloc[:400]
        test_df = df.iloc[400:]
        
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        yield {
            'workspace': workspace,
            'train_path': train_path,
            'test_path': test_path,
            'data_dir': data_dir,
            'train_df': train_df,
            'test_df': test_df
        }
        
        # Cleanup
        shutil.rmtree(workspace)
    
    @pytest.fixture
    def regression_settings(self, temp_workspace):
        """Create settings for regression E2E test."""
        config = Config(
            environment=Environment(name="e2e_test"),
            mlflow=MLflowConfig(
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
                experiment_name="e2e_regression_test"
            ),
            data_source=DataSource(
                name="file_storage",
                adapter_type="storage",
                config={"base_path": temp_workspace['data_dir']}
            ),
            feature_store=FeatureStore(provider="none")
        )
        
        recipe = Recipe(
            name="e2e_regression_recipe",
            task_choice="regression",  # Using new task_choice field
            data={
                "data_interface": {
                    "target_column": "house_price",
                    "drop_columns": []
                },
                "feature_view": {
                    "name": "regression_features",
                    "entities": [],
                    "features": ["square_footage", "bedrooms", "bathrooms", "age_years", "location_score"],
                    "source": {
                        "path": "train.csv",
                        "timestamp_column": None
                    }
                }
            },
            loader={
                "name": "csv_loader",
                "batch_size": 100,
                "shuffle": True
            },
            model={
                "class_path": "sklearn.ensemble.RandomForestRegressor",
                "init_args": {
                    "n_estimators": 50,  # Smaller for faster testing
                    "random_state": 42,
                    "max_depth": 10
                },
                "compile_args": {},
                "fit_args": {}
            },
            fetcher={
                "type": "pass_through"
            },
            preprocessor={
                "steps": [
                    {
                        "name": "scaler",
                        "params": {
                            "method": "standard",
                            "features": ["square_footage", "age_years", "location_score"]
                        }
                    }
                ]
            },
            trainer={
                "validation_split": 0.2,
                "stratify": False,  # No stratification for regression
                "random_state": 42
            }
        )
        
        return Settings(config=config, recipe=recipe)
    
    def test_complete_regression_pipeline_e2e(self, regression_settings, temp_workspace):
        """Test complete regression pipeline from training to inference."""
        # Phase 1: Training Pipeline
        print("🚀 Starting E2E Regression Training Pipeline...")
        
        # Set MLflow experiment
        mlflow.set_experiment(regression_settings.config.mlflow.experiment_name)
        
        # Run training pipeline
        train_result = run_train_pipeline(regression_settings)
        
        # Validate training results
        assert hasattr(train_result, 'run_id'), "Training should return run_id"
        assert hasattr(train_result, 'model_uri'), "Training should return model_uri"
        assert train_result.run_id is not None, "Run ID should not be None"
        assert train_result.model_uri.startswith('runs:/'), "Model URI should be MLflow format"
        
        print(f"✅ Training completed. Run ID: {train_result.run_id}")
        
        # Verify MLflow run
        run = mlflow.get_run(train_result.run_id)
        assert run.info.status == 'FINISHED', "MLflow run should be finished"
        
        # Verify metrics were logged
        metrics = run.data.metrics
        required_metrics = ['row_count', 'column_count']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        print(f"✅ Metrics verified: {list(metrics.keys())}")
        
        # Phase 2: Model Loading and Validation
        print("🔍 Validating trained model...")
        
        # Load model from MLflow
        model = mlflow.pyfunc.load_model(train_result.model_uri)
        assert model is not None, "Model should be loadable"
        
        # Test prediction with sample data
        test_data = pd.DataFrame({
            'square_footage': [1500, 2500, 3000],
            'bedrooms': [2, 3, 4],
            'bathrooms': [1.5, 2.5, 3],
            'age_years': [5, 15, 25],
            'location_score': [8.5, 7.0, 9.2]
        })
        
        predictions = model.predict(test_data)
        assert len(predictions) == 3, "Should predict for all samples"
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions), "Predictions should be numeric"
        assert all(pred > 0 for pred in predictions), "House prices should be positive"
        
        print(f"✅ Model predictions: {predictions}")
        
        # Phase 3: Inference Pipeline
        print("🔮 Running inference pipeline...")
        
        # Create inference settings with test data
        inference_settings = Settings(
            config=regression_settings.config,
            recipe=regression_settings.recipe
        )
        
        # Update data source to point to test file
        inference_settings.recipe.data["feature_view"]["source"]["path"] = "test.csv"
        
        # Run inference pipeline
        inference_context = SimpleNamespace(
            model_uri=train_result.model_uri,
            output_path=os.path.join(temp_workspace['workspace'], 'predictions.csv')
        )
        
        inference_result = run_inference_pipeline(inference_settings, inference_context)
        
        # Validate inference results
        assert hasattr(inference_result, 'predictions'), "Should return predictions"
        assert len(inference_result.predictions) > 0, "Should have predictions"
        
        # Verify predictions file was created
        predictions_path = inference_context.output_path
        assert os.path.exists(predictions_path), "Predictions file should be created"
        
        predictions_df = pd.read_csv(predictions_path)
        assert len(predictions_df) == 100, "Should predict for all test samples"  # 500 - 400 = 100
        assert 'prediction' in predictions_df.columns, "Should have prediction column"
        
        print(f"✅ Inference completed. Predictions saved to: {predictions_path}")
        
        # Phase 4: Component Integration Validation
        print("🔧 Validating component integration...")
        
        # Test Factory component creation
        factory = Factory(regression_settings)
        
        # Test data handler creation
        data_handler = factory.create_datahandler()
        assert data_handler is not None, "DataHandler should be created"
        assert hasattr(data_handler, 'prepare_data'), "DataHandler should have prepare_data method"
        
        # Test preprocessor creation
        preprocessor = factory.create_preprocessor()
        assert preprocessor is not None, "Preprocessor should be created"
        
        # Test trainer creation  
        trainer = factory.create_trainer()
        assert trainer is not None, "Trainer should be created"
        
        # Test evaluator creation
        evaluator = factory.create_evaluator()
        assert evaluator is not None, "Evaluator should be created"
        
        print("✅ All components validated successfully")
        
        # Phase 5: Regression-Specific Metrics Validation
        print("📊 Validating regression-specific metrics...")
        
        # Calculate regression metrics
        test_df = temp_workspace['test_df']
        true_prices = test_df['house_price'].values
        predicted_prices = predictions_df['prediction'].values
        
        # Calculate metrics
        mse = mean_squared_error(true_prices, predicted_prices)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_prices, predicted_prices)
        
        # Validate metrics are reasonable
        mean_price = np.mean(true_prices)
        relative_rmse = rmse / mean_price
        
        assert rmse > 0, "RMSE should be positive"
        assert relative_rmse < 1.0, f"Relative RMSE should be < 100%, got {relative_rmse:.2%}"
        assert r2 > -1, f"R² should be > -1, got {r2:.3f}"  # At least better than predicting mean
        
        # Check prediction reasonableness
        min_pred, max_pred = predicted_prices.min(), predicted_prices.max()
        min_true, max_true = true_prices.min(), true_prices.max()
        
        assert min_pred > 0, "All predictions should be positive"
        assert min_pred < max_pred, "Predictions should have reasonable range"
        
        print(f"✅ E2E Regression Pipeline completed successfully!")
        print(f"   - Training samples: 400")
        print(f"   - Test samples: 100")
        print(f"   - Model: RandomForestRegressor")
        print(f"   - RMSE: ${rmse:,.0f}")
        print(f"   - Relative RMSE: {relative_rmse:.2%}")
        print(f"   - R² Score: {r2:.3f}")
        print(f"   - Price range: ${min_true:,.0f} - ${max_true:,.0f} (true), ${min_pred:,.0f} - ${max_pred:,.0f} (pred)")
        print(f"   - MLflow run: {train_result.run_id}")
        
        return {
            'train_result': train_result,
            'inference_result': inference_result,
            'model': model,
            'predictions_path': predictions_path,
            'metrics': {
                'rmse': rmse,
                'r2': r2,
                'relative_rmse': relative_rmse
            }
        }