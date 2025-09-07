"""
E2E Test: Classification with Tabular Data
Tests complete classification pipeline with LogisticRegression model and tabular data handler.
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

from src.settings import Settings
from src.settings.config import Config, Environment, MLflow as MLflowConfig, DataSource, FeatureStore
from src.settings.recipe import Recipe
from src.pipelines.train_pipeline import run_train_pipeline
from src.pipelines.inference_pipeline import run_inference_pipeline
from src.factory import Factory


class TestClassificationTabularE2E:
    """End-to-end test for classification with tabular data."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for E2E test."""
        workspace = tempfile.mkdtemp()
        data_dir = os.path.join(workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate realistic classification dataset
        np.random.seed(42)
        n_samples = 500
        
        # Generate features with realistic patterns
        age = np.random.randint(18, 80, n_samples)
        income = np.random.normal(50000, 20000, n_samples)
        credit_score = np.random.randint(300, 850, n_samples)
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, 
                                   p=[0.4, 0.3, 0.2, 0.1])
        
        # Create target with realistic correlation
        target_prob = (age / 100) + (income / 100000) + (credit_score / 1000) - 1
        target_prob = 1 / (1 + np.exp(-target_prob))  # Sigmoid
        target = np.random.binomial(1, target_prob, n_samples)
        
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'credit_score': credit_score,
            'education': education,
            'approved': target
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
            'data_dir': data_dir
        }
        
        # Cleanup
        shutil.rmtree(workspace)
    
    @pytest.fixture
    def classification_settings(self, temp_workspace):
        """Create settings for classification E2E test."""
        config = Config(
            environment=Environment(name="e2e_test"),
            mlflow=MLflowConfig(
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
                experiment_name="e2e_classification_test"
            ),
            data_source=DataSource(
                name="file_storage",
                adapter_type="storage",
                config={"base_path": temp_workspace['data_dir']}
            ),
            feature_store=FeatureStore(provider="none")
        )
        
        recipe = Recipe(
            name="e2e_classification_recipe",
            task_choice="classification",  # Using new task_choice field
            data={
                "data_interface": {
                    "target_column": "approved",
                    "drop_columns": []
                },
                "feature_view": {
                    "name": "classification_features",
                    "entities": [],
                    "features": ["age", "income", "credit_score", "education"],
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
                "class_path": "sklearn.linear_model.LogisticRegression",
                "init_args": {
                    "random_state": 42,
                    "max_iter": 1000
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
                        "name": "encoder",
                        "params": {
                            "categorical_features": ["education"],
                            "encoding_type": "onehot"
                        }
                    },
                    {
                        "name": "scaler", 
                        "params": {
                            "method": "standard",
                            "features": ["age", "income", "credit_score"]
                        }
                    }
                ]
            },
            trainer={
                "validation_split": 0.2,
                "stratify": True,
                "random_state": 42
            }
        )
        
        return Settings(config=config, recipe=recipe)
    
    def test_complete_classification_pipeline_e2e(self, classification_settings, temp_workspace):
        """Test complete classification pipeline from training to inference."""
        # Phase 1: Training Pipeline
        print("🚀 Starting E2E Classification Training Pipeline...")
        
        # Set MLflow experiment
        mlflow.set_experiment(classification_settings.config.mlflow.experiment_name)
        
        # Run training pipeline
        train_result = run_train_pipeline(classification_settings)
        
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
            'age': [25, 45, 35],
            'income': [40000, 80000, 60000],
            'credit_score': [650, 750, 700],
            'education': ['Bachelor', 'Master', 'High School']
        })
        
        predictions = model.predict(test_data)
        assert len(predictions) == 3, "Should predict for all samples"
        assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary"
        
        print(f"✅ Model predictions: {predictions}")
        
        # Phase 3: Inference Pipeline
        print("🔮 Running inference pipeline...")
        
        # Create inference settings with test data
        inference_settings = Settings(
            config=classification_settings.config,
            recipe=classification_settings.recipe
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
        factory = Factory(classification_settings)
        
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
        
        # Phase 5: End-to-End Metrics Validation
        print("📊 Validating end-to-end metrics...")
        
        # Check that the complete pipeline produced reasonable results
        final_predictions = predictions_df['prediction'].values
        unique_predictions = set(final_predictions)
        
        # Should have both classes predicted (or at least valid binary predictions)
        assert unique_predictions.issubset({0, 1}), "Predictions should be binary (0 or 1)"
        assert len(unique_predictions) >= 1, "Should have at least one prediction class"
        
        # Prediction distribution should be reasonable (not all same class)
        class_0_ratio = sum(final_predictions == 0) / len(final_predictions)
        assert 0.1 <= class_0_ratio <= 0.9, f"Class distribution should be reasonable, got {class_0_ratio:.2f}"
        
        print(f"✅ E2E Classification Pipeline completed successfully!")
        print(f"   - Training samples: 400")
        print(f"   - Test samples: 100") 
        print(f"   - Model: LogisticRegression")
        print(f"   - Prediction accuracy reasonable: Class 0 ratio = {class_0_ratio:.2f}")
        print(f"   - MLflow run: {train_result.run_id}")
        
        return {
            'train_result': train_result,
            'inference_result': inference_result,
            'model': model,
            'predictions_path': predictions_path,
            'class_distribution': class_0_ratio
        }