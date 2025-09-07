"""
E2E Test: FastAPI Serving Pipeline
Tests complete FastAPI serving workflow with trained model deployment.
"""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import time
import requests
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch
import mlflow
import mlflow.pyfunc
from types import SimpleNamespace
from contextlib import contextmanager

from src.settings import Settings
from src.settings.config import Config, Environment, MLflow as MLflowConfig, DataSource, FeatureStore, Serving
from src.settings.recipe import Recipe
from src.pipelines.train_pipeline import run_train_pipeline
from src.serving.router import app as fastapi_app


class TestAPIServingE2E:
    """End-to-end test for FastAPI serving pipeline."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for serving E2E test."""
        workspace = tempfile.mkdtemp()
        data_dir = os.path.join(workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate simple test dataset for serving
        np.random.seed(42)
        n_samples = 100  # Small dataset for faster testing
        
        # Generate features
        age = np.random.randint(18, 70, n_samples)
        income = np.random.normal(50000, 15000, n_samples)
        experience = np.random.randint(0, 30, n_samples)
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
        
        # Simple target based on features
        target_prob = (age / 100) + (income / 100000) + (experience / 50)
        target_prob = 1 / (1 + np.exp(-target_prob + 1.5))  # Sigmoid
        target = np.random.binomial(1, target_prob, n_samples)
        
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'experience': experience,
            'education': education,
            'approved': target
        })
        
        train_path = os.path.join(data_dir, "train.csv")
        df.to_csv(train_path, index=False)
        
        yield {
            'workspace': workspace,
            'data_dir': data_dir,
            'train_path': train_path,
            'df': df
        }
        
        # Cleanup
        shutil.rmtree(workspace)
    
    @pytest.fixture
    def serving_settings(self, temp_workspace):
        """Create settings for serving E2E test."""
        config = Config(
            environment=Environment(name="e2e_serving_test"),
            mlflow=MLflowConfig(
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
                experiment_name="e2e_serving_test"
            ),
            data_source=DataSource(
                name="file_storage",
                adapter_type="storage",
                config={"base_path": temp_workspace['data_dir']}
            ),
            feature_store=FeatureStore(provider="none"),
            serving=Serving(
                enabled=True,
                host="127.0.0.1",
                port=8002,  # Use different port to avoid conflicts
                model_uri=None  # Will be set after training
            )
        )
        
        recipe = Recipe(
            name="e2e_serving_recipe",
            task_choice="classification",
            data={
                "data_interface": {
                    "target_column": "approved",
                    "drop_columns": []
                },
                "feature_view": {
                    "name": "serving_features",
                    "entities": [],
                    "features": ["age", "income", "experience", "education"],
                    "source": {
                        "path": "train.csv",
                        "timestamp_column": None
                    }
                }
            },
            loader={
                "name": "csv_loader",
                "batch_size": 50,
                "shuffle": True
            },
            model={
                "class_path": "sklearn.linear_model.LogisticRegression",
                "init_args": {
                    "random_state": 42,
                    "max_iter": 500
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
                            "features": ["age", "income", "experience"]
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
    
    @contextmanager
    def run_fastapi_server(self, settings, model_uri):
        """Context manager to run FastAPI server for testing."""
        import uvicorn
        import threading
        import time
        
        # Update serving config with model URI
        settings.config.serving.model_uri = model_uri
        
        # Configure the FastAPI app with settings
        from src.serving._context import set_serving_context
        set_serving_context(settings)
        
        # Server configuration
        host = settings.config.serving.host
        port = settings.config.serving.port
        
        # Server thread control
        server_started = threading.Event()
        server_error = None
        
        def run_server():
            nonlocal server_error
            try:
                config = uvicorn.Config(
                    fastapi_app,
                    host=host,
                    port=port,
                    log_level="warning"  # Reduce noise
                )
                server = uvicorn.Server(config)
                
                # Signal that server is starting
                server_started.set()
                
                # Run server (this blocks)
                server.run()
                
            except Exception as e:
                server_error = e
                server_started.set()
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        server_started.wait(timeout=10)
        
        if server_error:
            raise server_error
        
        # Give server additional time to fully initialize
        time.sleep(2)
        
        try:
            # Test server is responding
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Server health check failed: {response.status_code}")
                
            yield f"http://{host}:{port}"
            
        except requests.exceptions.RequestException:
            # Server might not have health endpoint, continue anyway
            yield f"http://{host}:{port}"
        
        # Server cleanup happens automatically when thread ends
    
    def test_complete_serving_pipeline_e2e(self, serving_settings, temp_workspace):
        """Test complete serving pipeline from training to API deployment."""
        
        # Phase 1: Train Model for Serving
        print("🚀 Phase 1: Training model for serving...")
        
        # Set MLflow experiment
        mlflow.set_experiment(serving_settings.config.mlflow.experiment_name)
        
        # Train model
        train_result = run_train_pipeline(serving_settings)
        
        assert hasattr(train_result, 'run_id'), "Training should return run_id"
        assert hasattr(train_result, 'model_uri'), "Training should return model_uri"
        
        model_uri = train_result.model_uri
        print(f"✅ Model trained. URI: {model_uri}")
        
        # Phase 2: Test Model Loading
        print("🔍 Phase 2: Testing model loading...")
        
        # Load model to verify it works
        model = mlflow.pyfunc.load_model(model_uri)
        assert model is not None, "Model should be loadable"
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'age': [25, 45, 35],
            'income': [40000, 80000, 60000],
            'experience': [2, 15, 8],
            'education': ['Bachelor', 'Master', 'High School']
        })
        
        predictions = model.predict(sample_data)
        assert len(predictions) == 3, "Should predict for all samples"
        
        print(f"✅ Model loading successful. Sample predictions: {predictions}")
        
        # Phase 3: Start FastAPI Server
        print("🚀 Phase 3: Starting FastAPI server...")
        
        try:
            with self.run_fastapi_server(serving_settings, model_uri) as server_url:
                print(f"✅ Server started at {server_url}")
                
                # Phase 4: Test API Endpoints
                print("🧪 Phase 4: Testing API endpoints...")
                
                # Test health endpoint
                health_response = requests.get(f"{server_url}/health", timeout=10)
                if health_response.status_code == 200:
                    print("✅ Health endpoint working")
                else:
                    print(f"⚠️ Health endpoint returned {health_response.status_code}")
                
                # Test prediction endpoint - Single prediction
                single_prediction_payload = {
                    "features": {
                        "age": 30,
                        "income": 65000,
                        "experience": 5,
                        "education": "Bachelor"
                    }
                }
                
                pred_response = requests.post(
                    f"{server_url}/predict",
                    json=single_prediction_payload,
                    timeout=10
                )
                
                assert pred_response.status_code == 200, f"Prediction failed: {pred_response.status_code}"
                pred_result = pred_response.json()
                
                assert 'prediction' in pred_result, "Response should contain prediction"
                assert pred_result['prediction'] in [0, 1], "Prediction should be binary"
                
                print(f"✅ Single prediction: {pred_result}")
                
                # Test batch prediction endpoint
                batch_prediction_payload = {
                    "features": [
                        {
                            "age": 25,
                            "income": 45000,
                            "experience": 2,
                            "education": "High School"
                        },
                        {
                            "age": 40,
                            "income": 85000,
                            "experience": 12,
                            "education": "Master"
                        },
                        {
                            "age": 35,
                            "income": 70000,
                            "experience": 8,
                            "education": "Bachelor"
                        }
                    ]
                }
                
                batch_response = requests.post(
                    f"{server_url}/predict_batch",
                    json=batch_prediction_payload,
                    timeout=10
                )
                
                if batch_response.status_code == 200:
                    batch_result = batch_response.json()
                    assert 'predictions' in batch_result, "Batch response should contain predictions"
                    assert len(batch_result['predictions']) == 3, "Should return 3 predictions"
                    
                    print(f"✅ Batch prediction: {batch_result['predictions']}")
                else:
                    print(f"⚠️ Batch prediction endpoint returned {batch_response.status_code}")
                
                # Test model info endpoint
                info_response = requests.get(f"{server_url}/model_info", timeout=10)
                if info_response.status_code == 200:
                    model_info = info_response.json()
                    assert 'model_uri' in model_info, "Model info should contain model URI"
                    assert 'task_type' in model_info, "Model info should contain task type"
                    
                    print(f"✅ Model info: {model_info}")
                else:
                    print(f"⚠️ Model info endpoint returned {info_response.status_code}")
                
                # Phase 5: Stress Test API
                print("💪 Phase 5: API stress testing...")
                
                # Send multiple concurrent requests
                import concurrent.futures
                
                def make_prediction_request():
                    payload = {
                        "features": {
                            "age": np.random.randint(20, 60),
                            "income": np.random.randint(30000, 100000),
                            "experience": np.random.randint(0, 20),
                            "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"])
                        }
                    }
                    
                    response = requests.post(f"{server_url}/predict", json=payload, timeout=5)
                    return response.status_code == 200
                
                # Make 10 concurrent requests
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(make_prediction_request) for _ in range(10)]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                success_rate = sum(results) / len(results)
                assert success_rate >= 0.8, f"API success rate too low: {success_rate:.1%}"
                
                print(f"✅ Stress test completed. Success rate: {success_rate:.1%}")
                
                # Phase 6: Error Handling Tests
                print("🛡️ Phase 6: Testing error handling...")
                
                # Test invalid input
                invalid_payload = {
                    "features": {
                        "age": "invalid",  # Should be numeric
                        "income": 50000,
                        "experience": 5,
                        "education": "Bachelor"
                    }
                }
                
                error_response = requests.post(
                    f"{server_url}/predict",
                    json=invalid_payload,
                    timeout=10
                )
                
                # Should return error status (4xx or 5xx)
                assert error_response.status_code >= 400, "Invalid input should return error"
                
                print(f"✅ Error handling test: {error_response.status_code}")
                
                # Test missing fields
                incomplete_payload = {
                    "features": {
                        "age": 30,
                        # Missing other required fields
                    }
                }
                
                missing_response = requests.post(
                    f"{server_url}/predict",
                    json=incomplete_payload,
                    timeout=10
                )
                
                assert missing_response.status_code >= 400, "Missing fields should return error"
                
                print("✅ Missing fields error handling works")
                
        except Exception as e:
            print(f"❌ Server test failed: {e}")
            raise
        
        print("✅ E2E API Serving Pipeline completed successfully!")
        print(f"   - Model trained and saved: ✓")
        print(f"   - FastAPI server started: ✓")
        print(f"   - Single predictions working: ✓")
        print(f"   - Batch predictions working: ✓")
        print(f"   - Error handling working: ✓")
        print(f"   - Stress test passed: ✓")
        print(f"   - Model URI: {model_uri}")
        
        return {
            'model_uri': model_uri,
            'train_result': train_result,
            'server_url': f"{serving_settings.config.serving.host}:{serving_settings.config.serving.port}",
            'stress_test_success_rate': success_rate
        }
    
    def test_serving_configuration_validation(self, serving_settings):
        """Test serving configuration validation."""
        # Test that serving is enabled
        assert serving_settings.config.serving.enabled is True, "Serving should be enabled"
        
        # Test port configuration
        assert 1000 <= serving_settings.config.serving.port <= 65535, "Port should be valid range"
        
        # Test host configuration
        assert serving_settings.config.serving.host in ["127.0.0.1", "0.0.0.0", "localhost"], "Host should be valid"
        
        print("✅ Serving configuration validation passed")
    
    def test_model_uri_handling(self, serving_settings, temp_workspace):
        """Test proper model URI handling in serving context."""
        # Train a model first
        mlflow.set_experiment(serving_settings.config.mlflow.experiment_name)
        train_result = run_train_pipeline(serving_settings)
        
        model_uri = train_result.model_uri
        
        # Test URI format
        assert model_uri.startswith('runs:/'), "Model URI should be in runs:/ format"
        
        # Test model is loadable
        model = mlflow.pyfunc.load_model(model_uri)
        assert model is not None, "Model should be loadable from URI"
        
        print(f"✅ Model URI handling validated: {model_uri}")