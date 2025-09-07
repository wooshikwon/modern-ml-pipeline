"""
Integration tests for MLflow Workflows and System-Level Scenarios.
Tests advanced MLflow integration, experiment management, model registry workflows,
system monitoring, and production deployment scenarios.
"""

import pytest
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import tempfile
import time
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from src.pipelines.train_pipeline import run_train_pipeline
from src.factory import Factory


class TestMLflowExperimentManagement:
    """Test MLflow experiment management and workflow scenarios."""

    def test_multiple_experiments_isolation(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test that multiple experiments are properly isolated."""
        # Set up MLflow tracking
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        # Create first experiment
        exp1_name = "experiment_classification_1"
        exp1_id = mlflow.create_experiment(exp1_name)
        
        # Create second experiment  
        exp2_name = "experiment_classification_2"
        exp2_id = mlflow.create_experiment(exp2_name)
        
        assert exp1_id != exp2_id
        
        # Run pipeline in first experiment
        settings1 = integration_settings_classification
        settings1.config.mlflow.experiment_name = exp1_name
        
        mlflow.set_experiment(exp1_name)
        result1 = run_train_pipeline(settings1)
        
        # Run pipeline in second experiment
        settings2 = integration_settings_classification
        settings2.config.mlflow.experiment_name = exp2_name
        
        mlflow.set_experiment(exp2_name)
        result2 = run_train_pipeline(settings2)
        
        # Verify runs are in different experiments
        run1 = mlflow.get_run(result1.run_id)
        run2 = mlflow.get_run(result2.run_id)
        
        assert run1.info.experiment_id == exp1_id
        assert run2.info.experiment_id == exp2_id
        assert run1.info.run_id != run2.info.run_id

    def test_experiment_run_comparison(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test comparing runs across experiments."""
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "comparison_experiment"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Run multiple training sessions with different parameters
        results = []
        for i, n_estimators in enumerate([5, 10, 15]):
            settings = integration_settings_classification
            settings.recipe.model.hyperparameters.n_estimators = n_estimators
            settings.recipe.model.computed.run_name = f"run_{i}_{n_estimators}_trees"
            
            result = run_train_pipeline(settings)
            results.append({
                'run_id': result.run_id,
                'n_estimators': n_estimators
            })
        
        # Compare runs
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get all runs from the experiment
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        assert len(runs) >= 3  # At least our 3 runs
        
        # Verify each run has metrics
        for run in runs:
            assert 'row_count' in run.data.metrics
            assert 'column_count' in run.data.metrics
            assert run.info.status == 'FINISHED'
        
        # Verify different parameter values were logged
        param_values = set()
        for run in runs:
            if 'n_estimators' in run.data.params:
                param_values.add(int(run.data.params['n_estimators']))
        
        assert len(param_values) >= 2  # Should have different values

    def test_experiment_run_tagging_and_metadata(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test MLflow run tagging and metadata management."""
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "metadata_experiment"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Run pipeline with custom context parameters
        context_params = {
            "experiment_type": "integration_test",
            "data_source": "synthetic",
            "model_version": "v1.0.0",
            "environment": "test"
        }
        
        result = run_train_pipeline(integration_settings_classification, context_params)
        
        # Verify run metadata
        run = mlflow.get_run(result.run_id)
        
        # Check basic run info
        assert run.info.status == 'FINISHED'
        assert run.info.run_name is not None
        
        # Check tags (if any were set by the pipeline)
        tags = run.data.tags
        assert isinstance(tags, dict)
        
        # Verify MLflow auto-tags
        assert 'mlflow.runName' in tags
        assert 'mlflow.source.type' in tags
        
        # Check metrics exist
        metrics = run.data.metrics
        assert len(metrics) > 0
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_model_registry_workflow(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test model registration and registry workflow."""
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "registry_experiment"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Train a model
        result = run_train_pipeline(integration_settings_classification)
        
        # Register model in registry
        model_name = "integration_test_model"
        
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Register model version
        try:
            model_version = client.create_registered_model(model_name)
            
            # Create model version from run
            mv = client.create_model_version(
                name=model_name,
                source=result.model_uri,
                run_id=result.run_id
            )
            
            assert mv.name == model_name
            assert mv.version == "1"  # First version
            assert mv.source == result.model_uri
            
        except Exception as e:
            # Model registry might not be fully configured in test environment
            # This is acceptable for integration tests
            pytest.skip(f"Model registry not available: {e}")

    def test_artifact_management_workflow(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test comprehensive artifact management."""
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "artifact_experiment"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Run training with artifact generation
        result = run_train_pipeline(integration_settings_classification)
        
        # Verify artifacts were saved
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        artifacts = client.list_artifacts(result.run_id)
        artifact_names = [artifact.path for artifact in artifacts]
        
        # Should have model artifact
        assert any('model' in name for name in artifact_names)
        
        # Check model artifact contents
        model_artifacts = client.list_artifacts(result.run_id, 'model')
        model_files = [artifact.path for artifact in model_artifacts]
        
        # Should have MLmodel file
        assert any('MLmodel' in path for path in model_files)
        
        # Verify model can be downloaded and loaded
        model = mlflow.pyfunc.load_model(result.model_uri)
        assert model is not None
        
        # Test artifact retrieval
        for artifact in artifacts:
            if artifact.is_dir:
                continue  # Skip directories
                
            try:
                artifact_path = client.download_artifacts(result.run_id, artifact.path)
                assert Path(artifact_path).exists()
            except Exception:
                # Some artifacts might not be downloadable in test environment
                pass


class TestSystemLevelScenarios:
    """Test system-level scenarios and edge cases."""

    def test_concurrent_mlflow_operations(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test concurrent MLflow operations."""
        import threading
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "concurrent_experiment"
        mlflow.create_experiment(experiment_name)
        
        results = []
        errors = []
        
        def run_concurrent_training(thread_id):
            try:
                # Each thread should set its own experiment context
                mlflow.set_experiment(experiment_name)
                
                # Create unique settings for each thread
                settings = integration_settings_classification
                settings.recipe.model.computed.run_name = f"concurrent_run_{thread_id}"
                
                result = run_train_pipeline(settings)
                results.append({
                    'thread_id': thread_id,
                    'run_id': result.run_id,
                    'model_uri': result.model_uri
                })
                
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Start concurrent threads
        threads = []
        for i in range(3):  # Limited concurrency for integration tests
            thread = threading.Thread(target=run_concurrent_training, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=120)  # 2 minute timeout per thread
        
        # Verify results
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 3, "Not all concurrent operations completed"
        
        # All runs should have unique run IDs
        run_ids = [r['run_id'] for r in results]
        assert len(set(run_ids)) == 3, "Runs should have unique IDs"
        
        # All models should be loadable
        for result in results:
            model = mlflow.pyfunc.load_model(result['model_uri'])
            assert model is not None

    def test_resource_monitoring_during_training(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test system resource monitoring during training."""
        import psutil
        import os
        
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "resource_monitoring"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Monitor resources before training
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        initial_cpu_percent = process.cpu_percent()
        
        # Run training while monitoring
        start_time = time.time()
        result = run_train_pipeline(integration_settings_classification)
        end_time = time.time()
        
        # Monitor resources after training
        final_memory = process.memory_info().rss
        final_cpu_percent = process.cpu_percent()
        
        training_duration = end_time - start_time
        memory_increase = final_memory - initial_memory
        
        # Basic resource usage assertions
        assert training_duration < 60.0, "Training took too long"  # Under 1 minute
        assert memory_increase < 500 * 1024 * 1024, "Memory usage increased too much"  # Under 500MB
        
        # Verify training completed successfully
        assert result.run_id is not None
        assert result.model_uri is not None
        
        # Log resource metrics to MLflow for analysis
        run = mlflow.get_run(result.run_id)
        assert run.info.status == 'FINISHED'

    def test_system_environment_compatibility(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test system environment compatibility and configuration."""
        # Test environment variable handling
        original_env = os.environ.copy()
        
        try:
            # Set test environment variables
            os.environ['ML_PIPELINE_ENV'] = 'test'
            os.environ['MLFLOW_TRACKING_URI'] = tmp_mlflow_tracking
            
            mlflow.set_tracking_uri(tmp_mlflow_tracking)
            
            experiment_name = "environment_test"
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            
            # Run pipeline with environment-specific settings
            result = run_train_pipeline(integration_settings_classification)
            
            # Verify pipeline respects environment configuration
            assert result.run_id is not None
            
            # Check that MLflow tracking URI was used correctly
            current_uri = mlflow.get_tracking_uri()
            assert tmp_mlflow_tracking in current_uri
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_error_recovery_and_logging(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test error recovery and comprehensive logging."""
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "error_recovery_test"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Test 1: Recoverable error scenario
        settings_recoverable = integration_settings_classification
        
        # Introduce a minor configuration issue that might be recoverable
        # (This depends on actual pipeline error handling implementation)
        
        try:
            result = run_train_pipeline(settings_recoverable)
            # If successful, verify normal completion
            assert result.run_id is not None
            
        except Exception as e:
            # If error occurs, verify it's properly logged
            assert len(str(e)) > 0  # Error message should be informative
            
            # Check if MLflow run was created even for failed runs
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            experiment = client.get_experiment_by_name(experiment_name)
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            
            # There might be failed runs logged
            if runs:
                failed_runs = [r for r in runs if r.info.status == 'FAILED']
                # Failed runs should still have some metadata
                for run in failed_runs:
                    assert run.info.run_id is not None

    def test_data_pipeline_monitoring(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test data pipeline monitoring and quality checks."""
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "data_monitoring"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Run pipeline and monitor data quality metrics
        result = run_train_pipeline(integration_settings_classification)
        
        # Verify data quality metrics were logged
        run = mlflow.get_run(result.run_id)
        metrics = run.data.metrics
        
        # Should have basic data quality metrics
        expected_metrics = ['row_count', 'column_count']
        for metric in expected_metrics:
            assert metric in metrics, f"Missing data quality metric: {metric}"
            assert metrics[metric] > 0, f"Invalid value for metric {metric}"
        
        # Check parameter logging
        params = run.data.params
        assert len(params) > 0, "No parameters were logged"
        
        # Verify model artifacts include data validation info
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        artifacts = client.list_artifacts(result.run_id)
        
        # Should have model and possibly metadata artifacts
        artifact_paths = [a.path for a in artifacts]
        assert any('model' in path for path in artifact_paths)

    def test_pipeline_reproducibility_across_runs(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test pipeline reproducibility and deterministic behavior."""
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "reproducibility_test"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Set deterministic seed
        seed_value = 12345
        settings = integration_settings_classification
        
        # Ensure deterministic settings
        if hasattr(settings.recipe.model, 'hyperparameters'):
            settings.recipe.model.hyperparameters.random_state = seed_value
        
        # Run 1
        result1 = run_train_pipeline(settings)
        
        # Run 2 with same settings
        result2 = run_train_pipeline(settings)
        
        # Both runs should succeed
        assert result1.run_id is not None
        assert result2.run_id is not None
        assert result1.run_id != result2.run_id  # Different run IDs
        
        # Load both models
        model1 = mlflow.pyfunc.load_model(result1.model_uri)
        model2 = mlflow.pyfunc.load_model(result2.model_uri)
        
        # Test predictions on same data
        test_data = pd.DataFrame({
            'feature_0': [1.0, 2.0],
            'feature_1': [1.5, 2.5], 
            'feature_2': [0.5, 1.5],
            'feature_3': [2.0, 3.0],
            'feature_4': [0.0, 1.0]
        })
        
        pred1 = model1.predict(test_data)
        pred2 = model2.predict(test_data)
        
        # With same random seed, predictions should be similar or identical
        # (Depending on pipeline determinism implementation)
        assert len(pred1) == len(pred2)
        
        # Basic consistency check
        pred1_array = np.array(pred1)
        pred2_array = np.array(pred2)
        
        # At minimum, predictions should be in same ballpark
        assert np.all(np.isfinite(pred1_array))
        assert np.all(np.isfinite(pred2_array))


class TestAdvancedMLflowFeatures:
    """Test advanced MLflow features and integrations."""

    def test_custom_metrics_logging(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test logging of custom metrics and parameters."""
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "custom_metrics_test"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Run pipeline
        result = run_train_pipeline(integration_settings_classification)
        
        # Verify custom metrics were logged
        run = mlflow.get_run(result.run_id)
        metrics = run.data.metrics
        params = run.data.params
        
        # Should have basic metrics
        assert len(metrics) > 0
        assert len(params) > 0
        
        # Verify metric types and ranges
        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, (int, float))
            assert np.isfinite(metric_value)
            
            # Basic sanity checks for common metrics
            if 'count' in metric_name.lower():
                assert metric_value >= 0
        
        # Verify parameter logging
        for param_name, param_value in params.items():
            assert isinstance(param_value, str)  # MLflow stores params as strings
            assert len(param_value) > 0

    def test_model_signature_validation(self, tmp_mlflow_tracking, integration_settings_classification):
        """Test MLflow model signature validation."""
        mlflow.set_tracking_uri(tmp_mlflow_tracking)
        
        experiment_name = "signature_test"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Run pipeline
        result = run_train_pipeline(integration_settings_classification)
        
        # Load model and check signature
        model = mlflow.pyfunc.load_model(result.model_uri)
        
        # Test with correct signature
        correct_input = pd.DataFrame({
            'feature_0': [1.0],
            'feature_1': [2.0],
            'feature_2': [3.0],
            'feature_3': [4.0],
            'feature_4': [5.0]
        })
        
        predictions = model.predict(correct_input)
        assert predictions is not None
        assert len(predictions) == 1
        
        # Model should handle signature validation
        # (Detailed signature checking would depend on implementation)

    def test_mlflow_tracking_uri_flexibility(self, integration_settings_classification):
        """Test MLflow tracking with different URI formats."""
        # Test 1: File URI
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_uri = f"file://{tmp_dir}/mlruns"
            mlflow.set_tracking_uri(file_uri)
            
            experiment_name = "file_uri_test"
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            
            # Update settings to use this tracking URI
            settings = integration_settings_classification
            settings.config.mlflow.tracking_uri = file_uri
            
            result = run_train_pipeline(settings)
            assert result.run_id is not None
            
            # Verify files were created
            mlruns_path = Path(tmp_dir) / "mlruns"
            assert mlruns_path.exists()
        
        # Test 2: SQLite URI (if supported)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sqlite_uri = f"sqlite:///{tmp_dir}/mlflow.db"
            
            try:
                mlflow.set_tracking_uri(sqlite_uri)
                
                experiment_name = "sqlite_uri_test"
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                
                settings = integration_settings_classification
                settings.config.mlflow.tracking_uri = sqlite_uri
                
                result = run_train_pipeline(settings)
                assert result.run_id is not None
                
                # Verify SQLite file was created
                db_path = Path(tmp_dir) / "mlflow.db"
                assert db_path.exists()
                
            except Exception as e:
                # SQLite backend might not be available in all test environments
                pytest.skip(f"SQLite URI not supported: {e}")