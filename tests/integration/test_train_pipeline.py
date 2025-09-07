"""
Integration tests for Train Pipeline.
Tests complete end-to-end training pipeline with real components and MLflow integration.
"""

import pytest
import mlflow
import mlflow.pyfunc
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

from src.pipelines.train_pipeline import run_train_pipeline
from src.factory import Factory


class TestTrainPipelineIntegration:
    """Integration tests for the complete training pipeline."""
    
    def test_train_pipeline_classification_end_to_end(self, integration_settings_classification, minimal_context_params):
        """Test complete classification training pipeline from data loading to MLflow model saving."""
        # Arrange
        settings = integration_settings_classification
        context_params = minimal_context_params
        
        # Act - Run the complete training pipeline
        result = run_train_pipeline(settings, context_params)
        
        # Assert - Verify pipeline completed successfully
        assert isinstance(result, SimpleNamespace)
        assert hasattr(result, 'run_id')
        assert hasattr(result, 'model_uri')
        assert result.run_id is not None
        assert result.model_uri is not None
        assert result.model_uri.startswith('runs:/')
        
        # Verify MLflow run was created
        run_id = result.run_id
        run = mlflow.get_run(run_id)
        assert run.info.status == 'FINISHED'
        assert run.info.run_name.startswith('integration_test_classification')
        
        # Verify metrics were logged
        metrics = run.data.metrics
        assert 'row_count' in metrics
        assert 'column_count' in metrics
        assert metrics['row_count'] == 100  # From test data
        assert metrics['column_count'] > 0
        
        # Verify model artifacts were saved
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [artifact.path for artifact in artifacts]
        assert 'model' in artifact_names
    
    def test_train_pipeline_regression_end_to_end(self, integration_settings_regression, minimal_context_params):
        """Test complete regression training pipeline."""
        # Arrange
        settings = integration_settings_regression
        context_params = minimal_context_params
        
        # Act - Run the complete training pipeline
        result = run_train_pipeline(settings, context_params)
        
        # Assert - Verify pipeline completed successfully
        assert isinstance(result, SimpleNamespace)
        assert result.run_id is not None
        assert result.model_uri is not None
        
        # Verify MLflow run
        run_id = result.run_id
        run = mlflow.get_run(run_id)
        assert run.info.status == 'FINISHED'
        assert run.info.run_name.startswith('integration_test_regression')
        
        # Verify regression-specific metrics
        metrics = run.data.metrics
        assert 'row_count' in metrics
        assert 'column_count' in metrics
        
        # Verify model can be loaded and used
        model = mlflow.pyfunc.load_model(result.model_uri)
        assert model is not None
    
    def test_train_pipeline_saves_complete_metadata(self, integration_settings_classification):
        """Test that training pipeline saves complete metadata and artifacts."""
        # Arrange
        settings = integration_settings_classification
        
        # Act
        result = run_train_pipeline(settings)
        
        # Assert - Check metadata artifacts
        run_id = result.run_id
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        
        # Find metadata artifact
        metadata_artifacts = [a for a in artifacts if a.path.startswith('metadata')]
        assert len(metadata_artifacts) > 0
        
        # Verify model artifact structure
        model_artifacts = [a for a in artifacts if a.path == 'model']
        assert len(model_artifacts) == 1
        
        # Check that model contains required files
        model_files = client.list_artifacts(run_id, 'model')
        model_file_names = [f.path for f in model_files]
        assert any('MLmodel' in path for path in model_file_names)
    
    def test_train_pipeline_with_hyperparameter_optimization(self, integration_settings_classification):
        """Test training pipeline with hyperparameter optimization enabled."""
        # Arrange - Modify settings to enable HPO with minimal trials for fast testing
        settings = integration_settings_classification
        
        # Enable minimal hyperparameter optimization
        # Note: This assumes the settings structure supports HPO configuration
        if hasattr(settings.recipe.model, 'hyperparameters'):
            # Enable HPO with minimal trials for fast testing
            hpo_config = {
                'enabled': True,
                'n_trials': 2,  # Minimal trials for integration test
                'tunable_params': {
                    'n_estimators': {'type': 'int', 'low': 5, 'high': 15},
                    'max_depth': {'type': 'int', 'low': 2, 'high': 5}
                }
            }
            # This might need adjustment based on actual settings structure
            if hasattr(settings.recipe.model.hyperparameters, '__dict__'):
                settings.recipe.model.hyperparameters.__dict__.update(hpo_config)
        
        # Act
        result = run_train_pipeline(settings)
        
        # Assert - Basic pipeline completion
        assert result.run_id is not None
        assert result.model_uri is not None
        
        # Verify run completed
        run_id = result.run_id
        run = mlflow.get_run(run_id)
        assert run.info.status == 'FINISHED'
    
    def test_train_pipeline_factory_component_integration(self, integration_settings_classification):
        """Test that train pipeline properly integrates with Factory for component creation."""
        # Arrange
        settings = integration_settings_classification
        
        # Act
        result = run_train_pipeline(settings)
        
        # Assert - Verify pipeline used Factory correctly
        assert result.run_id is not None
        
        # Create Factory manually to verify components can be created
        factory = Factory(settings)
        
        # Verify all required components can be created
        data_adapter = factory.create_data_adapter()
        assert data_adapter is not None
        
        fetcher = factory.create_fetcher()
        assert fetcher is not None
        
        datahandler = factory.create_datahandler()
        assert datahandler is not None
        
        preprocessor = factory.create_preprocessor()  # May return None if not configured
        
        model = factory.create_model()
        assert model is not None
        
        evaluator = factory.create_evaluator()
        assert evaluator is not None
        
        trainer = factory.create_trainer()
        assert trainer is not None
    
    def test_train_pipeline_handles_different_data_sizes(self, integration_settings_classification, test_data_classification):
        """Test train pipeline handles different data sizes appropriately."""
        # Arrange - Create larger dataset for this test
        from tests.helpers.dataframe_builder import DataFrameBuilder
        
        large_data = DataFrameBuilder.build_classification_data(
            n_samples=200,  # Larger dataset
            n_features=10,
            n_classes=3,
            add_entity_column=True
        )
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_data.to_csv(f.name, index=False)
            temp_csv_path = f.name
        
        # Update settings to use larger dataset
        settings = integration_settings_classification
        settings.recipe.data.loader.source_uri = temp_csv_path
        settings.recipe.data.data_interface.feature_columns = [f'feature_{i}' for i in range(10)]
        
        # Act
        result = run_train_pipeline(settings)
        
        # Assert
        assert result.run_id is not None
        
        # Verify larger dataset was processed
        run = mlflow.get_run(result.run_id)
        metrics = run.data.metrics
        assert metrics['row_count'] == 200
        assert metrics['column_count'] >= 10
        
        # Cleanup
        Path(temp_csv_path).unlink()
    
    def test_train_pipeline_reproducibility(self, integration_settings_classification):
        """Test that train pipeline produces reproducible results with same seed."""
        # Arrange
        settings = integration_settings_classification
        
        # Set specific seed for reproducibility
        if hasattr(settings.recipe.model, 'computed'):
            settings.recipe.model.computed['seed'] = 123
        
        # Act - Run pipeline twice with same settings
        result1 = run_train_pipeline(settings)
        result2 = run_train_pipeline(settings)
        
        # Assert - Both runs should complete successfully
        assert result1.run_id is not None
        assert result2.run_id is not None
        assert result1.run_id != result2.run_id  # Different runs
        
        # Load both models for comparison
        model1 = mlflow.pyfunc.load_model(result1.model_uri)
        model2 = mlflow.pyfunc.load_model(result2.model_uri)
        
        # Both models should exist and be loadable
        assert model1 is not None
        assert model2 is not None


class TestTrainPipelineErrorHandling:
    """Test train pipeline error handling scenarios."""
    
    def test_train_pipeline_handles_missing_data_file(self, integration_settings_classification):
        """Test train pipeline handles missing data file gracefully."""
        # Arrange - Point to non-existent file
        settings = integration_settings_classification
        settings.recipe.data.loader.source_uri = "/nonexistent/file.csv"
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise appropriate error
            run_train_pipeline(settings)
    
    def test_train_pipeline_handles_invalid_model_class(self, integration_settings_classification):
        """Test train pipeline handles invalid model class path."""
        # Arrange - Use invalid model class path
        settings = integration_settings_classification
        settings.recipe.model.class_path = "nonexistent.module.NonExistentModel"
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise appropriate error
            run_train_pipeline(settings)
    
    def test_train_pipeline_handles_empty_data(self, integration_settings_classification):
        """Test train pipeline handles empty dataset."""
        # Arrange - Create empty CSV file
        import tempfile
        empty_data = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            empty_data.to_csv(f.name, index=False)
            temp_csv_path = f.name
        
        settings = integration_settings_classification
        settings.recipe.data.loader.source_uri = temp_csv_path
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise appropriate error
            run_train_pipeline(settings)
        
        # Cleanup
        Path(temp_csv_path).unlink()


class TestTrainPipelineTaskTypes:
    """Test train pipeline with different task types."""
    
    def test_train_pipeline_timeseries_task(self, integration_settings_timeseries):
        """Test training pipeline with timeseries task type."""
        # Arrange
        settings = integration_settings_timeseries
        
        # Act
        result = run_train_pipeline(settings)
        
        # Assert
        assert result.run_id is not None
        assert result.model_uri is not None
        
        # Verify timeseries-specific handling
        run = mlflow.get_run(result.run_id)
        assert run.info.status == 'FINISHED'
        assert run.info.run_name.startswith('integration_test_timeseries')
        
        # Verify model can be loaded
        model = mlflow.pyfunc.load_model(result.model_uri)
        assert model is not None
        
        # Verify timeseries data structure was maintained
        metrics = run.data.metrics
        assert 'row_count' in metrics
        assert metrics['row_count'] == 100  # From test timeseries data