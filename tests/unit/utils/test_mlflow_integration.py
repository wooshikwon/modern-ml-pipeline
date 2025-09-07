"""
Unit tests for MLflow Integration utilities.
Tests MLflow setup, model management, and enhanced Phase 5 features.
"""

import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, Mock, MagicMock, mock_open
from contextlib import contextmanager

from src.utils.integrations.mlflow_integration import (
    setup_mlflow,
    start_run,
    get_latest_run_id,
    get_model_uri,
    load_pyfunc_model,
    download_artifacts,
    create_model_signature,
    _infer_pandas_dtype_to_mlflow_type,
    create_enhanced_model_signature_with_schema,
    log_enhanced_model_with_schema
)


class TestSetupMLflow:
    """Test MLflow setup functionality."""
    
    @patch('src.utils.integrations.mlflow_integration.RichConsoleManager')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_setup_mlflow_success(self, mock_mlflow, mock_console_class):
        """Test successful MLflow setup."""
        # Arrange
        mock_settings = Mock()
        mock_settings.config.mlflow.tracking_uri = "http://localhost:5000"
        mock_settings.config.mlflow.experiment_name = "test_experiment"
        mock_settings.config.environment.name = "test"
        
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        # Act
        setup_mlflow(mock_settings)
        
        # Assert
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_console.log_milestone.assert_called_once_with("MLflow setup completed", "mlflow")
        assert mock_console.print.call_count == 3  # URI, Experiment, Environment
    
    @patch('src.utils.integrations.mlflow_integration.RichConsoleManager')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_setup_mlflow_different_environments(self, mock_mlflow, mock_console_class):
        """Test MLflow setup with different environments."""
        environments = [
            {"tracking_uri": "file:///tmp/mlruns", "experiment": "local_exp", "env": "local"},
            {"tracking_uri": "http://remote:5000", "experiment": "prod_exp", "env": "production"},
            {"tracking_uri": "sqlite:///mlflow.db", "experiment": "dev_exp", "env": "development"}
        ]
        
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        for env_config in environments:
            mock_settings = Mock()
            mock_settings.config.mlflow.tracking_uri = env_config["tracking_uri"]
            mock_settings.config.mlflow.experiment_name = env_config["experiment"]
            mock_settings.config.environment.name = env_config["env"]
            
            setup_mlflow(mock_settings)
            
            mock_mlflow.set_tracking_uri.assert_called_with(env_config["tracking_uri"])
            mock_mlflow.set_experiment.assert_called_with(env_config["experiment"])


class TestStartRun:
    """Test MLflow run context manager."""
    
    @patch('src.utils.integrations.mlflow_integration.RichConsoleManager')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_start_run_success(self, mock_logger, mock_mlflow, mock_console_class):
        """Test successful run creation and completion."""
        # Arrange
        mock_settings = Mock()
        mock_settings.config.mlflow.experiment_name = "test_exp"
        
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.start_run.return_value.__exit__.return_value = None
        
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        # Act & Assert
        with start_run(mock_settings, "test_run"):
            mock_mlflow.set_experiment.assert_called_once_with("test_exp")
            mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
            mock_console.log_milestone.assert_called_with(
                "MLflow Run started: test_run_123 (test_run)", "mlflow"
            )
        
        # Check success status was set
        mock_mlflow.set_tag.assert_called_with("status", "success")
        assert mock_console.log_milestone.call_count == 2  # start and finish
    
    @patch('src.utils.integrations.mlflow_integration.RichConsoleManager')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_start_run_with_exception(self, mock_logger, mock_mlflow, mock_console_class):
        """Test run context manager with exception."""
        # Arrange
        mock_settings = Mock()
        mock_settings.config.mlflow.experiment_name = "test_exp"
        
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.start_run.return_value.__exit__.return_value = None
        
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        # Act & Assert
        test_exception = ValueError("Test error")
        with pytest.raises(ValueError, match="Test error"):
            with start_run(mock_settings, "test_run"):
                raise test_exception
        
        # Check failure status was set
        mock_mlflow.set_tag.assert_called_with("status", "failed")
        mock_console.log_milestone.assert_any_call(f"MLflow Run failed: {test_exception}", "error")
        mock_logger.error.assert_called()


class TestGetLatestRunId:
    """Test latest run ID retrieval."""
    
    @patch('src.utils.integrations.mlflow_integration.setup_mlflow')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_get_latest_run_id_success(self, mock_logger, mock_mlflow, mock_setup):
        """Test successful latest run ID retrieval."""
        # Arrange
        mock_settings = Mock()
        experiment_name = "test_experiment"
        
        mock_experiment = Mock()
        mock_experiment.experiment_id = "exp_123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Mock runs DataFrame
        runs_df = pd.DataFrame({
            'run_id': ['run_123', 'run_456'],
            'start_time': ['2023-01-02', '2023-01-01']
        })
        mock_mlflow.search_runs.return_value = runs_df
        
        # Act
        result = get_latest_run_id(mock_settings, experiment_name)
        
        # Assert
        assert result == "run_123"  # Most recent run
        mock_setup.assert_called_once_with(mock_settings)
        mock_mlflow.get_experiment_by_name.assert_called_once_with(experiment_name)
        mock_mlflow.search_runs.assert_called_once_with(
            experiment_ids=["exp_123"],
            filter_string="tags.status = 'success'",
            order_by=["start_time DESC"],
            max_results=1
        )
        mock_logger.info.assert_called()
    
    @patch('src.utils.integrations.mlflow_integration.setup_mlflow')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_get_latest_run_id_no_experiment(self, mock_logger, mock_mlflow, mock_setup):
        """Test error when experiment doesn't exist."""
        # Arrange
        mock_settings = Mock()
        mock_mlflow.get_experiment_by_name.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Experiment.*을 찾을 수 없습니다"):
            get_latest_run_id(mock_settings, "nonexistent_experiment")
        
        mock_logger.error.assert_called()
    
    @patch('src.utils.integrations.mlflow_integration.setup_mlflow')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_get_latest_run_id_no_successful_runs(self, mock_logger, mock_mlflow, mock_setup):
        """Test error when no successful runs exist."""
        # Arrange
        mock_settings = Mock()
        mock_experiment = Mock()
        mock_experiment.experiment_id = "exp_123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_runs.return_value = pd.DataFrame()  # Empty DataFrame
        
        # Act & Assert
        with pytest.raises(ValueError, match="성공한 run을 찾을 수 없습니다"):
            get_latest_run_id(mock_settings, "test_experiment")


class TestGetModelUri:
    """Test model URI generation."""
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_get_model_uri_default(self, mock_logger):
        """Test default model URI generation."""
        result = get_model_uri("run_123")
        assert result == "runs:/run_123/model"
        mock_logger.debug.assert_called_with("생성된 모델 URI: runs:/run_123/model")
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_get_model_uri_custom_path(self, mock_logger):
        """Test model URI generation with custom artifact path."""
        result = get_model_uri("run_456", "custom_model")
        assert result == "runs:/run_456/custom_model"
        mock_logger.debug.assert_called_with("생성된 모델 URI: runs:/run_456/custom_model")
    
    def test_get_model_uri_variations(self):
        """Test various URI generation scenarios."""
        test_cases = [
            ("short_run", "model", "runs:/short_run/model"),
            ("very_long_run_id_12345", "artifacts/model", "runs:/very_long_run_id_12345/artifacts/model"),
            ("run_with_special-chars", "model_v2", "runs:/run_with_special-chars/model_v2")
        ]
        
        for run_id, artifact_path, expected in test_cases:
            result = get_model_uri(run_id, artifact_path)
            assert result == expected


class TestLoadPyfuncModel:
    """Test PyFunc model loading."""
    
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_load_pyfunc_model_runs_uri(self, mock_logger, mock_mlflow):
        """Test loading model from runs:// URI."""
        # Arrange
        mock_settings = Mock()
        mock_settings.config.mlflow.tracking_uri = "http://localhost:5000"
        
        model_uri = "runs:/run_123/model"
        mock_model = Mock()
        
        with patch('mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.download_artifacts.return_value = "/tmp/local/model"
            mock_mlflow.pyfunc.load_model.return_value = mock_model
            
            # Act
            result = load_pyfunc_model(mock_settings, model_uri)
            
            # Assert
            assert result == mock_model
            mock_client_class.assert_called_once_with(tracking_uri="http://localhost:5000")
            mock_client.download_artifacts.assert_called_once_with(run_id="run_123", path="model")
            mock_mlflow.pyfunc.load_model.assert_called_once_with(model_uri="/tmp/local/model")
            mock_logger.info.assert_called()
    
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_load_pyfunc_model_local_path(self, mock_logger, mock_mlflow):
        """Test loading model from local path."""
        # Arrange
        mock_settings = Mock()
        mock_settings.config.mlflow.tracking_uri = "http://localhost:5000"
        
        model_uri = "/local/path/to/model"
        mock_model = Mock()
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        # Act
        result = load_pyfunc_model(mock_settings, model_uri)
        
        # Assert
        assert result == mock_model
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.pyfunc.load_model.assert_called_once_with(model_uri="/local/path/to/model")
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_load_pyfunc_model_invalid_runs_uri(self, mock_logger):
        """Test error with invalid runs:// URI."""
        mock_settings = Mock()
        invalid_uri = "runs:/invalid"
        
        with pytest.raises(ValueError, match="올바른 'runs:/' URI가 아닙니다"):
            load_pyfunc_model(mock_settings, invalid_uri)
        
        mock_logger.error.assert_called()
    
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_load_pyfunc_model_download_error(self, mock_logger, mock_mlflow):
        """Test error during artifact download."""
        mock_settings = Mock()
        mock_settings.config.mlflow.tracking_uri = "http://localhost:5000"
        
        with patch('mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.download_artifacts.side_effect = Exception("Download failed")
            
            with pytest.raises(Exception, match="Download failed"):
                load_pyfunc_model(mock_settings, "runs:/run_123/model")
            
            mock_logger.error.assert_called()


class TestDownloadArtifacts:
    """Test artifact download functionality."""
    
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_download_artifacts_success(self, mock_logger, mock_mlflow):
        """Test successful artifact download."""
        # Arrange
        mock_settings = Mock()
        mock_settings.config.mlflow.tracking_uri = "http://localhost:5000"
        
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/artifacts/model"
        
        # Act
        result = download_artifacts(mock_settings, "run_123", "model")
        
        # Assert
        assert result == "/tmp/artifacts/model"
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_123",
            artifact_path="model", 
            dst_path=None
        )
        mock_logger.info.assert_called()
    
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_download_artifacts_with_dst_path(self, mock_logger, mock_mlflow):
        """Test artifact download with destination path."""
        mock_settings = Mock()
        mock_settings.config.mlflow.tracking_uri = "http://localhost:5000"
        
        mock_mlflow.artifacts.download_artifacts.return_value = "/custom/path/model"
        
        result = download_artifacts(mock_settings, "run_456", "model", "/custom/path")
        
        assert result == "/custom/path/model"
        mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
            run_id="run_456",
            artifact_path="model",
            dst_path="/custom/path"
        )
    
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_download_artifacts_error(self, mock_logger, mock_mlflow):
        """Test error during artifact download."""
        mock_settings = Mock()
        mock_mlflow.artifacts.download_artifacts.side_effect = Exception("Download failed")
        
        with pytest.raises(Exception, match="Download failed"):
            download_artifacts(mock_settings, "run_123", "model")
        
        mock_logger.error.assert_called()


class TestCreateModelSignature:
    """Test ModelSignature creation."""
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_create_model_signature_basic(self, mock_logger):
        """Test basic ModelSignature creation."""
        # Arrange
        input_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [1.5, 2.5, 3.5],
            'category': ['A', 'B', 'C']
        })
        output_df = pd.DataFrame({
            'prediction': [0.1, 0.8, 0.3]
        })
        
        # Act
        signature = create_model_signature(input_df, output_df)
        
        # Assert
        assert signature is not None
        assert len(signature.inputs.inputs) == 3  # 3 input columns
        assert len(signature.outputs.inputs) == 1  # 1 output column
        assert len(signature.params.params) == 2   # run_mode, return_intermediate
        
        # Check parameter specs
        param_names = [p.name for p in signature.params.params]
        assert "run_mode" in param_names
        assert "return_intermediate" in param_names
        
        mock_logger.info.assert_called()
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_create_model_signature_different_types(self, mock_logger):
        """Test ModelSignature creation with different data types."""
        # Arrange with diverse data types
        input_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'bool_col': [True, False, True],
            'str_col': ['a', 'b', 'c'],
            'datetime_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })
        output_df = pd.DataFrame({
            'prediction': [0.1, 0.8, 0.3],
            'confidence': [0.9, 0.7, 0.85]
        })
        
        # Act
        signature = create_model_signature(input_df, output_df)
        
        # Assert
        assert len(signature.inputs.inputs) == 5  # 5 input columns
        assert len(signature.outputs.inputs) == 2  # 2 output columns
        
        # Check input types are properly inferred
        input_types = [col.type for col in signature.inputs.inputs]
        assert "long" in input_types      # int_col
        assert "double" in input_types    # float_col
        assert "boolean" in input_types   # bool_col
        assert "string" in input_types    # str_col
        assert "datetime" in input_types  # datetime_col
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_create_model_signature_error_handling(self, mock_logger):
        """Test ModelSignature creation error handling."""
        # Arrange with problematic data
        input_df = pd.DataFrame()  # Empty DataFrame
        output_df = pd.DataFrame({'prediction': [1, 2, 3]})
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise some kind of error
            create_model_signature(input_df, output_df)
        
        mock_logger.error.assert_called()


class TestInferPandasDtypeToMlflowType:
    """Test pandas dtype to MLflow type conversion."""
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_integer_types(self, mock_logger):
        """Test integer dtype conversion."""
        integer_dtypes = [
            (pd.Series([1, 2, 3], dtype='int8').dtype, "long"),
            (pd.Series([1, 2, 3], dtype='int16').dtype, "long"),
            (pd.Series([1, 2, 3], dtype='int32').dtype, "long"),
            (pd.Series([1, 2, 3], dtype='int64').dtype, "long"),
            (pd.Series([1, 2, 3], dtype='uint8').dtype, "long"),
            (pd.Series([1, 2, 3], dtype='uint16').dtype, "long"),
            (pd.Series([1, 2, 3], dtype='uint32').dtype, "long"),
            (pd.Series([1, 2, 3], dtype='uint64').dtype, "long")
        ]
        
        for pandas_dtype, expected_mlflow_type in integer_dtypes:
            result = _infer_pandas_dtype_to_mlflow_type(pandas_dtype)
            assert result == expected_mlflow_type
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_float_types(self, mock_logger):
        """Test float dtype conversion."""
        float_dtypes = [
            (pd.Series([1.1, 2.2, 3.3], dtype='float16').dtype, "double"),
            (pd.Series([1.1, 2.2, 3.3], dtype='float32').dtype, "double"),
            (pd.Series([1.1, 2.2, 3.3], dtype='float64').dtype, "double")
        ]
        
        for pandas_dtype, expected_mlflow_type in float_dtypes:
            result = _infer_pandas_dtype_to_mlflow_type(pandas_dtype)
            assert result == expected_mlflow_type
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_other_types(self, mock_logger):
        """Test other dtype conversions."""
        other_dtypes = [
            (pd.Series([True, False, True]).dtype, "boolean"),
            (pd.Series(['a', 'b', 'c']).dtype, "string"),
            (pd.to_datetime(['2023-01-01', '2023-01-02']).dtype, "datetime")
        ]
        
        for pandas_dtype, expected_mlflow_type in other_dtypes:
            result = _infer_pandas_dtype_to_mlflow_type(pandas_dtype)
            assert result == expected_mlflow_type
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_unknown_dtype(self, mock_logger):
        """Test unknown dtype handling."""
        # Create a custom dtype that should fall to default
        unknown_dtype = Mock()
        unknown_dtype.name = "unknown_type"
        
        result = _infer_pandas_dtype_to_mlflow_type(unknown_dtype)
        
        assert result == "string"  # Default fallback
        mock_logger.warning.assert_called()


class TestCreateEnhancedModelSignatureWithSchema:
    """Test Phase 5 enhanced model signature creation."""
    
    @patch('src.utils.integrations.mlflow_integration.generate_training_schema_metadata')
    @patch('src.utils.integrations.mlflow_integration.create_model_signature')
    @patch('src.utils.integrations.mlflow_integration.logger')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_create_enhanced_signature_success(self, mock_mlflow, mock_logger, 
                                              mock_create_signature, mock_generate_schema):
        """Test successful enhanced signature creation."""
        # Arrange
        mock_mlflow.__version__ = "2.0.0"
        
        training_df = pd.DataFrame({
            'entity_id': [1, 2, 3],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'feature1': [1.1, 2.2, 3.3]
        })
        
        data_interface_config = {"entity_columns": ["entity_id"], "timestamp_column": "timestamp"}
        
        # Mock schema generation
        mock_schema_metadata = {
            'inference_columns': ['entity_id', 'timestamp'],
            'timestamp_column': 'timestamp',
            'schema_version': '2.0'
        }
        mock_generate_schema.return_value = mock_schema_metadata
        
        # Mock signature creation
        mock_signature = Mock()
        mock_create_signature.return_value = mock_signature
        
        # Act
        signature, data_schema = create_enhanced_model_signature_with_schema(
            training_df, data_interface_config
        )
        
        # Assert
        assert signature == mock_signature
        assert isinstance(data_schema, dict)
        
        # Check enhanced schema includes Phase 5 features
        assert data_schema['mlflow_version'] == "2.0.0"
        assert 'signature_created_at' in data_schema
        assert 'phase_integration' in data_schema
        assert data_schema['phase_integration']['phase_5_enhanced_artifact'] is True
        assert data_schema['artifact_self_descriptive'] is True
        assert data_schema['reproduction_guaranteed'] is True
        
        mock_generate_schema.assert_called_once_with(training_df, data_interface_config)
        mock_logger.info.assert_called()
    
    @patch('src.utils.integrations.mlflow_integration.generate_training_schema_metadata')
    @patch('src.utils.integrations.mlflow_integration.create_model_signature')
    @patch('src.utils.integrations.mlflow_integration.logger')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_create_enhanced_signature_timestamp_conversion(self, mock_mlflow, mock_logger,
                                                           mock_create_signature, mock_generate_schema):
        """Test timestamp column conversion in enhanced signature."""
        # Arrange
        mock_mlflow.__version__ = "2.0.0"
        
        training_df = pd.DataFrame({
            'entity_id': [1, 2, 3],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],  # String timestamps
            'feature1': [1.1, 2.2, 3.3]
        })
        
        data_interface_config = {"entity_columns": ["entity_id"]}
        
        mock_schema_metadata = {
            'inference_columns': ['entity_id', 'timestamp'],
            'timestamp_column': 'timestamp',
            'schema_version': '2.0'
        }
        mock_generate_schema.return_value = mock_schema_metadata
        
        mock_signature = Mock()
        mock_create_signature.return_value = mock_signature
        
        # Act
        signature, data_schema = create_enhanced_model_signature_with_schema(
            training_df, data_interface_config
        )
        
        # Assert
        assert signature == mock_signature
        # Check that create_model_signature was called with processed data
        mock_create_signature.assert_called_once()
        call_args = mock_create_signature.call_args[0]
        input_example = call_args[0]
        
        # Check columns are filtered correctly
        assert list(input_example.columns) == ['entity_id', 'timestamp']


class TestLogEnhancedModelWithSchema:
    """Test Phase 5 enhanced model logging."""
    
    @patch('src.utils.integrations.mlflow_integration.RichConsoleManager')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.json')
    def test_log_enhanced_model_success(self, mock_json, mock_mlflow, mock_console_class):
        """Test successful enhanced model logging."""
        # Arrange
        mock_mlflow.__version__ = "2.0.0"
        
        mock_python_model = Mock()
        mock_signature = Mock()
        data_schema = {
            'schema_version': '2.0',
            'inference_columns': ['entity_id', 'timestamp'],
            'artifact_self_descriptive': True
        }
        input_example = pd.DataFrame({'entity_id': [1], 'timestamp': ['2023-01-01']})
        
        mock_console = Mock()
        mock_progress_tracker = Mock()
        mock_console.progress_tracker.return_value.__enter__.return_value = Mock()
        mock_console.progress_tracker.return_value.__exit__.return_value = None
        mock_console_class.return_value = mock_console
        
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.active_run.return_value = mock_run
        
        mock_json.dumps.return_value = '{"schema": "data"}'
        
        # Act
        log_enhanced_model_with_schema(
            mock_python_model, mock_signature, data_schema, input_example
        )
        
        # Assert
        # Check main model logging
        mock_mlflow.pyfunc.log_model.assert_called_once_with(
            artifact_path="model",
            python_model=mock_python_model,
            signature=mock_signature,
            pip_requirements=None,
            input_example=input_example,
            metadata={"data_schema": '{"schema": "data"}'}
        )
        
        # Check additional metadata logging
        assert mock_mlflow.log_dict.call_count == 3  # data_schema, compatibility_info, phase_summary
        
        # Check console interactions
        mock_console.log_phase.assert_called_with("MLflow Experiment Tracking", "📤")
        mock_console.display_run_info.assert_called_with(
            run_id="test_run_123",
            model_uri="runs:/test_run_123/model"
        )
        mock_console.log_milestone.assert_called_with(
            "Enhanced Model + metadata MLflow storage completed", "success"
        )
    
    @patch('src.utils.integrations.mlflow_integration.RichConsoleManager')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_log_enhanced_model_with_pip_requirements(self, mock_mlflow, mock_console_class):
        """Test enhanced model logging with pip requirements."""
        # Arrange
        mock_mlflow.__version__ = "2.0.0"
        
        mock_python_model = Mock()
        mock_signature = Mock()
        data_schema = {'schema_version': '2.0'}
        input_example = pd.DataFrame({'feature': [1]})
        pip_requirements = ["pandas==1.5.0", "scikit-learn==1.2.0"]
        
        mock_console = Mock()
        mock_console.progress_tracker.return_value.__enter__.return_value = Mock()
        mock_console.progress_tracker.return_value.__exit__.return_value = None
        mock_console_class.return_value = mock_console
        
        mock_mlflow.active_run.return_value = None  # No active run
        
        # Act
        log_enhanced_model_with_schema(
            mock_python_model, mock_signature, data_schema, input_example, pip_requirements
        )
        
        # Assert
        call_args = mock_mlflow.pyfunc.log_model.call_args[1]
        assert call_args['pip_requirements'] == pip_requirements
    
    @patch('src.utils.integrations.mlflow_integration.RichConsoleManager')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_log_enhanced_model_compatibility_info(self, mock_mlflow, mock_console_class):
        """Test compatibility info structure in enhanced model logging."""
        # Arrange
        mock_mlflow.__version__ = "2.1.0"
        
        mock_console = Mock()
        mock_console.progress_tracker.return_value.__enter__.return_value = Mock()
        mock_console.progress_tracker.return_value.__exit__.return_value = None
        mock_console_class.return_value = mock_console
        
        # Act
        log_enhanced_model_with_schema(Mock(), Mock(), {}, pd.DataFrame({'x': [1]}))
        
        # Assert
        # Find the compatibility_info log_dict call
        compatibility_call = None
        for call in mock_mlflow.log_dict.call_args_list:
            if call[0][1] == "model/compatibility_info.json":
                compatibility_call = call[0][0]
                break
        
        assert compatibility_call is not None
        assert compatibility_call['artifact_version'] == '2.0'
        assert compatibility_call['mlflow_version'] == '2.1.0'
        assert compatibility_call['features_enabled']['self_descriptive_artifact'] is True
        assert compatibility_call['backward_compatibility']['supports_legacy_models'] is False
        assert compatibility_call['quality_assurance']['reproducibility_guaranteed'] is True
    
    @patch('src.utils.integrations.mlflow_integration.RichConsoleManager')
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_log_enhanced_model_phase_summary(self, mock_mlflow, mock_console_class):
        """Test phase summary structure in enhanced model logging."""
        # Arrange
        mock_console = Mock()
        mock_console.progress_tracker.return_value.__enter__.return_value = Mock()
        mock_console.progress_tracker.return_value.__exit__.return_value = None
        mock_console_class.return_value = mock_console
        
        # Act
        log_enhanced_model_with_schema(Mock(), Mock(), {}, pd.DataFrame({'x': [1]}))
        
        # Assert
        # Find the phase_summary log_dict call
        phase_summary_call = None
        for call in mock_mlflow.log_dict.call_args_list:
            if call[0][1] == "model/phase_integration_summary.json":
                phase_summary_call = call[0][0]
                break
        
        assert phase_summary_call is not None
        assert 'phase_1' in phase_summary_call
        assert 'phase_2' in phase_summary_call
        assert 'phase_3' in phase_summary_call
        assert 'phase_4' in phase_summary_call
        assert 'phase_5' in phase_summary_call
        
        # Check specific phase information
        phase_5 = phase_summary_call['phase_5']
        assert phase_5['name'] == '완전 자기 기술 Artifact'
        assert '100% 재현성 보장' in phase_5['achievements']


class TestMLflowIntegrationEdgeCases:
    """Test edge cases and error scenarios."""
    
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_parse_runs_uri_invalid_formats(self, mock_mlflow):
        """Test _parse_runs_uri with invalid URI formats."""
        mock_settings = Mock()
        
        invalid_uris = [
            "runs:/",
            "runs://",
            "runs:/run_id_only",
            "not_runs_uri",
            "runs:run_without_slash",
            ""
        ]
        
        for invalid_uri in invalid_uris:
            with pytest.raises((ValueError, Exception)):
                load_pyfunc_model(mock_settings, invalid_uri)
    
    def test_get_model_uri_edge_cases(self):
        """Test model URI generation with edge case inputs."""
        edge_cases = [
            ("", "model", "runs:///model"),
            ("run_id", "", "runs:/run_id/"),
            ("", "", "runs:///"),
            ("run/with/slashes", "model", "runs:/run/with/slashes/model")
        ]
        
        for run_id, artifact_path, expected in edge_cases:
            result = get_model_uri(run_id, artifact_path)
            assert result == expected
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_create_model_signature_empty_dataframes(self, mock_logger):
        """Test signature creation with empty DataFrames."""
        empty_input = pd.DataFrame()
        empty_output = pd.DataFrame()
        
        with pytest.raises(Exception):
            create_model_signature(empty_input, empty_output)
        
        mock_logger.error.assert_called()
    
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_dtype_inference_edge_cases(self, mock_logger):
        """Test dtype inference with edge case dtypes."""
        # Create a mock dtype with unusual name
        unusual_dtype = Mock()
        unusual_dtype.name = "custom_dtype_123"
        
        result = _infer_pandas_dtype_to_mlflow_type(unusual_dtype)
        
        assert result == "string"  # Should default to string
        mock_logger.warning.assert_called_with(
            f"알 수 없는 pandas dtype: {unusual_dtype}, 'string'으로 처리"
        )


class TestMLflowIntegrationPerformance:
    """Test performance characteristics of MLflow integration."""
    
    @patch('src.utils.integrations.mlflow_integration.create_model_signature')
    @patch('src.utils.integrations.mlflow_integration.logger')
    def test_large_dataframe_signature_creation(self, mock_logger, mock_create_signature):
        """Test signature creation with large DataFrames."""
        # Arrange large DataFrames
        n_rows = 10000
        n_cols = 100
        
        large_input = pd.DataFrame(
            np.random.randn(n_rows, n_cols),
            columns=[f'feature_{i}' for i in range(n_cols)]
        )
        large_output = pd.DataFrame({
            'prediction': np.random.randn(n_rows)
        })
        
        # Mock to avoid actual processing
        mock_signature = Mock()
        mock_create_signature.return_value = mock_signature
        
        # Act - should handle large data efficiently
        result = create_model_signature(large_input, large_output)
        
        # Assert
        assert result == mock_signature
        mock_create_signature.assert_called_once()
    
    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_multiple_concurrent_setups(self, mock_mlflow):
        """Test multiple MLflow setups don't interfere."""
        # Arrange multiple settings
        settings_list = []
        for i in range(5):
            settings = Mock()
            settings.config.mlflow.tracking_uri = f"http://localhost:500{i}"
            settings.config.mlflow.experiment_name = f"experiment_{i}"
            settings.config.environment.name = f"env_{i}"
            settings_list.append(settings)
        
        # Act - setup multiple MLflow instances
        for settings in settings_list:
            setup_mlflow(settings)
        
        # Assert - each setup was called correctly
        assert mock_mlflow.set_tracking_uri.call_count == 5
        assert mock_mlflow.set_experiment.call_count == 5
        
        # Check last setup values
        mock_mlflow.set_tracking_uri.assert_called_with("http://localhost:5004")
        mock_mlflow.set_experiment.assert_called_with("experiment_4")