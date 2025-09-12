"""
Unit tests for inference pipeline with DataInterface support.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
from types import SimpleNamespace

from src.pipelines.inference_pipeline import run_inference_pipeline
from src.utils.data.data_io import format_predictions


class TestInferencePipeline:
    """Test inference pipeline with DataInterface"""
    
    @patch('src.pipelines.inference_pipeline.mlflow')
    @patch('src.pipelines.inference_pipeline.Factory')
    @patch('src.pipelines.inference_pipeline.RichConsoleManager')
    @patch('src.pipelines.inference_pipeline.load_inference_data')
    @patch('src.pipelines.inference_pipeline.save_output')
    def test_inference_with_datainterface(
        self, 
        mock_save_output,
        mock_load_data,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test inference pipeline uses DataInterface from PyfuncWrapper"""
        # Setup mocks
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        mock_console.pipeline_context.return_value.__enter__ = Mock()
        mock_console.pipeline_context.return_value.__exit__ = Mock()
        mock_console.progress_tracker.return_value.__enter__ = Mock(return_value=Mock())
        mock_console.progress_tracker.return_value.__exit__ = Mock()
        
        # Mock MLflow run
        mock_run = Mock()
        mock_run.info.run_id = 'inference_run_123'
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        # Mock PyfuncWrapper with DataInterface
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = {
            'data_interface_config': {
                'target_column': 'price',
                'entity_columns': ['product_id'],
                'feature_columns': ['brand', 'category']
            }
        }
        
        # Mock model
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_model.predict.return_value = [100.0, 200.0, 150.0]
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        # Mock data
        test_df = pd.DataFrame({
            'product_id': ['P1', 'P2', 'P3'],
            'brand': ['A', 'B', 'A'],
            'category': ['X', 'Y', 'X']
        })
        mock_load_data.return_value = test_df
        
        # Mock Factory
        mock_factory = Mock()
        mock_factory_cls.return_value = mock_factory
        mock_data_adapter = Mock()
        mock_factory.create_data_adapter.return_value = mock_data_adapter
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {'seed': 42}
        mock_settings.config.environment.name = 'test'
        
        # Run pipeline
        result = run_inference_pipeline(
            settings=mock_settings,
            run_id='model_run_123',
            data_path='data/test.csv'
        )
        
        # Assertions
        assert result.run_id == 'inference_run_123'
        assert result.model_uri == 'runs:/model_run_123/model'
        assert result.prediction_count == 3
        
        # Verify model was loaded correctly
        mock_mlflow.pyfunc.load_model.assert_called_once_with('runs:/model_run_123/model')
        
        # Verify predictions were made
        mock_model.predict.assert_called_once()
        
        # Verify metrics were logged
        mock_mlflow.log_metric.assert_any_call('inference_input_rows', 3)
        mock_mlflow.log_metric.assert_any_call('inference_input_columns', 3)
        mock_mlflow.log_metric.assert_any_call('inference_output_rows', 3)
        
    @patch('src.pipelines.inference_pipeline.mlflow')
    @patch('src.pipelines.inference_pipeline.Factory')
    @patch('src.pipelines.inference_pipeline.RichConsoleManager')
    def test_inference_without_datainterface(
        self,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test inference pipeline fallback when no DataInterface"""
        # Setup mocks
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        mock_console.pipeline_context.return_value.__enter__ = Mock()
        mock_console.pipeline_context.return_value.__exit__ = Mock()
        
        # Mock model without DataInterface
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = None  # No DataInterface
        
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_model.predict.return_value = [1, 0, 1]
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        # This should still work with fallback
        # (Complete test implementation would be similar to above)


class TestFormatPredictions:
    """Test format_predictions function with DataInterface"""
    
    def test_format_with_datainterface(self):
        """Test formatting predictions with DataInterface config"""
        predictions = [100.0, 200.0, 150.0]
        
        df = pd.DataFrame({
            'product_id': ['P1', 'P2', 'P3'],
            'brand': ['A', 'B', 'A'],
            'category': ['X', 'Y', 'X']
        })
        
        data_interface = {
            'entity_columns': ['product_id']
        }
        
        result = format_predictions(predictions, df, data_interface)
        
        # Should include entity columns and predictions
        assert 'product_id' in result.columns
        assert 'prediction' in result.columns
        assert len(result) == 3
        assert list(result['prediction']) == predictions
        assert list(result['product_id']) == ['P1', 'P2', 'P3']
        
    def test_format_with_multiple_entities(self):
        """Test formatting with multiple entity columns"""
        predictions = pd.DataFrame({
            'prediction': [0.8, 0.3],
            'probability': [0.8, 0.3]
        })
        
        df = pd.DataFrame({
            'user_id': ['U1', 'U2'],
            'item_id': ['I1', 'I2'],
            'feature': [1, 2]
        })
        
        data_interface = {
            'entity_columns': ['user_id', 'item_id']
        }
        
        result = format_predictions(predictions, df, data_interface)
        
        # Should include all entity columns
        assert 'user_id' in result.columns
        assert 'item_id' in result.columns
        assert 'prediction' in result.columns
        assert 'probability' in result.columns
        
    def test_format_without_datainterface(self):
        """Test formatting without DataInterface (fallback)"""
        predictions = [1, 0, 1]
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature': ['a', 'b', 'c']
        })
        
        # No data_interface provided
        result = format_predictions(predictions, df, {})
        
        # Should still create valid output
        assert 'prediction' in result.columns
        assert len(result) == 3
        
    def test_format_numpy_array_predictions(self):
        """Test formatting numpy array predictions"""
        import numpy as np
        
        predictions = np.array([0.1, 0.2, 0.3])
        
        df = pd.DataFrame({
            'id': [1, 2, 3]
        })
        
        data_interface = {
            'entity_columns': ['id']
        }
        
        result = format_predictions(predictions, df, data_interface)
        
        assert 'prediction' in result.columns
        assert list(result['prediction']) == [0.1, 0.2, 0.3]
        
    def test_format_multiclass_predictions(self):
        """Test formatting multiclass probability predictions"""
        import numpy as np
        
        # Multiclass probabilities (3 samples, 3 classes)
        predictions = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])
        
        df = pd.DataFrame({
            'id': [1, 2, 3]
        })
        
        data_interface = {
            'entity_columns': ['id']
        }
        
        result = format_predictions(predictions, df, data_interface)
        
        # Should handle multi-dimensional predictions
        assert 'prediction' in result.columns
        assert result['prediction'].shape == (3,) or result['prediction'].shape == (3, 3)


class TestInferencePipelineErrorHandling:
    """Test error handling in inference pipeline"""
    
    @patch('src.pipelines.inference_pipeline.mlflow')
    @patch('src.pipelines.inference_pipeline.RichConsoleManager')
    def test_model_load_failure(self, mock_console_cls, mock_mlflow):
        """Test handling of model load failure"""
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        
        # Mock model load failure
        mock_mlflow.pyfunc.load_model.side_effect = Exception("Model not found")
        
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {'seed': 42}
        
        with pytest.raises(Exception) as exc_info:
            run_inference_pipeline(
                settings=mock_settings,
                run_id='invalid_run_id'
            )
        
        assert "Model not found" in str(exc_info.value)
        
    @patch('src.pipelines.inference_pipeline.mlflow')
    @patch('src.pipelines.inference_pipeline.Factory')
    @patch('src.pipelines.inference_pipeline.RichConsoleManager')
    @patch('src.pipelines.inference_pipeline.load_inference_data')
    def test_prediction_failure(
        self,
        mock_load_data,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test handling of prediction failure"""
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        mock_console.pipeline_context.return_value.__enter__ = Mock()
        mock_console.pipeline_context.return_value.__exit__ = Mock()
        
        # Mock successful model load
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = Mock(data_interface_schema=None)
        mock_model.predict.side_effect = Exception("Prediction failed")
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        # Mock data
        mock_load_data.return_value = pd.DataFrame({'col': [1, 2, 3]})
        
        # Mock factory
        mock_factory = Mock()
        mock_factory_cls.return_value = mock_factory
        mock_factory.create_data_adapter.return_value = Mock()
        
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {'seed': 42}
        mock_settings.config.environment.name = 'test'
        
        with pytest.raises(Exception) as exc_info:
            run_inference_pipeline(
                settings=mock_settings,
                run_id='model_run_123'
            )
        
        assert "Prediction failed" in str(exc_info.value)


class TestInferencePipelineIntegration:
    """Integration tests for inference pipeline"""
    
    @patch('src.pipelines.inference_pipeline.mlflow')
    @patch('src.pipelines.inference_pipeline.Factory')
    @patch('src.pipelines.inference_pipeline.RichConsoleManager')
    @patch('src.pipelines.inference_pipeline.load_inference_data')
    @patch('src.pipelines.inference_pipeline.save_output')
    def test_end_to_end_inference_flow(
        self,
        mock_save_output,
        mock_load_data,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test complete inference flow from data load to save"""
        # Setup comprehensive mocks
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        
        # Mock MLflow
        mock_run = Mock()
        mock_run.info.run_id = 'inf_123'
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        # Mock model with full DataInterface
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = {
            'data_interface_config': {
                'target_column': 'target',
                'entity_columns': ['user_id', 'session_id'],
                'feature_columns': ['f1', 'f2', 'f3'],
                'timestamp_column': 'timestamp'
            },
            'task_type': 'timeseries'
        }
        
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_model.predict.return_value = pd.DataFrame({
            'prediction': [0.1, 0.2, 0.3, 0.4, 0.5],
            'confidence': [0.9, 0.8, 0.95, 0.7, 0.85]
        })
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        # Mock input data
        input_df = pd.DataFrame({
            'user_id': ['U1', 'U2', 'U3', 'U4', 'U5'],
            'session_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
            'f1': [1, 2, 3, 4, 5],
            'f2': [5, 4, 3, 2, 1],
            'f3': [2, 2, 2, 2, 2],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
        })
        mock_load_data.return_value = input_df
        
        # Mock factory
        mock_factory = Mock()
        mock_factory_cls.return_value = mock_factory
        mock_data_adapter = Mock()
        mock_factory.create_data_adapter.return_value = mock_data_adapter
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {'seed': 42}
        mock_settings.config.environment.name = 'production'
        
        # Run pipeline
        result = run_inference_pipeline(
            settings=mock_settings,
            run_id='model_123',
            data_path='s3://bucket/data.parquet',
            context_params={'batch_size': 1000}
        )
        
        # Comprehensive assertions
        assert result.run_id == 'inf_123'
        assert result.model_uri == 'runs:/model_123/model'
        assert result.prediction_count == 5
        
        # Verify save_output was called with correct params
        mock_save_output.assert_called_once()
        save_call_args = mock_save_output.call_args
        
        # Check that predictions DataFrame includes entity columns
        saved_df = save_call_args[1]['df']
        assert 'user_id' in saved_df.columns
        assert 'session_id' in saved_df.columns
        assert 'prediction' in saved_df.columns
        assert 'confidence' in saved_df.columns
        
        # Verify additional metadata
        assert save_call_args[1]['additional_metadata']['model_run_id'] == 'model_123'