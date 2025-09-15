"""
Unit tests for training pipeline with preprocessor improvements.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
import numpy as np
from types import SimpleNamespace

from src.pipelines.train_pipeline import run_train_pipeline


class TestTrainPipeline:
    """Test training pipeline with improved preprocessor handling"""
    
    @patch('src.pipelines.train_pipeline.mlflow')
    @patch('src.pipelines.train_pipeline.Factory')
    @patch('src.pipelines.train_pipeline.Console')
    @patch('src.pipelines.train_pipeline.log_enhanced_model_with_schema')
    @patch('src.pipelines.train_pipeline.get_pip_requirements')
    def test_train_with_preprocessor(
        self,
        mock_get_pip_reqs,
        mock_log_model,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test training pipeline with preprocessor"""
        # Setup mocks
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        mock_console.pipeline_context.return_value.__enter__ = Mock()
        mock_console.pipeline_context.return_value.__exit__ = Mock()
        
        # Mock MLflow
        mock_run = Mock()
        mock_run.info.run_id = 'train_123'
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        # Mock Factory and components
        mock_factory = Mock()
        mock_factory_cls.return_value = mock_factory
        
        # Mock data adapter
        mock_data_adapter = Mock()
        train_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        mock_data_adapter.read.return_value = train_df
        mock_factory.create_data_adapter.return_value = mock_data_adapter
        
        # Mock components
        mock_fetcher = Mock()
        mock_fetcher.fetch.return_value = train_df  # No augmentation
        mock_factory.create_fetcher.return_value = mock_fetcher
        
        mock_datahandler = Mock()
        X_train = train_df[['feature1', 'feature2']].iloc[:3]
        X_test = train_df[['feature1', 'feature2']].iloc[3:]
        y_train = train_df['target'].iloc[:3]
        y_test = train_df['target'].iloc[3:]
        mock_datahandler.split_and_prepare.return_value = (
            X_train, y_train, {}, X_test, y_test, {}
        )
        mock_factory.create_datahandler.return_value = mock_datahandler
        
        # Mock preprocessor
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.side_effect = lambda x: x.values  # Convert to numpy
        mock_factory.create_preprocessor.return_value = mock_preprocessor
        
        # Mock trainer
        mock_trainer = Mock()
        mock_model = Mock()
        mock_trainer.train.return_value = (mock_model, {'epochs': 10})
        mock_factory.create_trainer.return_value = mock_trainer
        
        # Mock model
        mock_factory.create_model.return_value = Mock()
        
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'accuracy': 0.95}
        mock_factory.create_evaluator.return_value = mock_evaluator
        
        # Mock PyfuncWrapper
        mock_wrapper = Mock()
        mock_wrapper.signature = Mock()
        mock_wrapper.data_schema = Mock()
        mock_factory.create_pyfunc_wrapper.return_value = mock_wrapper
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {
            'seed': 42,
            'run_name': 'test_run'
        }
        mock_settings.recipe.task_choice = 'classification'
        mock_settings.recipe.model.class_path = 'sklearn.RandomForestClassifier'
        mock_settings.config.environment.name = 'test'
        mock_settings.recipe.data.loader.source_uri = 'data/train.csv'
        mock_settings.recipe.data.data_interface = Mock()
        mock_settings.recipe.data.fetcher = None
        mock_settings.config = Mock()
        mock_settings.config.output = None  # No preprocessed output
        
        # Run pipeline
        result = run_train_pipeline(
            settings=mock_settings,
            context_params={'test': True}
        )
        
        # Assertions
        assert result.run_id == 'train_123'
        assert result.model_uri == 'runs:/train_123/model'
        
        # Verify preprocessor was used correctly
        mock_preprocessor.fit.assert_called_once()
        assert mock_preprocessor.transform.call_count == 2  # train and test
        
        # Verify trainer received preprocessed data
        trainer_call = mock_trainer.train.call_args
        assert trainer_call[1]['X_train'] is not None
        assert trainer_call[1]['X_val'] is not None
        
    @patch('src.pipelines.train_pipeline.mlflow')
    @patch('src.pipelines.train_pipeline.Factory')
    @patch('src.pipelines.train_pipeline.Console')
    @patch('src.pipelines.train_pipeline.save_output')
    def test_train_with_preprocessed_output(
        self,
        mock_save_output,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test saving preprocessed data when configured"""
        # Setup mocks (similar to above but simplified)
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        
        mock_run = Mock()
        mock_run.info.run_id = 'train_456'
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        # Mock factory and minimal components
        mock_factory = Mock()
        mock_factory_cls.return_value = mock_factory
        
        # Create test data
        train_df = pd.DataFrame({
            'f1': [1, 2, 3, 4],
            'f2': ['a', 'b', 'a', 'b'],
            'target': [0, 1, 0, 1]
        })
        
        mock_data_adapter = Mock()
        mock_data_adapter.read.return_value = train_df
        mock_factory.create_data_adapter.return_value = mock_data_adapter
        
        # Mock components with proper return values
        mock_factory.create_fetcher.return_value = None
        
        mock_datahandler = Mock()
        X_train = train_df[['f1', 'f2']].iloc[:2]
        X_test = train_df[['f1', 'f2']].iloc[2:]
        y_train = train_df['target'].iloc[:2]
        y_test = train_df['target'].iloc[2:]
        mock_datahandler.split_and_prepare.return_value = (
            X_train, y_train, {}, X_test, y_test, {}
        )
        mock_factory.create_datahandler.return_value = mock_datahandler
        
        # Mock preprocessor that transforms data
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        # Simulate preprocessing (e.g., one-hot encoding)
        preprocessed_train = np.array([[1, 1, 0], [2, 0, 1]])
        preprocessed_test = np.array([[3, 1, 0], [4, 0, 1]])
        mock_preprocessor.transform.side_effect = [preprocessed_train, preprocessed_test]
        mock_factory.create_preprocessor.return_value = mock_preprocessor
        
        # Other mocks
        mock_factory.create_trainer.return_value = Mock(
            train=Mock(return_value=(Mock(), {}))
        )
        mock_factory.create_model.return_value = Mock()
        mock_factory.create_evaluator.return_value = Mock(
            evaluate=Mock(return_value={'acc': 0.9})
        )
        mock_factory.create_pyfunc_wrapper.return_value = Mock(
            signature=Mock(), data_schema=Mock()
        )
        
        # Mock settings WITH preprocessed output config
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {'seed': 42, 'run_name': 'test'}
        mock_settings.recipe.task_choice = 'classification'
        mock_settings.recipe.model.class_path = 'xgboost.XGBClassifier'
        mock_settings.config.environment.name = 'test'
        mock_settings.recipe.data.loader.source_uri = 'data.csv'
        mock_settings.recipe.data.data_interface = Mock()
        mock_settings.recipe.data.fetcher = None
        
        # Enable preprocessed output
        mock_settings.config.output = Mock()
        mock_settings.config.output.preprocessed = Mock()
        
        # Run pipeline
        result = run_train_pipeline(settings=mock_settings)
        
        # Verify save_output was called for preprocessed data
        mock_save_output.assert_called_once()
        save_call = mock_save_output.call_args
        
        # Check saved DataFrame structure
        saved_df = save_call[1]['df']
        assert 'split' in saved_df.columns  # Should have split indicator
        assert 'target' in saved_df.columns  # Should have target
        assert len(saved_df) == 4  # train + test samples
        
        # Verify output type
        assert save_call[1]['output_type'] == 'preprocessed'
        
    @patch('src.pipelines.train_pipeline.mlflow')
    @patch('src.pipelines.train_pipeline.Factory')
    @patch('src.pipelines.train_pipeline.Console')
    def test_train_without_preprocessor(
        self,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test training pipeline when preprocessor is None"""
        # Setup minimal mocks
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        
        mock_run = Mock()
        mock_run.info.run_id = 'train_789'
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        mock_factory = Mock()
        mock_factory_cls.return_value = mock_factory
        
        # Test data
        train_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [0, 1, 0]
        })
        
        mock_factory.create_data_adapter.return_value = Mock(
            read=Mock(return_value=train_df)
        )
        mock_factory.create_fetcher.return_value = None
        
        X = train_df[['x']]
        y = train_df['y']
        mock_factory.create_datahandler.return_value = Mock(
            split_and_prepare=Mock(return_value=(X[:2], y[:2], {}, X[2:], y[2:], {}))
        )
        
        # No preprocessor
        mock_factory.create_preprocessor.return_value = None
        
        # Mock trainer to verify it receives original data
        mock_trainer = Mock()
        mock_trainer.train.return_value = (Mock(), {})
        mock_factory.create_trainer.return_value = mock_trainer
        
        mock_factory.create_model.return_value = Mock()
        mock_factory.create_evaluator.return_value = Mock(
            evaluate=Mock(return_value={})
        )
        mock_factory.create_pyfunc_wrapper.return_value = Mock(
            signature=Mock(), data_schema=Mock()
        )
        
        # Settings
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {'seed': 42, 'run_name': 'test'}
        mock_settings.recipe.task_choice = 'regression'
        mock_settings.recipe.model.class_path = 'sklearn.LinearRegression'
        mock_settings.config.environment.name = 'test'
        mock_settings.recipe.data.loader.source_uri = 'data.csv'
        mock_settings.recipe.data.data_interface = Mock()
        mock_settings.recipe.data.fetcher = None
        mock_settings.config.output = None
        
        # Run pipeline
        result = run_train_pipeline(settings=mock_settings)
        
        # Verify trainer received original DataFrame data (not transformed)
        trainer_call = mock_trainer.train.call_args
        X_train_arg = trainer_call[1]['X_train']
        
        # Should be DataFrame when no preprocessor
        assert isinstance(X_train_arg, pd.DataFrame) or isinstance(X_train_arg, pd.Series)


class TestTrainPipelineErrorHandling:
    """Test error handling in training pipeline"""
    
    @patch('src.pipelines.train_pipeline.mlflow')
    @patch('src.pipelines.train_pipeline.Factory')
    @patch('src.pipelines.train_pipeline.Console')
    def test_data_load_failure(
        self,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test handling of data load failure"""
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        
        mock_factory = Mock()
        mock_factory_cls.return_value = mock_factory
        
        # Mock data adapter that fails
        mock_data_adapter = Mock()
        mock_data_adapter.read.side_effect = Exception("Data not found")
        mock_factory.create_data_adapter.return_value = mock_data_adapter
        
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {'seed': 42, 'run_name': 'test'}
        mock_settings.recipe.task_choice = 'classification'
        mock_settings.config.environment.name = 'test'
        mock_settings.recipe.data.loader.source_uri = 'missing.csv'
        
        with pytest.raises(Exception) as exc_info:
            run_train_pipeline(settings=mock_settings)
        
        assert "Data not found" in str(exc_info.value)
        
    @patch('src.pipelines.train_pipeline.mlflow')
    @patch('src.pipelines.train_pipeline.Factory')
    @patch('src.pipelines.train_pipeline.Console')
    def test_model_training_failure(
        self,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test handling of model training failure"""
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        
        mock_run = Mock()
        mock_run.info.run_id = 'train_fail'
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        mock_factory = Mock()
        mock_factory_cls.return_value = mock_factory
        
        # Setup components
        train_df = pd.DataFrame({'x': [1, 2], 'y': [0, 1]})
        mock_factory.create_data_adapter.return_value = Mock(
            read=Mock(return_value=train_df)
        )
        mock_factory.create_fetcher.return_value = None
        mock_factory.create_datahandler.return_value = Mock(
            split_and_prepare=Mock(return_value=(
                train_df[['x']], train_df['y'], {}, 
                train_df[['x']], train_df['y'], {}
            ))
        )
        mock_factory.create_preprocessor.return_value = None
        mock_factory.create_model.return_value = Mock()
        
        # Mock trainer that fails
        mock_trainer = Mock()
        mock_trainer.train.side_effect = Exception("Training failed: OOM")
        mock_factory.create_trainer.return_value = mock_trainer
        
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {'seed': 42, 'run_name': 'test'}
        mock_settings.recipe.task_choice = 'classification'
        mock_settings.recipe.model.class_path = 'model.Model'
        mock_settings.config.environment.name = 'test'
        mock_settings.recipe.data.loader.source_uri = 'data.csv'
        mock_settings.recipe.data.data_interface = Mock()
        mock_settings.recipe.data.fetcher = None
        mock_settings.config.output = None
        
        with pytest.raises(Exception) as exc_info:
            run_train_pipeline(settings=mock_settings)
        
        assert "Training failed: OOM" in str(exc_info.value)


class TestPyfuncWrapperCreation:
    """Test PyfuncWrapper creation with DataInterface schema"""
    
    @patch('src.pipelines.train_pipeline.mlflow')
    @patch('src.pipelines.train_pipeline.Factory')
    @patch('src.pipelines.train_pipeline.Console')
    @patch('src.pipelines.train_pipeline.log_enhanced_model_with_schema')
    def test_pyfunc_wrapper_with_datainterface(
        self,
        mock_log_model,
        mock_console_cls,
        mock_factory_cls,
        mock_mlflow
    ):
        """Test that PyfuncWrapper receives DataInterface schema"""
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        
        mock_run = Mock()
        mock_run.info.run_id = 'wrapper_test'
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        mock_factory = Mock()
        mock_factory_cls.return_value = mock_factory
        
        # Setup minimal components
        train_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'feature': [10, 20, 30],
            'target': [0, 1, 0]
        })
        
        mock_factory.create_data_adapter.return_value = Mock(
            read=Mock(return_value=train_df)
        )
        mock_factory.create_fetcher.return_value = None
        mock_factory.create_datahandler.return_value = Mock(
            split_and_prepare=Mock(return_value=(
                train_df[['user_id', 'feature']], 
                train_df['target'], {}, 
                train_df[['user_id', 'feature']], 
                train_df['target'], {}
            ))
        )
        mock_factory.create_preprocessor.return_value = None
        mock_factory.create_trainer.return_value = Mock(
            train=Mock(return_value=(Mock(), {}))
        )
        mock_factory.create_model.return_value = Mock()
        mock_factory.create_evaluator.return_value = Mock(
            evaluate=Mock(return_value={})
        )
        
        # Mock PyfuncWrapper creation
        mock_wrapper = Mock()
        mock_wrapper.signature = Mock()
        mock_wrapper.data_schema = Mock()
        mock_factory.create_pyfunc_wrapper.return_value = mock_wrapper
        
        # Settings with DataInterface
        mock_settings = Mock()
        mock_settings.recipe.model.computed = {'seed': 42, 'run_name': 'test'}
        mock_settings.recipe.task_choice = 'classification'
        mock_settings.recipe.model.class_path = 'model.Model'
        mock_settings.config.environment.name = 'test'
        mock_settings.recipe.data.loader.source_uri = 'data.csv'
        
        # DataInterface configuration
        mock_data_interface = Mock()
        mock_data_interface.entity_columns = ['user_id']
        mock_data_interface.target_column = 'target'
        mock_data_interface.feature_columns = ['feature']
        mock_settings.recipe.data.data_interface = mock_data_interface
        mock_settings.recipe.data.fetcher = None
        mock_settings.config.output = None
        
        # Run pipeline
        result = run_train_pipeline(settings=mock_settings)
        
        # Verify create_pyfunc_wrapper was called with training_df
        mock_factory.create_pyfunc_wrapper.assert_called_once()
        wrapper_call = mock_factory.create_pyfunc_wrapper.call_args
        
        # Should pass training DataFrame
        assert wrapper_call[1]['training_df'] is not None
        assert isinstance(wrapper_call[1]['training_df'], pd.DataFrame)
        
        # Should pass training results
        assert 'training_results' in wrapper_call[1]
        
        # Verify log_enhanced_model_with_schema was called
        mock_log_model.assert_called_once()
        log_call = mock_log_model.call_args
        
        # Should pass the wrapper and schemas
        assert log_call[1]['python_model'] == mock_wrapper
        assert log_call[1]['signature'] == mock_wrapper.signature
        assert log_call[1]['data_schema'] == mock_wrapper.data_schema
        assert 'input_example' in log_call[1]