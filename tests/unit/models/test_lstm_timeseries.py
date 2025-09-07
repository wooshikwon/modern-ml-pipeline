"""
Unit tests for LSTM TimeSeries model.
Tests LSTM-based time series forecasting with 3D sequence data handling.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock, MagicMock

from src.models.custom.lstm_timeseries import LSTMTimeSeries


class TestLSTMTimeSeriesInitialization:
    """Test LSTM TimeSeries initialization."""
    
    def test_lstm_timeseries_default_initialization(self):
        """Test default parameter initialization."""
        model = LSTMTimeSeries()
        
        # Assert default parameters
        assert model.hidden_dim == 64
        assert model.num_layers == 2
        assert model.dropout == 0.2
        assert model.epochs == 100
        assert model.batch_size == 32
        assert model.learning_rate == 0.001
        assert model.early_stopping_patience == 10
        assert model.bidirectional is False
        
        # Assert internal state
        assert model.model is None
        assert model.is_fitted is False
        assert model.sequence_info is None
        assert model.handles_own_preprocessing is True
    
    def test_lstm_timeseries_custom_initialization(self):
        """Test initialization with custom parameters."""
        model = LSTMTimeSeries(
            hidden_dim=128,
            num_layers=3,
            dropout=0.3,
            epochs=50,
            batch_size=64,
            learning_rate=0.01,
            early_stopping_patience=5,
            bidirectional=True
        )
        
        assert model.hidden_dim == 128
        assert model.num_layers == 3
        assert model.dropout == 0.3
        assert model.epochs == 50
        assert model.batch_size == 64
        assert model.learning_rate == 0.01
        assert model.early_stopping_patience == 5
        assert model.bidirectional is True
    
    @patch('src.models.custom.lstm_timeseries.get_device')
    def test_lstm_timeseries_device_initialization(self, mock_get_device):
        """Test device initialization."""
        mock_device = torch.device('cuda:0')
        mock_get_device.return_value = mock_device
        
        model = LSTMTimeSeries()
        
        assert model.device == mock_device
        mock_get_device.assert_called_once()


class TestLSTMTimeSeriesSequenceInfoExtraction:
    """Test sequence information extraction methods."""
    
    def test_extract_sequence_info_with_metadata(self):
        """Test sequence info extraction from metadata."""
        model = LSTMTimeSeries()
        
        X = pd.DataFrame(np.random.randn(10, 30))  # 10 samples, 30 features (flattened)
        kwargs = {
            'original_sequence_shape': (10, 5, 6)  # 10 samples, 5 timesteps, 6 features
        }
        
        seq_info = model._extract_sequence_info(X, kwargs)
        
        assert seq_info['original_shape'] == (10, 5, 6)
        assert seq_info['sequence_length'] == 5
        assert seq_info['n_features'] == 6
        assert seq_info['from_datahandler'] is True
    
    def test_extract_sequence_info_without_metadata(self):
        """Test sequence info extraction fallback."""
        model = LSTMTimeSeries()
        
        # Create DataFrame with seq_feat naming pattern
        columns = []
        for seq in range(3):  # 3 timesteps
            for feat in range(4):  # 4 features
                columns.append(f'seq{seq}_feat{feat}')
        
        X = pd.DataFrame(np.random.randn(5, 12), columns=columns)
        kwargs = {}
        
        with patch.object(model, '_infer_sequence_info_from_dataframe') as mock_infer:
            mock_infer.return_value = {'test': 'value'}
            
            seq_info = model._extract_sequence_info(X, kwargs)
            
            assert seq_info == {'test': 'value'}
            mock_infer.assert_called_once_with(X)
    
    def test_infer_sequence_info_from_dataframe_seq_pattern(self):
        """Test sequence info inference from seq_feat column pattern."""
        model = LSTMTimeSeries()
        
        # Create DataFrame with seq_feat pattern: 3 timesteps, 4 features
        columns = []
        for seq in range(3):
            for feat in range(4):
                columns.append(f'seq{seq}_feat{feat}')
        
        X = pd.DataFrame(np.random.randn(5, 12), columns=columns)
        
        seq_info = model._infer_sequence_info_from_dataframe(X)
        
        assert seq_info['original_shape'] == (5, 3, 4)
        assert seq_info['sequence_length'] == 3
        assert seq_info['n_features'] == 4
        assert seq_info['from_datahandler'] is False
    
    def test_infer_sequence_info_from_dataframe_fallback(self):
        """Test sequence info inference fallback to default values."""
        model = LSTMTimeSeries()
        
        # Create DataFrame without seq_feat pattern
        X = pd.DataFrame(np.random.randn(5, 20), columns=[f'col{i}' for i in range(20)])
        
        seq_info = model._infer_sequence_info_from_dataframe(X)
        
        # Default: seq_len=10, n_feat = n_cols // seq_len = 20 // 10 = 2
        assert seq_info['original_shape'] == (5, 10, 2)
        assert seq_info['sequence_length'] == 10
        assert seq_info['n_features'] == 2
        assert seq_info['from_datahandler'] is False


class TestLSTMTimeSeriesDataReconstruction:
    """Test 3D sequence data reconstruction."""
    
    def test_reconstruct_3d_sequences_valid(self):
        """Test successful 3D sequence reconstruction."""
        model = LSTMTimeSeries()
        
        # Create flattened data: 5 samples, 3 timesteps, 4 features -> (5, 12)
        X_flat = pd.DataFrame(np.random.randn(5, 12))
        seq_info = {
            'original_shape': (10, 3, 4),  # Original shape (different sample count is OK)
            'sequence_length': 3,
            'n_features': 4
        }
        
        X_3d = model._reconstruct_3d_sequences(X_flat, seq_info)
        
        assert X_3d.shape == (5, 3, 4)  # Actual sample count used
        assert X_3d.dtype == np.float32
    
    def test_reconstruct_3d_sequences_invalid_features(self):
        """Test 3D reconstruction with invalid feature count."""
        model = LSTMTimeSeries()
        
        # Wrong number of features: expecting 3*4=12, but providing 15
        X_flat = pd.DataFrame(np.random.randn(5, 15))
        seq_info = {
            'original_shape': (10, 3, 4),
            'sequence_length': 3,
            'n_features': 4
        }
        
        with pytest.raises(ValueError, match="입력 데이터의 특성 수가 맞지 않습니다"):
            model._reconstruct_3d_sequences(X_flat, seq_info)


class TestLSTMTimeSeriesValidation:
    """Test data validation methods."""
    
    def test_validate_sequence_data_valid(self):
        """Test successful sequence data validation."""
        model = LSTMTimeSeries()
        
        X = np.random.randn(20, 5, 3).astype(np.float32)  # 20 samples, 5 timesteps, 3 features
        y = pd.Series(np.random.randn(20))
        
        # Should not raise any exception
        model._validate_sequence_data(X, y)
    
    def test_validate_sequence_data_not_3d(self):
        """Test validation failure for non-3D data."""
        model = LSTMTimeSeries()
        
        X = np.random.randn(20, 15)  # 2D instead of 3D
        y = pd.Series(np.random.randn(20))
        
        with pytest.raises(ValueError, match="3D 시퀀스 데이터가 필요합니다"):
            model._validate_sequence_data(X, y)
    
    def test_validate_sequence_data_length_mismatch(self):
        """Test validation failure for X-y length mismatch."""
        model = LSTMTimeSeries()
        
        X = np.random.randn(20, 5, 3).astype(np.float32)
        y = pd.Series(np.random.randn(15))  # Different length
        
        with pytest.raises(ValueError, match="X와 y의 길이가 다릅니다"):
            model._validate_sequence_data(X, y)
    
    def test_validate_sequence_data_small_dataset_warning(self):
        """Test warning for small datasets."""
        model = LSTMTimeSeries()
        
        X = np.random.randn(5, 5, 3).astype(np.float32)  # Small dataset
        y = pd.Series(np.random.randn(5))
        
        with patch('src.models.custom.lstm_timeseries.logger') as mock_logger:
            model._validate_sequence_data(X, y)
            mock_logger.warning.assert_called_once()


class TestLSTMTimeSeriesTimeSplit:
    """Test time-based data splitting."""
    
    def test_time_based_split_normal(self):
        """Test normal time-based splitting."""
        model = LSTMTimeSeries()
        
        X = np.random.randn(100, 5, 3).astype(np.float32)
        y = pd.Series(np.random.randn(100))
        
        X_train, X_val, y_train, y_val = model._time_based_split(X, y, val_ratio=0.2)
        
        assert X_train.shape == (80, 5, 3)
        assert X_val.shape == (20, 5, 3)
        assert len(y_train) == 80
        assert len(y_val) == 20
    
    def test_time_based_split_small_dataset(self):
        """Test time-based splitting with small dataset."""
        model = LSTMTimeSeries()
        
        X = np.random.randn(4, 5, 3).astype(np.float32)  # Very small
        y = pd.Series(np.random.randn(4))
        
        with patch('src.models.custom.lstm_timeseries.logger') as mock_logger:
            X_train, X_val, y_train, y_val = model._time_based_split(X, y)
            
            # Should skip validation split
            assert X_train.shape == (4, 5, 3)
            assert X_val is None
            assert len(y_train) == 4
            assert y_val is None
            mock_logger.warning.assert_called_once()
    
    def test_time_based_split_custom_ratio(self):
        """Test time-based splitting with custom validation ratio."""
        model = LSTMTimeSeries()
        
        X = np.random.randn(50, 5, 3).astype(np.float32)
        y = pd.Series(np.random.randn(50))
        
        X_train, X_val, y_train, y_val = model._time_based_split(X, y, val_ratio=0.3)
        
        assert X_train.shape == (35, 5, 3)  # 50 * 0.7 = 35
        assert X_val.shape == (15, 5, 3)    # 50 * 0.3 = 15
        assert len(y_train) == 35
        assert len(y_val) == 15


class TestLSTMTimeSeriesModelBuilding:
    """Test LSTM model architecture building."""
    
    def test_build_lstm_model_basic(self):
        """Test basic LSTM model building."""
        model = LSTMTimeSeries(
            hidden_dim=32, 
            num_layers=2, 
            dropout=0.1, 
            bidirectional=False
        )
        
        lstm_model = model._build_lstm_model(input_size=5)
        
        assert isinstance(lstm_model, nn.Module)
        assert hasattr(lstm_model, 'lstm')
        assert hasattr(lstm_model, 'fc')
        assert hasattr(lstm_model, 'dropout')
        
        # Check LSTM layer configuration
        assert lstm_model.lstm.input_size == 5
        assert lstm_model.lstm.hidden_size == 32
        assert lstm_model.lstm.num_layers == 2
        assert lstm_model.lstm.batch_first is True
        assert lstm_model.lstm.bidirectional is False
        
        # Check fully connected layer
        assert lstm_model.fc.in_features == 32  # hidden_dim
        assert lstm_model.fc.out_features == 1
    
    def test_build_lstm_model_bidirectional(self):
        """Test bidirectional LSTM model building."""
        model = LSTMTimeSeries(
            hidden_dim=64, 
            num_layers=3, 
            dropout=0.2, 
            bidirectional=True
        )
        
        lstm_model = model._build_lstm_model(input_size=10)
        
        assert lstm_model.lstm.bidirectional is True
        assert lstm_model.fc.in_features == 128  # hidden_dim * 2 for bidirectional
    
    def test_build_lstm_model_single_layer_no_dropout(self):
        """Test single layer LSTM (no dropout in LSTM layer)."""
        model = LSTMTimeSeries(
            hidden_dim=32,
            num_layers=1,
            dropout=0.3,
            bidirectional=False
        )
        
        lstm_model = model._build_lstm_model(input_size=5)
        
        # For single layer, LSTM dropout should be 0, but final dropout should be 0.3
        assert lstm_model.lstm.dropout == 0.0  # Single layer -> no LSTM dropout
        assert isinstance(lstm_model.dropout, nn.Dropout)
    
    def test_lstm_model_forward_pass(self):
        """Test LSTM model forward pass."""
        model = LSTMTimeSeries(hidden_dim=32, num_layers=2, dropout=0.1)
        lstm_model = model._build_lstm_model(input_size=5)
        lstm_model.eval()
        
        # Create test input: (batch_size=3, seq_len=10, input_size=5)
        test_input = torch.randn(3, 10, 5)
        
        with torch.no_grad():
            output = lstm_model(test_input)
        
        assert output.shape == (3, 1)  # (batch_size, 1)
        assert not torch.isnan(output).any()


class TestLSTMTimeSeriesFit:
    """Test LSTM model fitting process."""
    
    @patch('src.models.custom.lstm_timeseries.train_pytorch_model')
    @patch('src.models.custom.lstm_timeseries.create_dataloader') 
    @patch('src.models.custom.lstm_timeseries.set_seed')
    def test_fit_success(self, mock_set_seed, mock_create_dataloader, mock_train_pytorch_model):
        """Test successful model fitting."""
        # Setup
        model = LSTMTimeSeries(epochs=10, batch_size=16)
        
        # Create test data
        X_flat = pd.DataFrame(np.random.randn(50, 12))  # 50 samples, 12 features (3*4)
        y = pd.Series(np.random.randn(50))
        kwargs = {'original_sequence_shape': (50, 3, 4)}
        
        # Mock training components
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_create_dataloader.side_effect = [mock_train_loader, mock_val_loader]
        
        mock_train_pytorch_model.return_value = {
            'train_loss': [0.5, 0.3, 0.2],
            'val_loss': [0.6, 0.4, 0.3],
            'best_epoch': 2
        }
        
        # Execute
        result = model.fit(X_flat, y, **kwargs)
        
        # Assertions
        assert result is model  # Returns self
        assert model.is_fitted is True
        assert model.sequence_info is not None
        assert model.model is not None
        
        # Check method calls
        mock_set_seed.assert_called_once_with(42)
        assert mock_create_dataloader.call_count == 2  # train and val loaders
        mock_train_pytorch_model.assert_called_once()
    
    def test_fit_data_validation_failure(self):
        """Test fit failure due to data validation."""
        model = LSTMTimeSeries()
        
        # Invalid data: 2D instead of reconstructed 3D
        X_flat = pd.DataFrame(np.random.randn(10, 15))  # Wrong feature count
        y = pd.Series(np.random.randn(10))
        kwargs = {'original_sequence_shape': (10, 3, 4)}  # Expects 3*4=12 features, not 15
        
        with pytest.raises(ValueError):
            model.fit(X_flat, y, **kwargs)
    
    @patch('src.models.custom.lstm_timeseries.train_pytorch_model')
    @patch('src.models.custom.lstm_timeseries.create_dataloader')
    @patch('src.models.custom.lstm_timeseries.set_seed')
    def test_fit_small_dataset_no_validation(self, mock_set_seed, mock_create_dataloader, mock_train_pytorch_model):
        """Test fitting with small dataset (no validation split)."""
        model = LSTMTimeSeries()
        
        # Small dataset
        X_flat = pd.DataFrame(np.random.randn(4, 12))
        y = pd.Series(np.random.randn(4))
        kwargs = {'original_sequence_shape': (4, 3, 4)}
        
        mock_train_loader = MagicMock()  
        mock_create_dataloader.return_value = mock_train_loader  # Only train loader for small dataset
        
        mock_train_pytorch_model.return_value = {
            'train_loss': [0.5, 0.3],
            'val_loss': [],  # No validation
            'best_epoch': 1
        }
        
        result = model.fit(X_flat, y, **kwargs)
        
        assert result is model
        assert model.is_fitted is True
        
        # For small dataset, may only call create_dataloader once (no validation split)
        assert mock_create_dataloader.call_count >= 1
        
        # Check that training was called with appropriate parameters
        if mock_create_dataloader.call_count >= 1:
            train_call = mock_create_dataloader.call_args_list[0]  
            if len(train_call) > 1 and 'batch_size' in train_call[1]:
                assert train_call[1]['batch_size'] == model.batch_size
                assert train_call[1]['shuffle'] is False  # Time series should not shuffle


class TestLSTMTimeSeriesPredict:
    """Test LSTM model prediction."""
    
    def test_predict_not_fitted_error(self):
        """Test prediction error when model is not fitted."""
        model = LSTMTimeSeries()
        
        X = pd.DataFrame(np.random.randn(10, 12))
        
        with pytest.raises(RuntimeError, match="모델이 학습되지 않았습니다"):
            model.predict(X)
    
    @patch('src.models.custom.lstm_timeseries.predict_with_pytorch_model')
    @patch('src.models.custom.lstm_timeseries.create_dataloader')
    def test_predict_success(self, mock_create_dataloader, mock_predict_pytorch_model):
        """Test successful prediction."""
        # Setup fitted model
        model = LSTMTimeSeries()
        model.is_fitted = True
        model.model = MagicMock()  # Mock PyTorch model
        model.sequence_info = {
            'original_shape': (50, 3, 4),
            'sequence_length': 3,
            'n_features': 4
        }
        
        # Test data
        X_test = pd.DataFrame(np.random.randn(10, 12), index=range(100, 110))
        
        # Mock components
        mock_test_loader = MagicMock()
        mock_create_dataloader.return_value = mock_test_loader
        
        mock_predictions = np.array([0.1, 0.3, -0.2, 0.5, 0.8, -0.1, 0.4, 0.2, -0.3, 0.6])
        mock_predict_pytorch_model.return_value = mock_predictions
        
        # Execute
        result = model.predict(X_test)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)
        assert list(result.columns) == ['prediction']
        assert list(result.index) == list(range(100, 110))
        np.testing.assert_array_equal(result['prediction'].values, mock_predictions)
        
        # Check method calls
        dataloader_call = mock_create_dataloader.call_args
        assert dataloader_call[1]['batch_size'] == model.batch_size
        assert dataloader_call[1]['y'] is None
        assert dataloader_call[1]['shuffle'] is False
        mock_predict_pytorch_model.assert_called_once_with(
            model.model, mock_test_loader, model.device
        )
    
    def test_predict_reconstruction_failure(self):
        """Test prediction failure due to sequence reconstruction error."""
        model = LSTMTimeSeries()
        model.is_fitted = True
        model.sequence_info = {
            'original_shape': (50, 3, 4),
            'sequence_length': 3,
            'n_features': 4
        }
        
        # Wrong feature count: expects 3*4=12, providing 15
        X_test = pd.DataFrame(np.random.randn(10, 15))
        
        with pytest.raises(ValueError):
            model.predict(X_test)


class TestLSTMTimeSeriesModelInfo:
    """Test model information retrieval."""
    
    def test_get_model_info_not_fitted(self):
        """Test model info when not fitted."""
        model = LSTMTimeSeries()
        
        info = model.get_model_info()
        
        assert info == {"status": "not_fitted"}
    
    @patch('src.models.custom.pytorch_utils.count_parameters')
    def test_get_model_info_fitted(self, mock_count_parameters):
        """Test model info when fitted."""
        # Setup fitted model
        model = LSTMTimeSeries(
            hidden_dim=64, 
            num_layers=2, 
            dropout=0.2, 
            bidirectional=True,
            learning_rate=0.01
        )
        model.is_fitted = True
        model.model = MagicMock()
        model.sequence_info = {
            'original_shape': (100, 5, 6),
            'sequence_length': 5,
            'n_features': 6
        }
        model.device = torch.device('cuda:0')
        
        mock_count_parameters.return_value = {'total': 1024, 'trainable': 1024}
        
        # Execute
        info = model.get_model_info()
        
        # Assertions
        assert info['status'] == 'fitted'
        assert info['architecture'] == 'LSTM'
        
        expected_hyperparams = {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': True,
            'learning_rate': 0.01
        }
        assert info['hyperparameters'] == expected_hyperparams
        
        assert info['sequence_info'] == model.sequence_info
        assert info['model_parameters'] == {'total': 1024, 'trainable': 1024}
        assert info['device'] == 'cuda:0'


class TestLSTMTimeSeriesIntegration:
    """Test LSTM TimeSeries integration scenarios."""
    
    @patch('src.models.custom.lstm_timeseries.train_pytorch_model')
    @patch('src.models.custom.lstm_timeseries.predict_with_pytorch_model')
    @patch('src.models.custom.lstm_timeseries.create_dataloader')
    @patch('src.models.custom.lstm_timeseries.set_seed')
    def test_full_pipeline_integration(self, mock_set_seed, mock_create_dataloader, 
                                      mock_predict_pytorch_model, mock_train_pytorch_model):
        """Test complete fit-predict pipeline."""
        model = LSTMTimeSeries(hidden_dim=32, epochs=5, batch_size=8)
        
        # Training data
        X_train = pd.DataFrame(np.random.randn(20, 12))  # 20 samples, 3*4 features
        y_train = pd.Series(np.random.randn(20))
        train_kwargs = {'original_sequence_shape': (20, 3, 4)}
        
        # Prediction data
        X_test = pd.DataFrame(np.random.randn(5, 12))
        
        # Mock training
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_create_dataloader.side_effect = [mock_train_loader, mock_val_loader, mock_test_loader]
        
        mock_train_pytorch_model.return_value = {
            'train_loss': [0.8, 0.5, 0.3, 0.2, 0.1],
            'val_loss': [0.9, 0.6, 0.4, 0.3, 0.2],
            'best_epoch': 4
        }
        
        mock_predictions = np.array([0.1, 0.3, -0.2, 0.5, 0.8])
        mock_predict_pytorch_model.return_value = mock_predictions
        
        # Execute complete pipeline
        model.fit(X_train, y_train, **train_kwargs)
        predictions = model.predict(X_test)
        
        # Assertions
        assert model.is_fitted is True
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape == (5, 1)
        assert list(predictions.columns) == ['prediction']
        
        # Check all method calls were made
        mock_set_seed.assert_called_once_with(42)
        assert mock_create_dataloader.call_count == 3  # train, val, test loaders
        mock_train_pytorch_model.assert_called_once()
        mock_predict_pytorch_model.assert_called_once()


class TestLSTMTimeSeriesEdgeCases:
    """Test LSTM TimeSeries edge cases and error handling."""
    
    def test_empty_dataframe_error(self):
        """Test error handling for empty DataFrame."""
        model = LSTMTimeSeries()
        
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)
        
        with pytest.raises((ValueError, IndexError)):
            model.fit(X_empty, y_empty)
    
    def test_single_sample_edge_case(self):
        """Test handling of single sample."""
        model = LSTMTimeSeries()
        
        X_single = pd.DataFrame(np.random.randn(1, 12))
        y_single = pd.Series([1.0])
        kwargs = {'original_sequence_shape': (1, 3, 4)}
        
        # Should handle single sample gracefully (likely skip validation split)
        with patch('src.models.custom.lstm_timeseries.train_pytorch_model') as mock_train:
            with patch('src.models.custom.lstm_timeseries.create_dataloader') as mock_dataloader:
                mock_dataloader.return_value = MagicMock()
                mock_train.return_value = {'train_loss': [0.5], 'val_loss': [], 'best_epoch': 0}
                
                result = model.fit(X_single, y_single, **kwargs)
                assert result is model
                assert model.is_fitted is True
    
    def test_extreme_hyperparameters(self):
        """Test model with extreme hyperparameters."""
        # Very large model
        model_large = LSTMTimeSeries(
            hidden_dim=512,
            num_layers=10,
            dropout=0.8,
            bidirectional=True
        )
        
        # Very small model
        model_small = LSTMTimeSeries(
            hidden_dim=1,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )
        
        # Both should initialize without errors
        assert model_large.hidden_dim == 512
        assert model_small.hidden_dim == 1
        
        # Test building models with extreme params
        large_net = model_large._build_lstm_model(input_size=10)
        small_net = model_small._build_lstm_model(input_size=10)
        
        assert isinstance(large_net, nn.Module)
        assert isinstance(small_net, nn.Module)
    
    def test_inconsistent_sequence_shapes(self):
        """Test handling of inconsistent sequence shapes between fit and predict."""
        model = LSTMTimeSeries()
        
        # Fit with one shape
        X_fit = pd.DataFrame(np.random.randn(10, 12))  # 3*4 = 12
        y_fit = pd.Series(np.random.randn(10))
        fit_kwargs = {'original_sequence_shape': (10, 3, 4)}
        
        with patch('src.models.custom.lstm_timeseries.train_pytorch_model') as mock_train:
            with patch('src.models.custom.lstm_timeseries.create_dataloader'):
                mock_train.return_value = {'train_loss': [0.5], 'val_loss': [], 'best_epoch': 0}
                model.fit(X_fit, y_fit, **fit_kwargs)
        
        # Try to predict with different shape
        X_pred = pd.DataFrame(np.random.randn(5, 15))  # Different feature count
        
        with pytest.raises(ValueError):
            model.predict(X_pred)


class TestLSTMTimeSeriesPerformance:
    """Test LSTM TimeSeries performance characteristics."""
    
    def test_large_dataset_handling(self):
        """Test model can handle reasonably large datasets."""
        model = LSTMTimeSeries(batch_size=64, epochs=1)  # Small epochs for testing
        
        # Create larger dataset: 1000 samples
        X_large = pd.DataFrame(np.random.randn(1000, 60))  # 10 timesteps, 6 features
        y_large = pd.Series(np.random.randn(1000))
        kwargs = {'original_sequence_shape': (1000, 10, 6)}
        
        # Mock training to avoid actual computation
        with patch('src.models.custom.lstm_timeseries.train_pytorch_model') as mock_train:
            with patch('src.models.custom.lstm_timeseries.create_dataloader') as mock_dataloader:
                mock_dataloader.return_value = MagicMock()
                mock_train.return_value = {'train_loss': [0.5], 'val_loss': [0.6], 'best_epoch': 0}
                
                result = model.fit(X_large, y_large, **kwargs)
                assert result is model
                assert model.is_fitted is True
                
                # Check that batch processing was configured correctly  
                assert mock_dataloader.call_count >= 2  # At least train and val calls
                # Check batch size was passed to training components
                train_call = mock_dataloader.call_args_list[0]
                if len(train_call) > 1 and 'batch_size' in train_call[1]:
                    assert train_call[1]['batch_size'] == 64
    
    def test_memory_efficient_prediction(self):
        """Test memory-efficient prediction with batching."""
        model = LSTMTimeSeries(batch_size=32)
        model.is_fitted = True
        model.model = MagicMock()
        model.sequence_info = {'original_shape': (100, 5, 4), 'sequence_length': 5, 'n_features': 4}
        
        # Large prediction dataset
        X_pred = pd.DataFrame(np.random.randn(200, 20))  # 200 samples, 5*4 features
        
        with patch('src.models.custom.lstm_timeseries.create_dataloader') as mock_dataloader:
            with patch('src.models.custom.lstm_timeseries.predict_with_pytorch_model') as mock_predict:
                mock_dataloader.return_value = MagicMock()
                mock_predict.return_value = np.random.randn(200)
                
                result = model.predict(X_pred)
                
                assert result.shape == (200, 1)
                
                # Check batching was used
                dataloader_call = mock_dataloader.call_args
                assert dataloader_call[1]['batch_size'] == 32
                assert dataloader_call[1]['shuffle'] is False  # Important for time series