"""
LSTM TimeSeries model comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/models/custom/lstm_timeseries.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, Mock, MagicMock

from src.models.custom.lstm_timeseries import LSTMTimeSeries


class TestLSTMTimeSeriesInitialization:
    """LSTM TimeSeries 초기화 테스트 - Context 클래스 기반"""

    def test_lstm_timeseries_default_initialization(self, component_test_context):
        """기본 하이퍼파라미터로 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                # Verify default hyperparameters
                assert model.hidden_dim == 64
                assert model.num_layers == 2
                assert model.dropout == 0.2
                assert model.epochs == 100
                assert model.batch_size == 32
                assert model.learning_rate == 0.001
                assert model.early_stopping_patience == 10
                assert model.bidirectional is False

                # Verify internal state
                assert model.device == 'cpu'
                assert model.model is None
                assert model.is_fitted is False
                assert model.sequence_info is None
                assert model.handles_own_preprocessing is True

    def test_lstm_timeseries_custom_hyperparameters(self, component_test_context):
        """커스텀 하이퍼파라미터로 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cuda:0'):
                custom_params = {
                    'hidden_dim': 128,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'epochs': 50,
                    'batch_size': 64,
                    'learning_rate': 0.01,
                    'early_stopping_patience': 5,
                    'bidirectional': True
                }

                model = LSTMTimeSeries(**custom_params)

                # Verify custom hyperparameters
                assert model.hidden_dim == 128
                assert model.num_layers == 3
                assert model.dropout == 0.3
                assert model.epochs == 50
                assert model.batch_size == 64
                assert model.learning_rate == 0.01
                assert model.early_stopping_patience == 5
                assert model.bidirectional is True
                assert model.device == 'cuda:0'

    def test_lstm_timeseries_extra_kwargs_handling(self, component_test_context):
        """추가 키워드 인자 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                # Extra kwargs should not cause errors
                model = LSTMTimeSeries(hidden_dim=32, extra_param='ignored')

                assert model.hidden_dim == 32
                # extra_param should be ignored gracefully


class TestLSTMTimeSeriesSequenceInfoExtraction:
    """시퀀스 정보 추출 테스트"""

    def test_extract_sequence_info_from_metadata(self, component_test_context):
        """메타데이터에서 시퀀스 정보 추출 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                # Mock DataFrame
                X = pd.DataFrame({'col1': [1, 2, 3]})

                # Test with metadata
                kwargs = {'original_sequence_shape': (100, 10, 5)}
                seq_info = model._extract_sequence_info(X, kwargs)

                assert seq_info['original_shape'] == (100, 10, 5)
                assert seq_info['sequence_length'] == 10
                assert seq_info['n_features'] == 5
                assert seq_info['from_datahandler'] is True

    def test_infer_sequence_info_from_dataframe_with_seq_pattern(self, component_test_context):
        """DataFrame 컬럼명 패턴에서 시퀀스 정보 추론 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                # Create DataFrame with seq pattern
                columns = ['seq0_feat0', 'seq0_feat1', 'seq1_feat0', 'seq1_feat1', 'seq2_feat0', 'seq2_feat1']
                X = pd.DataFrame(np.zeros((10, 6)), columns=columns)

                seq_info = model._infer_sequence_info_from_dataframe(X)

                assert seq_info['original_shape'] == (10, 3, 2)  # (n_samples, seq_len=3, n_features=2)
                assert seq_info['sequence_length'] == 3
                assert seq_info['n_features'] == 2
                assert seq_info['from_datahandler'] is False

    def test_infer_sequence_info_from_dataframe_fallback(self, component_test_context):
        """DataFrame 패턴 추론 실패 시 기본값 사용 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                # Create DataFrame without seq pattern
                X = pd.DataFrame(np.zeros((5, 20)))  # 20 columns

                seq_info = model._infer_sequence_info_from_dataframe(X)

                assert seq_info['original_shape'] == (5, 10, 2)  # Default: seq_len=10, n_features=20//10=2
                assert seq_info['sequence_length'] == 10
                assert seq_info['n_features'] == 2
                assert seq_info['from_datahandler'] is False

    def test_extract_sequence_info_no_metadata(self, component_test_context):
        """메타데이터 없을 때 DataFrame 추론 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                # DataFrame with seq pattern
                columns = ['seq0_feat0', 'seq0_feat1', 'seq1_feat0', 'seq1_feat1']
                X = pd.DataFrame(np.zeros((3, 4)), columns=columns)

                seq_info = model._extract_sequence_info(X, {})  # Empty kwargs

                assert seq_info['sequence_length'] == 2
                assert seq_info['n_features'] == 2
                assert seq_info['from_datahandler'] is False


class TestLSTMTimeSeries3DReconstruction:
    """3D 시퀀스 복원 테스트"""

    def test_reconstruct_3d_sequences_success(self, component_test_context):
        """성공적인 3D 시퀀스 복원 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                # Create flattened sequence data
                X = pd.DataFrame(np.arange(24).reshape(2, 12))  # 2 samples, 12 features
                seq_info = {
                    'original_shape': (2, 4, 3),  # 2 samples, seq_len=4, n_features=3
                    'sequence_length': 4,
                    'n_features': 3
                }

                X_3d = model._reconstruct_3d_sequences(X, seq_info)

                assert X_3d.shape == (2, 4, 3)
                assert X_3d.dtype == np.float32
                # Verify data is correctly reshaped
                assert X_3d[0, 0, 0] == 0
                assert X_3d[0, 1, 0] == 3
                assert X_3d[1, 0, 0] == 12

    def test_reconstruct_3d_sequences_different_sample_count(self, component_test_context):
        """다른 샘플 개수로 3D 시퀀스 복원 테스트 (예측 시)"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                # Prediction data with different sample count than training
                X = pd.DataFrame(np.arange(18).reshape(3, 6))  # 3 samples, 6 features
                seq_info = {
                    'original_shape': (10, 2, 3),  # Training had 10 samples, but prediction has 3
                    'sequence_length': 2,
                    'n_features': 3
                }

                X_3d = model._reconstruct_3d_sequences(X, seq_info)

                assert X_3d.shape == (3, 2, 3)  # Uses actual sample count, not stored count

    def test_reconstruct_3d_sequences_feature_mismatch_error(self, component_test_context):
        """특성 개수 불일치 시 에러 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                X = pd.DataFrame(np.arange(10).reshape(2, 5))  # 2 samples, 5 features
                seq_info = {
                    'original_shape': (2, 2, 3),  # Expects 2*3=6 features, but got 5
                    'sequence_length': 2,
                    'n_features': 3
                }

                with pytest.raises(ValueError) as exc_info:
                    model._reconstruct_3d_sequences(X, seq_info)

                assert "입력 데이터의 특성 수가 맞지 않습니다" in str(exc_info.value)
                assert "기대값: 6" in str(exc_info.value)
                assert "실제값: 5" in str(exc_info.value)


class TestLSTMTimeSeriesDataValidation:
    """데이터 검증 테스트"""

    def test_validate_sequence_data_success(self, component_test_context):
        """성공적인 시퀀스 데이터 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                X = np.random.random((20, 5, 3))  # Valid 3D data
                y = pd.Series(np.random.random(20))

                # Should not raise any exception
                model._validate_sequence_data(X, y)

    def test_validate_sequence_data_not_3d_error(self, component_test_context):
        """3D가 아닌 데이터 검증 에러 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                X = np.random.random((20, 15))  # 2D data instead of 3D
                y = pd.Series(np.random.random(20))

                with pytest.raises(ValueError) as exc_info:
                    model._validate_sequence_data(X, y)

                assert "3D 시퀀스 데이터가 필요합니다" in str(exc_info.value)

    def test_validate_sequence_data_length_mismatch_error(self, component_test_context):
        """X와 y 길이 불일치 에러 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                X = np.random.random((20, 5, 3))
                y = pd.Series(np.random.random(15))  # Different length

                with pytest.raises(ValueError) as exc_info:
                    model._validate_sequence_data(X, y)

                assert "X와 y의 길이가 다릅니다: 20 vs 15" in str(exc_info.value)

    def test_validate_sequence_data_small_dataset_warning(self, component_test_context):
        """작은 데이터셋에 대한 경고 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                X = np.random.random((5, 3, 2))  # Small dataset
                y = pd.Series(np.random.random(5))

                # Should not raise exception but may log warning
                model._validate_sequence_data(X, y)


class TestLSTMTimeSeriesTimeSplit:
    """시간 기반 분할 테스트"""

    def test_time_based_split_normal(self, component_test_context):
        """정상적인 시간 기반 분할 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                X = np.random.random((20, 5, 3))
                y = pd.Series(np.random.random(20))

                X_train, X_val, y_train, y_val = model._time_based_split(X, y, val_ratio=0.2)

                # Check split sizes (80% train, 20% validation)
                assert len(X_train) == 16
                assert len(X_val) == 4
                assert len(y_train) == 16
                assert len(y_val) == 4

                # Check temporal order is preserved
                np.testing.assert_array_equal(X_train, X[:16])
                np.testing.assert_array_equal(X_val, X[16:])

    def test_time_based_split_custom_ratio(self, component_test_context):
        """커스텀 분할 비율 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                X = np.random.random((10, 5, 3))
                y = pd.Series(np.random.random(10))

                X_train, X_val, y_train, y_val = model._time_based_split(X, y, val_ratio=0.3)

                # Check split sizes (70% train, 30% validation)
                assert len(X_train) == 7
                assert len(X_val) == 3

    def test_time_based_split_small_dataset_no_validation(self, component_test_context):
        """작은 데이터셋에서 validation 생략 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                X = np.random.random((3, 5, 2))  # Very small dataset
                y = pd.Series(np.random.random(3))

                X_train, X_val, y_train, y_val = model._time_based_split(X, y)

                # Should skip validation split
                assert len(X_train) == 3
                assert X_val is None
                assert len(y_train) == 3
                assert y_val is None


class TestLSTMTimeSeriesArchitecture:
    """LSTM 아키텍처 테스트"""

    def test_build_lstm_model_unidirectional(self, component_test_context):
        """단방향 LSTM 모델 구축 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries(hidden_dim=32, num_layers=2, dropout=0.1, bidirectional=False)

                lstm_model = model._build_lstm_model(input_size=5)

                # Check model structure
                assert isinstance(lstm_model, nn.Module)
                assert hasattr(lstm_model, 'lstm')
                assert hasattr(lstm_model, 'fc')
                assert hasattr(lstm_model, 'dropout')

                # Check LSTM configuration
                assert lstm_model.lstm.input_size == 5
                assert lstm_model.lstm.hidden_size == 32
                assert lstm_model.lstm.num_layers == 2
                assert lstm_model.lstm.bidirectional is False

                # Check Linear layer (unidirectional: hidden_dim -> 1)
                assert lstm_model.fc.in_features == 32
                assert lstm_model.fc.out_features == 1

    def test_build_lstm_model_bidirectional(self, component_test_context):
        """양방향 LSTM 모델 구축 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries(hidden_dim=64, num_layers=3, bidirectional=True)

                lstm_model = model._build_lstm_model(input_size=10)

                # Check LSTM configuration
                assert lstm_model.lstm.bidirectional is True

                # Check Linear layer (bidirectional: hidden_dim * 2 -> 1)
                assert lstm_model.fc.in_features == 128  # 64 * 2

    def test_build_lstm_model_single_layer_no_dropout(self, component_test_context):
        """단일 레이어 LSTM의 dropout=0 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries(num_layers=1, dropout=0.3)

                lstm_model = model._build_lstm_model(input_size=3)

                # Single layer LSTM should have dropout=0 in LSTM
                # (PyTorch requirement: dropout is ignored for single layer)
                assert lstm_model.lstm.num_layers == 1

    def test_lstm_model_forward_pass(self, component_test_context):
        """LSTM 모델 순전파 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries(hidden_dim=16, num_layers=1)
                lstm_model = model._build_lstm_model(input_size=3)

                # Test forward pass
                batch_size, seq_len, n_features = 2, 5, 3
                x = torch.randn(batch_size, seq_len, n_features)

                with torch.no_grad():
                    output = lstm_model(x)

                # Should output (batch_size, 1) for regression
                assert output.shape == (batch_size, 1)


class TestLSTMTimeSeriesFit:
    """LSTM 학습 테스트"""

    def test_fit_complete_workflow(self, component_test_context):
        """완전한 학습 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'), \
                 patch('src.models.custom.lstm_timeseries.set_seed'), \
                 patch('src.models.custom.lstm_timeseries.create_dataloader') as mock_dataloader, \
                 patch('src.models.custom.lstm_timeseries.train_pytorch_model') as mock_train:

                # Mock training history
                mock_train.return_value = {
                    'best_epoch': 20,
                    'train_loss': [1.0, 0.8, 0.6, 0.5],
                    'val_loss': [1.2, 0.9, 0.7, 0.6]
                }

                # Create model and data
                model = LSTMTimeSeries(epochs=50, batch_size=16)

                # Create DataFrame with sequence pattern
                columns = ['seq0_feat0', 'seq0_feat1', 'seq1_feat0', 'seq1_feat1']
                X = pd.DataFrame(np.random.random((10, 4)), columns=columns)
                y = pd.Series(np.random.random(10))

                # Test fit
                result = model.fit(X, y)

                # Verify workflow
                assert result is model
                assert model.is_fitted is True
                assert model.model is not None
                assert model.sequence_info is not None

                # Verify sequence info was extracted
                assert model.sequence_info['sequence_length'] == 2
                assert model.sequence_info['n_features'] == 2

                # Verify training was called
                mock_train.assert_called_once()

    def test_fit_with_metadata(self, component_test_context):
        """메타데이터와 함께 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'), \
                 patch('src.models.custom.lstm_timeseries.set_seed'), \
                 patch('src.models.custom.lstm_timeseries.create_dataloader'), \
                 patch('src.models.custom.lstm_timeseries.train_pytorch_model') as mock_train:

                mock_train.return_value = {'best_epoch': 10, 'train_loss': [0.5], 'val_loss': [0.6]}

                model = LSTMTimeSeries()
                X = pd.DataFrame(np.random.random((5, 15)))  # 15 features
                y = pd.Series(np.random.random(5))

                # Pass metadata
                kwargs = {'original_sequence_shape': (5, 5, 3)}
                model.fit(X, y, **kwargs)

                # Should use metadata instead of inferring
                assert model.sequence_info['from_datahandler'] is True
                assert model.sequence_info['sequence_length'] == 5
                assert model.sequence_info['n_features'] == 3

    def test_fit_training_parameters_passed_correctly(self, component_test_context):
        """학습 파라미터가 올바르게 전달되는지 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'), \
                 patch('src.models.custom.lstm_timeseries.set_seed'), \
                 patch('src.models.custom.lstm_timeseries.create_dataloader'), \
                 patch('src.models.custom.lstm_timeseries.train_pytorch_model') as mock_train:

                mock_train.return_value = {'best_epoch': 5, 'train_loss': [0.3], 'val_loss': None}

                # Custom training parameters
                model = LSTMTimeSeries(
                    epochs=25,
                    learning_rate=0.005,
                    early_stopping_patience=3
                )

                X = pd.DataFrame(np.random.random((8, 6)))
                y = pd.Series(np.random.random(8))
                kwargs = {'original_sequence_shape': (8, 2, 3)}

                model.fit(X, y, **kwargs)

                # Check that training was called with correct parameters
                call_args = mock_train.call_args[1]
                assert call_args['epochs'] == 25
                assert call_args['learning_rate'] == 0.005
                assert call_args['early_stopping_patience'] == 3
                assert call_args['device'] == 'cpu'


class TestLSTMTimeSeriesPredict:
    """LSTM 예측 테스트"""

    def test_predict_before_fit_error(self, component_test_context):
        """학습 전 예측 시도 시 에러 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()
                X = pd.DataFrame(np.random.random((3, 6)))

                with pytest.raises(RuntimeError) as exc_info:
                    model.predict(X)

                assert "모델이 학습되지 않았습니다" in str(exc_info.value)
                assert "fit()을 먼저 호출하세요" in str(exc_info.value)

    def test_predict_complete_workflow(self, component_test_context):
        """완전한 예측 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'), \
                 patch('src.models.custom.lstm_timeseries.create_dataloader') as mock_dataloader, \
                 patch('src.models.custom.lstm_timeseries.predict_with_pytorch_model') as mock_predict:

                # Mock prediction results
                mock_predict.return_value = np.array([1.5, 2.3, 0.8])

                # Create and "fit" model (mock fitted state)
                model = LSTMTimeSeries()
                model.is_fitted = True
                model.model = Mock()  # Mock PyTorch model
                model.sequence_info = {
                    'original_shape': (10, 3, 2),
                    'sequence_length': 3,
                    'n_features': 2,
                    'from_datahandler': True
                }

                # Test prediction
                X = pd.DataFrame(np.random.random((3, 6)), index=[10, 11, 12])  # 3 samples, 6 features
                predictions = model.predict(X)

                # Verify prediction workflow
                mock_predict.assert_called_once()

                # Verify output format
                assert isinstance(predictions, pd.DataFrame)
                assert list(predictions.index) == [10, 11, 12]
                assert predictions.columns.tolist() == ['prediction']
                np.testing.assert_array_equal(predictions['prediction'].values, [1.5, 2.3, 0.8])

    def test_predict_sequence_reconstruction(self, component_test_context):
        """예측 시 시퀀스 복원 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'), \
                 patch('src.models.custom.lstm_timeseries.create_dataloader') as mock_dataloader, \
                 patch('src.models.custom.lstm_timeseries.predict_with_pytorch_model') as mock_predict:

                mock_predict.return_value = np.array([0.5])

                # Setup fitted model
                model = LSTMTimeSeries()
                model.is_fitted = True
                model.model = Mock()
                model.sequence_info = {
                    'original_shape': (100, 4, 3),  # Training shape
                    'sequence_length': 4,
                    'n_features': 3,
                    'from_datahandler': True
                }

                # Prediction data (different sample count than training)
                X = pd.DataFrame(np.random.random((1, 12)))  # 1 sample, 12 features (4*3)

                predictions = model.predict(X)

                # Check that dataloader was called (indicating successful 3D reconstruction)
                mock_dataloader.assert_called_once()
                dataloader_args = mock_dataloader.call_args
                X_3d_used = dataloader_args[0][0]
                assert X_3d_used.shape == (1, 4, 3)  # Correctly reconstructed


class TestLSTMTimeSeriesModelInfo:
    """모델 정보 테스트"""

    def test_get_model_info_not_fitted(self, component_test_context):
        """학습되지 않은 모델 정보 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                info = model.get_model_info()

                assert info == {"status": "not_fitted"}

    def test_get_model_info_fitted(self, component_test_context):
        """학습된 모델 정보 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cuda:0'), \
                 patch('src.models.custom.lstm_timeseries.count_parameters') as mock_count_params:

                # Mock parameter counting
                mock_count_params.return_value = {
                    'total_params': 1000,
                    'trainable_params': 1000,
                    'non_trainable_params': 0
                }

                # Setup fitted model
                model = LSTMTimeSeries(hidden_dim=128, num_layers=3, dropout=0.3, bidirectional=True, learning_rate=0.01)
                model.is_fitted = True
                model.model = Mock()
                model.device = 'cuda:0'
                model.sequence_info = {
                    'original_shape': (50, 10, 5),
                    'sequence_length': 10,
                    'n_features': 5
                }

                info = model.get_model_info()

                # Verify complete model info
                assert info['status'] == 'fitted'
                assert info['architecture'] == 'LSTM'
                assert info['hyperparameters']['hidden_dim'] == 128
                assert info['hyperparameters']['num_layers'] == 3
                assert info['hyperparameters']['dropout'] == 0.3
                assert info['hyperparameters']['bidirectional'] is True
                assert info['hyperparameters']['learning_rate'] == 0.01
                assert info['sequence_info'] == model.sequence_info
                assert info['model_parameters'] == mock_count_params.return_value
                assert info['device'] == 'cuda:0'


class TestLSTMTimeSeriesIntegration:
    """LSTM TimeSeries 통합 테스트"""

    def test_complete_lstm_workflow_with_mocked_pytorch(self, component_test_context):
        """PyTorch 모킹된 완전한 LSTM 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'), \
                 patch('src.models.custom.lstm_timeseries.set_seed'), \
                 patch('src.models.custom.lstm_timeseries.create_dataloader') as mock_dataloader, \
                 patch('src.models.custom.lstm_timeseries.train_pytorch_model') as mock_train, \
                 patch('src.models.custom.lstm_timeseries.predict_with_pytorch_model') as mock_predict:

                # Setup mocks
                mock_train.return_value = {
                    'best_epoch': 15,
                    'train_loss': [1.5, 1.2, 0.9, 0.7],
                    'val_loss': [1.6, 1.3, 1.0, 0.8]
                }
                mock_predict.return_value = np.array([2.1, 1.8, 2.5])

                # Create model with custom hyperparameters
                model = LSTMTimeSeries(
                    hidden_dim=64,
                    num_layers=2,
                    epochs=30,
                    batch_size=16,
                    bidirectional=True
                )

                # Training data
                X_train = pd.DataFrame(np.random.random((15, 18)))  # 15 samples, 18 features
                y_train = pd.Series(np.random.random(15))
                kwargs = {'original_sequence_shape': (15, 6, 3)}  # seq_len=6, n_features=3

                # Test complete fit workflow
                model.fit(X_train, y_train, **kwargs)

                # Verify fit completed successfully
                assert model.is_fitted is True
                assert model.sequence_info['sequence_length'] == 6
                assert model.sequence_info['n_features'] == 3
                mock_train.assert_called_once()

                # Test prediction
                X_test = pd.DataFrame(np.random.random((3, 18)), index=[100, 101, 102])
                predictions = model.predict(X_test)

                # Verify prediction results
                assert isinstance(predictions, pd.DataFrame)
                assert list(predictions.index) == [100, 101, 102]
                assert predictions.columns.tolist() == ['prediction']
                np.testing.assert_array_equal(predictions['prediction'].values, [2.1, 1.8, 2.5])

                # Test model info
                with patch('src.models.custom.lstm_timeseries.count_parameters', return_value={'total_params': 5000}):
                    info = model.get_model_info()
                    assert info['status'] == 'fitted'
                    assert info['architecture'] == 'LSTM'
                    assert info['hyperparameters']['bidirectional'] is True

    def test_lstm_error_resilience_workflow(self, component_test_context):
        """LSTM 오류 복원력 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.lstm_timeseries.get_device', return_value='cpu'):
                model = LSTMTimeSeries()

                # Test various error scenarios

                # 1. Unfitted model prediction
                X = pd.DataFrame(np.random.random((2, 6)))
                with pytest.raises(RuntimeError):
                    model.predict(X)

                # 2. Invalid 3D reconstruction (feature mismatch)
                model.sequence_info = {'original_shape': (10, 3, 4), 'sequence_length': 3, 'n_features': 4}
                X_wrong = pd.DataFrame(np.random.random((2, 8)))  # Should be 12 features (3*4) but got 8

                with pytest.raises(ValueError) as exc_info:
                    model._reconstruct_3d_sequences(X_wrong, model.sequence_info)
                assert "입력 데이터의 특성 수가 맞지 않습니다" in str(exc_info.value)

                # 3. Invalid sequence validation
                X_invalid = np.random.random((5, 3))  # 2D instead of 3D
                y = pd.Series(np.random.random(5))
                with pytest.raises(ValueError) as exc_info:
                    model._validate_sequence_data(X_invalid, y)
                assert "3D 시퀀스 데이터가 필요합니다" in str(exc_info.value)