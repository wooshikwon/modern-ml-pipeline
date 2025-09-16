"""
PyTorch utilities comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/models/custom/pytorch_utils.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, Mock, MagicMock

from src.models.custom.pytorch_utils import (
    get_device,
    create_dataloader,
    train_pytorch_model,
    validate_pytorch_model,
    predict_with_pytorch_model,
    get_model_device,
    count_parameters,
    set_seed
)


class TestDeviceSelection:
    """디바이스 선택 기능 테스트 - Context 클래스 기반"""

    def test_get_device_cuda_available(self, component_test_context):
        """CUDA 사용 가능 시 GPU 선택 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.get_device_name', return_value='Tesla V100'):

                device = get_device()

                assert device.type == 'cuda'
                assert isinstance(device, torch.device)

    def test_get_device_cuda_not_available(self, component_test_context):
        """CUDA 사용 불가능 시 CPU 선택 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('torch.cuda.is_available', return_value=False):

                device = get_device()

                assert device.type == 'cpu'
                assert isinstance(device, torch.device)

    def test_get_device_logging(self, component_test_context):
        """디바이스 선택 시 로깅 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.get_device_name', return_value='RTX 3080'):

                device = get_device()

                # Should log GPU information
                assert device.type == 'cuda'


class TestDataLoaderCreation:
    """DataLoader 생성 기능 테스트"""

    def test_create_dataloader_with_pandas_dataframe_and_series(self, component_test_context):
        """pandas DataFrame과 Series로 DataLoader 생성 테스트"""
        with component_test_context.classification_stack() as ctx:
            X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
            y = pd.Series([0, 1, 0, 1])

            dataloader = create_dataloader(X, y, batch_size=2, shuffle=False)

            assert isinstance(dataloader, DataLoader)
            assert dataloader.batch_size == 2
            assert len(dataloader.dataset) == 4

            # Check data conversion
            first_batch = next(iter(dataloader))
            assert len(first_batch) == 2  # X and y
            X_batch, y_batch = first_batch
            assert X_batch.shape == (2, 2)  # batch_size=2, features=2
            assert y_batch.shape == (2,)

    def test_create_dataloader_with_numpy_arrays(self, component_test_context):
        """numpy array로 DataLoader 생성 테스트"""
        with component_test_context.classification_stack() as ctx:
            X = np.random.random((10, 5))
            y = np.random.random(10)

            dataloader = create_dataloader(X, y, batch_size=3, shuffle=True)

            assert isinstance(dataloader, DataLoader)
            assert dataloader.batch_size == 3
            assert len(dataloader.dataset) == 10

    def test_create_dataloader_without_y(self, component_test_context):
        """y 없이 DataLoader 생성 테스트 (예측용)"""
        with component_test_context.classification_stack() as ctx:
            X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

            dataloader = create_dataloader(X, y=None, batch_size=2)

            assert isinstance(dataloader, DataLoader)
            assert len(dataloader.dataset) == 3

            # Check that it only returns X
            first_batch = next(iter(dataloader))
            assert len(first_batch) == 1  # Only X, no y
            X_batch = first_batch[0]
            assert X_batch.shape == (2, 2)  # batch_size=2, features=2

    def test_create_dataloader_custom_batch_size_and_shuffle(self, component_test_context):
        """커스텀 배치 크기와 shuffle 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            X = np.ones((8, 3))
            y = np.ones(8)

            # Test custom batch size
            dataloader = create_dataloader(X, y, batch_size=4, shuffle=False)
            assert dataloader.batch_size == 4
            assert len(list(dataloader)) == 2  # 8 samples / 4 batch_size = 2 batches

            # Test shuffle parameter is preserved
            dataloader_shuffled = create_dataloader(X, y, batch_size=2, shuffle=True)
            assert dataloader_shuffled.batch_size == 2


class TestModelTraining:
    """모델 학습 기능 테스트"""

    def create_simple_model(self):
        """테스트용 간단한 모델 생성"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 1)

            def forward(self, x):
                return self.linear(x)

        return SimpleModel()

    def test_train_pytorch_model_basic_training(self, component_test_context):
        """기본 모델 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_simple_model()

            # Create simple training data
            X = torch.randn(20, 3)
            y = torch.randn(20)
            train_loader = DataLoader(TensorDataset(X, y), batch_size=4)

            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')):
                history = train_pytorch_model(
                    model=model,
                    train_loader=train_loader,
                    epochs=5,
                    learning_rate=0.01,
                    log_interval=2
                )

            # Check training history
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert 'best_epoch' in history
            assert len(history['train_loss']) <= 5  # Could be less due to early stopping
            assert history['val_loss'] == []  # No validation loader provided
            assert history['best_epoch'] == 0  # Default when no validation

    def test_train_pytorch_model_with_validation(self, component_test_context):
        """검증 데이터와 함께 모델 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_simple_model()

            # Training and validation data
            X_train = torch.randn(16, 3)
            y_train = torch.randn(16)
            X_val = torch.randn(8, 3)
            y_val = torch.randn(8)

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=4)

            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')):
                history = train_pytorch_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=3,
                    early_stopping_patience=2
                )

            # Check validation history
            assert len(history['val_loss']) > 0
            assert len(history['train_loss']) == len(history['val_loss'])

    def test_train_pytorch_model_early_stopping(self, component_test_context):
        """조기 종료 기능 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_simple_model()

            X = torch.randn(12, 3)
            y = torch.randn(12)
            train_loader = DataLoader(TensorDataset(X, y), batch_size=4)
            val_loader = DataLoader(TensorDataset(X, y), batch_size=4)  # Same data for simplicity

            # Mock validation to return increasing loss (trigger early stopping)
            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')), \
                 patch('src.models.custom.pytorch_utils.validate_pytorch_model') as mock_validate:

                # Return increasing validation loss to trigger early stopping
                mock_validate.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0]

                history = train_pytorch_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=10,
                    early_stopping_patience=2
                )

            # Should stop early (before 10 epochs)
            assert len(history['train_loss']) < 10

    def test_train_pytorch_model_custom_criterion_and_optimizer(self, component_test_context):
        """커스텀 손실함수와 옵티마이저 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_simple_model()

            X = torch.randn(12, 3)
            y = torch.randn(12)
            train_loader = DataLoader(TensorDataset(X, y), batch_size=3)

            # Use custom criterion
            custom_criterion = nn.L1Loss()

            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')):
                history = train_pytorch_model(
                    model=model,
                    train_loader=train_loader,
                    epochs=2,
                    criterion=custom_criterion,
                    learning_rate=0.001
                )

            # Should complete without errors
            assert len(history['train_loss']) == 2

    def test_train_pytorch_model_unsupervised_learning(self, component_test_context):
        """비지도 학습 (y 없는 경우) 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Autoencoder-like model
            class AutoEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Linear(4, 2)
                    self.decoder = nn.Linear(2, 4)

                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded

            model = AutoEncoder()
            X = torch.randn(16, 4)
            train_loader = DataLoader(TensorDataset(X), batch_size=4)  # Only X, no y

            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')):
                history = train_pytorch_model(
                    model=model,
                    train_loader=train_loader,
                    epochs=3
                )

            # Should handle unsupervised case (compares output to input)
            assert len(history['train_loss']) == 3


class TestModelValidation:
    """모델 검증 기능 테스트"""

    def create_simple_model(self):
        """테스트용 간단한 모델 생성"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)

            def forward(self, x):
                return self.linear(x)

        return SimpleModel()

    def test_validate_pytorch_model_with_targets(self, component_test_context):
        """타겟이 있는 모델 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_simple_model()

            X = torch.randn(8, 2)
            y = torch.randn(8)
            val_loader = DataLoader(TensorDataset(X, y), batch_size=4)
            criterion = nn.MSELoss()
            device = torch.device('cpu')

            val_loss = validate_pytorch_model(model, val_loader, criterion, device)

            assert isinstance(val_loss, float)
            assert val_loss >= 0  # Loss should be non-negative

    def test_validate_pytorch_model_without_targets(self, component_test_context):
        """타겟이 없는 모델 검증 테스트 (비지도 학습)"""
        with component_test_context.classification_stack() as ctx:
            # Identity model for testing
            class IdentityModel(nn.Module):
                def forward(self, x):
                    return x

            model = IdentityModel()
            X = torch.randn(6, 3)
            val_loader = DataLoader(TensorDataset(X), batch_size=2)  # No y
            criterion = nn.MSELoss()
            device = torch.device('cpu')

            val_loss = validate_pytorch_model(model, val_loader, criterion, device)

            assert isinstance(val_loss, float)
            # For identity model, loss should be 0 (output == input)
            assert val_loss < 1e-6


class TestModelPrediction:
    """모델 예측 기능 테스트"""

    def create_predictable_model(self):
        """예측 가능한 테스트 모델 생성"""
        class PredictableModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Fixed weights for predictable output
                self.linear = nn.Linear(2, 1)
                with torch.no_grad():
                    self.linear.weight.fill_(1.0)
                    self.linear.bias.fill_(0.0)

            def forward(self, x):
                return self.linear(x)

        return PredictableModel()

    def test_predict_with_pytorch_model_basic(self, component_test_context):
        """기본 모델 예측 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_predictable_model()

            # Test data: [1, 1] should give output close to 2 (1*1 + 1*1 + 0 = 2)
            X = torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
            data_loader = DataLoader(TensorDataset(X), batch_size=2)

            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')):
                predictions = predict_with_pytorch_model(model, data_loader)

            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == 3

            # Check predictable outputs
            np.testing.assert_array_almost_equal(predictions, [2.0, 2.0, 3.0], decimal=5)

    def test_predict_with_pytorch_model_custom_device(self, component_test_context):
        """커스텀 디바이스로 예측 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_predictable_model()

            X = torch.tensor([[1.0, 2.0]])
            data_loader = DataLoader(TensorDataset(X), batch_size=1)

            # Test with explicitly provided device
            predictions = predict_with_pytorch_model(
                model, data_loader, device=torch.device('cpu')
            )

            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == 1

    def test_predict_with_pytorch_model_multiple_batches(self, component_test_context):
        """여러 배치를 통한 예측 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_predictable_model()

            # Large dataset to test multiple batches
            X = torch.ones(10, 2)  # All [1, 1] inputs
            data_loader = DataLoader(TensorDataset(X), batch_size=3)

            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')):
                predictions = predict_with_pytorch_model(model, data_loader)

            assert len(predictions) == 10
            # All predictions should be 2.0 (since all inputs are [1, 1])
            np.testing.assert_array_almost_equal(predictions, np.full(10, 2.0), decimal=5)


class TestModelUtilities:
    """모델 유틸리티 기능 테스트"""

    def create_test_model(self):
        """테스트용 모델 생성"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 3)  # 5*3 + 3 = 18 parameters
                self.linear2 = nn.Linear(3, 1)  # 3*1 + 1 = 4 parameters
                # Total: 22 parameters

            def forward(self, x):
                x = self.linear1(x)
                return self.linear2(x)

        return TestModel()

    def test_get_model_device_cpu(self, component_test_context):
        """CPU에 있는 모델의 디바이스 확인 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_test_model()
            device = get_model_device(model)

            assert device.type == 'cpu'

    def test_get_model_device_cuda(self, component_test_context):
        """GPU에 있는 모델의 디바이스 확인 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Skip this test if CUDA is not actually available
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available for testing")

            model = self.create_test_model()
            model.cuda()
            device = get_model_device(model)
            assert device.type == 'cuda'

    def test_count_parameters_all_trainable(self, component_test_context):
        """모든 파라미터가 학습 가능한 모델 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_test_model()
            param_info = count_parameters(model)

            assert 'total_parameters' in param_info
            assert 'trainable_parameters' in param_info
            assert 'non_trainable_parameters' in param_info

            # Check expected parameter counts
            assert param_info['total_parameters'] == 22  # 5*3 + 3 + 3*1 + 1
            assert param_info['trainable_parameters'] == 22
            assert param_info['non_trainable_parameters'] == 0

    def test_count_parameters_with_frozen_layers(self, component_test_context):
        """일부 레이어가 frozen된 모델 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = self.create_test_model()

            # Freeze first layer
            for param in model.linear1.parameters():
                param.requires_grad = False

            param_info = count_parameters(model)

            assert param_info['total_parameters'] == 22
            assert param_info['trainable_parameters'] == 4  # Only linear2: 3*1 + 1
            assert param_info['non_trainable_parameters'] == 18  # linear1: 5*3 + 3

    def test_count_parameters_empty_model(self, component_test_context):
        """파라미터가 없는 모델 테스트"""
        with component_test_context.classification_stack() as ctx:
            class EmptyModel(nn.Module):
                def forward(self, x):
                    return x

            model = EmptyModel()
            param_info = count_parameters(model)

            assert param_info['total_parameters'] == 0
            assert param_info['trainable_parameters'] == 0
            assert param_info['non_trainable_parameters'] == 0


class TestSeedSetting:
    """시드 설정 기능 테스트"""

    def test_set_seed_default_value(self, component_test_context):
        """기본 시드 값(42) 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('torch.manual_seed') as mock_manual_seed, \
                 patch('torch.cuda.manual_seed') as mock_cuda_seed, \
                 patch('torch.cuda.manual_seed_all') as mock_cuda_seed_all, \
                 patch('torch.backends.cudnn') as mock_cudnn:

                set_seed()

                # Check that all seed functions were called with default value (42)
                mock_manual_seed.assert_called_once_with(42)
                mock_cuda_seed.assert_called_once_with(42)
                mock_cuda_seed_all.assert_called_once_with(42)

                # Check CUDNN settings
                assert mock_cudnn.deterministic is True
                assert mock_cudnn.benchmark is False

    def test_set_seed_custom_value(self, component_test_context):
        """커스텀 시드 값 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            custom_seed = 12345

            with patch('torch.manual_seed') as mock_manual_seed, \
                 patch('torch.cuda.manual_seed') as mock_cuda_seed, \
                 patch('torch.cuda.manual_seed_all') as mock_cuda_seed_all, \
                 patch('torch.backends.cudnn') as mock_cudnn:

                set_seed(custom_seed)

                # Check that all seed functions were called with custom value
                mock_manual_seed.assert_called_once_with(custom_seed)
                mock_cuda_seed.assert_called_once_with(custom_seed)
                mock_cuda_seed_all.assert_called_once_with(custom_seed)

                # Check CUDNN deterministic settings
                assert mock_cudnn.deterministic is True
                assert mock_cudnn.benchmark is False

    def test_set_seed_reproducibility_effect(self, component_test_context):
        """시드 설정의 재현성 효과 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test that same seed produces same results
            set_seed(123)
            first_random = torch.randn(3)

            set_seed(123)  # Same seed
            second_random = torch.randn(3)

            # Should be identical
            torch.testing.assert_close(first_random, second_random)

    def test_set_seed_different_seeds_different_results(self, component_test_context):
        """다른 시드로 다른 결과 생성 확인 테스트"""
        with component_test_context.classification_stack() as ctx:
            set_seed(100)
            first_random = torch.randn(5)

            set_seed(200)  # Different seed
            second_random = torch.randn(5)

            # Should be different
            assert not torch.allclose(first_random, second_random)


class TestPyTorchUtilsIntegration:
    """PyTorch 유틸리티 통합 테스트"""

    def test_complete_training_prediction_workflow(self, component_test_context):
        """완전한 학습-예측 워크플로우 통합 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create simple regression model
            class RegressionModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(2, 1)

                def forward(self, x):
                    return self.linear(x)

            # Step 1: Set seed for reproducibility
            set_seed(42)

            # Step 2: Create model and data
            model = RegressionModel()
            X_train = pd.DataFrame({'x1': [1, 2, 3, 4], 'x2': [2, 4, 6, 8]})
            y_train = pd.Series([3, 6, 9, 12])  # y = x1 + x2

            X_test = pd.DataFrame({'x1': [5, 6], 'x2': [10, 12]})

            # Step 3: Create data loaders
            train_loader = create_dataloader(X_train, y_train, batch_size=2, shuffle=False)
            test_loader = create_dataloader(X_test, y=None, batch_size=2)

            # Step 4: Train model
            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')):
                history = train_pytorch_model(
                    model=model,
                    train_loader=train_loader,
                    epochs=10,
                    learning_rate=0.01
                )

            # Step 5: Check training completed
            assert len(history['train_loss']) <= 10
            assert model is not None

            # Step 6: Make predictions
            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')):
                predictions = predict_with_pytorch_model(model, test_loader)

            # Step 7: Verify predictions
            assert len(predictions) == 2
            assert isinstance(predictions, np.ndarray)

            # Step 8: Check model info
            param_info = count_parameters(model)
            assert param_info['total_parameters'] == 3  # 2 weights + 1 bias

            # Step 9: Check device
            device = get_model_device(model)
            assert device.type == 'cpu'

    def test_pytorch_utils_error_handling_integration(self, component_test_context):
        """PyTorch 유틸리티 오류 처리 통합 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test various error scenarios

            # 1. Empty data loader
            empty_X = torch.empty(0, 3)
            empty_loader = DataLoader(TensorDataset(empty_X), batch_size=1)

            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(3, 1)

                def forward(self, x):
                    return self.linear(x)

            model = SimpleModel()

            # Should handle empty data gracefully
            with patch('src.models.custom.pytorch_utils.get_device', return_value=torch.device('cpu')):
                predictions = predict_with_pytorch_model(model, empty_loader)
            assert len(predictions) == 0

            # 2. Test parameter counting on model without parameters
            class NoParamModel(nn.Module):
                def forward(self, x):
                    return x.mean(dim=1, keepdim=True)

            no_param_model = NoParamModel()
            param_info = count_parameters(no_param_model)
            assert param_info['total_parameters'] == 0

    def test_dataloader_creation_with_different_input_types(self, component_test_context):
        """다양한 입력 타입으로 DataLoader 생성 통합 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test with different pandas/numpy combinations

            # 1. DataFrame + Series
            X_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            y_series = pd.Series([5, 6])
            loader1 = create_dataloader(X_df, y_series, batch_size=1)
            assert len(loader1.dataset) == 2

            # 2. DataFrame + numpy array
            y_array = np.array([7, 8])
            loader2 = create_dataloader(X_df, y_array, batch_size=1)
            assert len(loader2.dataset) == 2

            # 3. numpy array + Series
            X_array = np.array([[1, 2], [3, 4]])
            loader3 = create_dataloader(X_array, y_series, batch_size=1)
            assert len(loader3.dataset) == 2

            # 4. Both numpy arrays
            loader4 = create_dataloader(X_array, y_array, batch_size=1)
            assert len(loader4.dataset) == 2

            # 5. Only X (no y)
            loader5 = create_dataloader(X_df, y=None, batch_size=1)
            assert len(loader5.dataset) == 2

            # All loaders should work consistently
            for loader in [loader1, loader2, loader3, loader4]:
                batch = next(iter(loader))
                assert len(batch) == 2  # X and y

            # loader5 should only have X
            batch = next(iter(loader5))
            assert len(batch) == 1  # Only X