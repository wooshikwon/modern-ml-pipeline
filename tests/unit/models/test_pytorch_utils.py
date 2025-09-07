"""
Unit tests for the PyTorch utilities module.
Tests GPU/CPU device selection, DataLoader creation, and common training functions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

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


class TestGetDevice:
    """Test get_device function."""
    
    def test_get_device_returns_device(self):
        """Test that get_device returns a torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'cpu']
    
    @patch('torch.cuda.is_available')
    def test_get_device_cpu_when_no_cuda(self, mock_cuda_available):
        """Test device selection when CUDA is not available."""
        mock_cuda_available.return_value = False
        device = get_device()
        assert device.type == 'cpu'
    
    @patch('torch.cuda.is_available')
    def test_get_device_cuda_when_available(self, mock_cuda_available):
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        device = get_device()
        assert device.type == 'cuda'


class TestCreateDataloader:
    """Test create_dataloader function."""
    
    def test_create_dataloader_with_numpy_arrays(self):
        """Test DataLoader creation with numpy arrays."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        dataloader = create_dataloader(X, y, batch_size=16, shuffle=True)
        
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 16
        assert len(dataloader.dataset) == 100
    
    def test_create_dataloader_with_pandas(self):
        """Test DataLoader creation with pandas DataFrame and Series."""
        X = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.randn(50))
        
        dataloader = create_dataloader(X, y, batch_size=8, shuffle=False)
        
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 8
        assert len(dataloader.dataset) == 50
    
    def test_create_dataloader_without_y(self):
        """Test DataLoader creation without target variable."""
        X = np.random.randn(30, 4)
        
        dataloader = create_dataloader(X, y=None, batch_size=10, shuffle=True)
        
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert len(dataloader.dataset) == 30
        
        # Test that dataset contains only X (no y)
        for batch in dataloader:
            assert len(batch) == 1  # Only X, no y
            break
    
    def test_create_dataloader_default_parameters(self):
        """Test DataLoader creation with default parameters."""
        X = np.random.randn(64, 2)
        y = np.random.randn(64)
        
        dataloader = create_dataloader(X, y)
        
        assert dataloader.batch_size == 32  # default
        assert len(dataloader.dataset) == 64


class TestModelFunctions:
    """Test model-related utility functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Simple linear model for testing
        self.model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        # Sample data
        self.X_train = torch.randn(100, 5)
        self.y_train = torch.randn(100)
        self.X_val = torch.randn(20, 5)
        self.y_val = torch.randn(20)
        
        # DataLoaders
        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        val_dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_val), batch_size=16
        )
    
    def test_validate_pytorch_model(self):
        """Test model validation function."""
        criterion = nn.MSELoss()
        device = torch.device('cpu')
        
        val_loss = validate_pytorch_model(self.model, self.val_loader, criterion, device)
        
        assert isinstance(val_loss, float)
        assert val_loss >= 0  # Loss should be non-negative
    
    def test_predict_with_pytorch_model(self):
        """Test prediction function."""
        predictions = predict_with_pytorch_model(self.model, self.test_loader, device=torch.device('cpu'))
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 20  # Same as validation data size
    
    @patch('src.models.custom.pytorch_utils.logger')
    def test_train_pytorch_model_basic(self, mock_logger):
        """Test basic training functionality."""
        history = train_pytorch_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=2,  # Short training for test
            learning_rate=0.01,
            log_interval=1
        )
        
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'best_epoch' in history
        assert len(history['train_loss']) <= 2  # Should have at most 2 epochs
        assert len(history['val_loss']) <= 2
    
    def test_train_pytorch_model_without_validation(self):
        """Test training without validation data."""
        history = train_pytorch_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=None,
            epochs=1,
            learning_rate=0.01
        )
        
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 1
        assert len(history['val_loss']) == 0  # No validation
    
    @patch('src.models.custom.pytorch_utils.logger')
    def test_train_pytorch_model_early_stopping(self, mock_logger):
        """Test early stopping functionality."""
        # Create a model that will have consistent loss (trigger early stopping)
        simple_model = nn.Linear(5, 1)
        
        history = train_pytorch_model(
            model=simple_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=20,
            learning_rate=0.001,
            early_stopping_patience=3,
            log_interval=5
        )
        
        assert isinstance(history, dict)
        assert 'best_epoch' in history
        # Early stopping should trigger before 20 epochs in most cases
        assert len(history['train_loss']) <= 20


class TestModelUtilities:
    """Test model utility functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def test_get_model_device(self):
        """Test getting model device."""
        # Initially on CPU
        device = get_model_device(self.model)
        assert device.type == 'cpu'
        
        # Move to CPU explicitly (should still be CPU)
        self.model = self.model.to(torch.device('cpu'))
        device = get_model_device(self.model)
        assert device.type == 'cpu'
    
    def test_count_parameters(self):
        """Test parameter counting function."""
        param_info = count_parameters(self.model)
        
        assert isinstance(param_info, dict)
        assert 'total_parameters' in param_info
        assert 'trainable_parameters' in param_info
        assert 'non_trainable_parameters' in param_info
        
        # All parameters should be trainable by default
        assert param_info['total_parameters'] == param_info['trainable_parameters']
        assert param_info['non_trainable_parameters'] == 0
        
        # Check reasonable parameter count for our model
        # (10*50 + 50) + (50*20 + 20) + (20*1 + 1) = 500 + 50 + 1000 + 20 + 20 + 1 = 1591
        expected_params = (10*50 + 50) + (50*20 + 20) + (20*1 + 1)
        assert param_info['total_parameters'] == expected_params
    
    def test_count_parameters_with_frozen_layers(self):
        """Test parameter counting with frozen layers."""
        # Freeze first layer
        for param in self.model[0].parameters():
            param.requires_grad = False
        
        param_info = count_parameters(self.model)
        
        # First layer has (10*50 + 50) = 550 parameters
        frozen_params = 10*50 + 50
        total_params = (10*50 + 50) + (50*20 + 20) + (20*1 + 1)
        trainable_params = total_params - frozen_params
        
        assert param_info['total_parameters'] == total_params
        assert param_info['trainable_parameters'] == trainable_params
        assert param_info['non_trainable_parameters'] == frozen_params


class TestSetSeed:
    """Test set_seed function."""
    
    @patch('src.models.custom.pytorch_utils.logger')
    def test_set_seed_default(self, mock_logger):
        """Test setting seed with default value."""
        try:
            set_seed()
            assert True
        except Exception as e:
            pytest.fail(f"Default seed setting failed: {e}")
    
    @patch('src.models.custom.pytorch_utils.logger')
    def test_set_seed_custom_value(self, mock_logger):
        """Test setting seed with custom value."""
        try:
            set_seed(12345)
            assert True
        except Exception as e:
            pytest.fail(f"Custom seed setting failed: {e}")
    
    def test_set_seed_reproducibility(self):
        """Test that seed setting provides reproducibility."""
        seed_value = 42
        
        # Set seed and generate random tensors
        set_seed(seed_value)
        random1 = torch.randn(5, 3)
        
        # Reset seed and generate again
        set_seed(seed_value)
        random2 = torch.randn(5, 3)
        
        # Should be identical
        assert torch.equal(random1, random2)
    
    @patch('torch.cuda.is_available')
    @patch('src.models.custom.pytorch_utils.logger')
    def test_set_seed_with_cuda_available(self, mock_logger, mock_cuda_available):
        """Test seed setting when CUDA is available."""
        mock_cuda_available.return_value = True
        
        try:
            set_seed(42)
            assert True
        except Exception as e:
            pytest.fail(f"CUDA seed setting failed: {e}")