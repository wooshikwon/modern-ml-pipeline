"""
PyTorch Models Unit Tests - No Mock Hell Approach
Real neural networks, real training, real predictions validation
Following comprehensive testing strategy document principles
"""

import pytest
import pandas as pd
import numpy as np

# Skip all tests in this module if PyTorch is not installed
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
optim = pytest.importorskip("torch.optim")
from torch.utils.data import DataLoader, TensorDataset


class SimpleClassificationNet(nn.Module):
    """Simple neural network for classification testing."""
    
    def __init__(self, input_dim, hidden_dim=16, output_dim=2):
        super(SimpleClassificationNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleRegressionNet(nn.Module):
    """Simple neural network for regression testing."""
    
    def __init__(self, input_dim, hidden_dim=16):
        super(SimpleRegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze()


class TestPyTorchModels:
    """Test PyTorch models with real training and prediction."""
    
    def test_pytorch_classification_model_training(self, test_data_generator):
        """Test PyTorch classification model training with real data."""
        # Given: Real classification data
        X, y = test_data_generator.classification_data(n_samples=100, n_features=5)
        # Remove entity_id column for PyTorch training
        X_features = X.drop('entity_id', axis=1) if 'entity_id' in X.columns else X
        X_tensor = torch.FloatTensor(X_features.values if hasattr(X_features, 'values') else X_features)
        y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y)
        
        # When: Training PyTorch model
        model = SimpleClassificationNet(input_dim=5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        model.train()
        for epoch in range(5):  # Quick training
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Then: Model is trained successfully
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor)
            pred_classes = torch.argmax(predictions, dim=1)
            accuracy = (pred_classes == y_tensor).float().mean()
            assert accuracy > 0.5  # Better than random
    
    def test_pytorch_regression_model_training(self, test_data_generator):
        """Test PyTorch regression model training with real data."""
        # Given: Real regression data
        X, y = test_data_generator.regression_data(n_samples=100, n_features=5)
        # Remove entity_id column for PyTorch training
        X_features = X.drop('entity_id', axis=1) if 'entity_id' in X.columns else X
        X_tensor = torch.FloatTensor(X_features.values if hasattr(X_features, 'values') else X_features)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
        
        # When: Training PyTorch regression model
        model = SimpleRegressionNet(input_dim=5)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        model.train()
        initial_loss = float('inf')
        for epoch in range(10):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch == 0:
                initial_loss = epoch_loss
        
        # Then: Model is trained and loss decreases
        assert epoch_loss < initial_loss  # Training reduces loss
        
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor)
            assert predictions.shape == y_tensor.shape
    
    def test_pytorch_model_save_and_load(self, test_data_generator, isolated_temp_directory):
        """Test PyTorch model save and load functionality."""
        # Given: Trained model
        X, y = test_data_generator.classification_data(n_samples=50, n_features=4)
        model = SimpleClassificationNet(input_dim=4)
        
        # Simple training - remove entity_id column
        X_features = X.drop('entity_id', axis=1) if 'entity_id' in X.columns else X
        X_tensor = torch.FloatTensor(X_features.values if hasattr(X_features, 'values') else X_features)
        optimizer = optim.Adam(model.parameters())
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = outputs.mean()  # Dummy loss
            loss.backward()
            optimizer.step()
        
        # When: Saving and loading model
        save_path = isolated_temp_directory / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        new_model = SimpleClassificationNet(input_dim=4)
        new_model.load_state_dict(torch.load(save_path))
        
        # Then: Models produce same output
        model.eval()
        new_model.eval()
        with torch.no_grad():
            original_output = model(X_tensor)
            loaded_output = new_model(X_tensor)
            assert torch.allclose(original_output, loaded_output)
    
    def test_pytorch_model_gpu_compatibility(self):
        """Test PyTorch model device handling."""
        # Given: Simple model
        model = SimpleClassificationNet(input_dim=5)
        
        # When: Checking device compatibility
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Then: Model is on correct device
        if torch.cuda.is_available():
            assert next(model.parameters()).is_cuda
        else:
            assert not next(model.parameters()).is_cuda
        
        # Test forward pass on device
        X = torch.randn(10, 5).to(device)
        output = model(X)
        assert output.device == X.device
    
    def test_pytorch_model_batch_processing(self, test_data_generator):
        """Test PyTorch model with different batch sizes."""
        # Given: Model and data
        X, _ = test_data_generator.classification_data(n_samples=100, n_features=5)
        # Remove entity_id column for PyTorch training
        X_features = X.drop('entity_id', axis=1) if 'entity_id' in X.columns else X
        model = SimpleClassificationNet(input_dim=5)
        model.eval()
        
        # When: Processing different batch sizes
        batch_sizes = [1, 10, 32, 100]
        outputs = []
        
        with torch.no_grad():
            for batch_size in batch_sizes:
                X_batch_data = X_features[:batch_size]
                X_batch = torch.FloatTensor(X_batch_data.values if hasattr(X_batch_data, 'values') else X_batch_data)
                output = model(X_batch)
                outputs.append(output)
        
        # Then: All batch sizes work correctly
        assert outputs[0].shape[0] == 1
        assert outputs[1].shape[0] == 10
        assert outputs[2].shape[0] == 32
        assert outputs[3].shape[0] == 100