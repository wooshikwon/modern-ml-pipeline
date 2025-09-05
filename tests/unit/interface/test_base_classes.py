"""
Unit tests for interface base classes.
Tests abstract base classes and their contracts.
"""

import pytest
from abc import ABC
from unittest.mock import Mock, patch
import pandas as pd
from typing import Dict, Any, Optional, List

from src.interface import (
    BaseAdapter,
    BaseFetcher,
    BaseEvaluator,
    BaseFactory,
    BasePreprocessor,
    BaseTrainer,
    BaseModel
)
from src.settings import Settings
from tests.helpers.builders import ConfigBuilder, RecipeBuilder


class TestBaseAdapter:
    """Test BaseAdapter abstract base class."""
    
    def test_base_adapter_is_abstract(self):
        """Test that BaseAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAdapter()
    
    def test_base_adapter_abstract_methods(self):
        """Test that BaseAdapter has required abstract methods."""
        # Create a concrete implementation
        class ConcreteAdapter(BaseAdapter):
            def __init__(self, settings):
                super().__init__(settings)
            
            def load_data(self, source_uri: str, **kwargs) -> pd.DataFrame:
                return pd.DataFrame({'test': [1, 2, 3]})
            
            def save_data(self, data: pd.DataFrame, target_uri: str, **kwargs) -> None:
                pass
            
            def get_connection_info(self) -> Dict[str, Any]:
                return {"type": "test"}
        
        settings = Mock(spec=Settings)
        adapter = ConcreteAdapter(settings)
        
        assert hasattr(adapter, 'load_data')
        assert hasattr(adapter, 'save_data')
        assert hasattr(adapter, 'get_connection_info')
        assert adapter.settings == settings
    
    def test_base_adapter_inheritance(self):
        """Test BaseAdapter inheritance and ABC behavior."""
        assert issubclass(BaseAdapter, ABC)
        
        # Test missing abstract method
        class IncompleteAdapter(BaseAdapter):
            def __init__(self, settings):
                super().__init__(settings)
            
            def load_data(self, source_uri: str, **kwargs) -> pd.DataFrame:
                return pd.DataFrame()
            
            # Missing save_data and get_connection_info
        
        with pytest.raises(TypeError):
            settings = Mock(spec=Settings)
            IncompleteAdapter(settings)


class TestBaseFetcher:
    """Test BaseFetcher abstract base class."""
    
    def test_base_fetcher_is_abstract(self):
        """Test that BaseFetcher cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFetcher()
    
    def test_base_fetcher_concrete_implementation(self):
        """Test concrete implementation of BaseFetcher."""
        class ConcreteFetcher(BaseFetcher):
            def fetch(self, entities: pd.DataFrame, run_mode: str = "batch", **kwargs) -> pd.DataFrame:
                return entities.copy()
            
            def get_feature_names(self) -> List[str]:
                return ["feature1", "feature2"]
            
            def validate_connection(self) -> bool:
                return True
        
        fetcher = ConcreteFetcher()
        
        assert hasattr(fetcher, 'fetch')
        assert hasattr(fetcher, 'get_feature_names')
        assert hasattr(fetcher, 'validate_connection')
        
        # Test functionality
        test_df = pd.DataFrame({'id': [1, 2, 3]})
        result = fetcher.fetch(test_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        features = fetcher.get_feature_names()
        assert isinstance(features, list)
        assert len(features) == 2
        
        assert fetcher.validate_connection() is True


class TestBaseEvaluator:
    """Test BaseEvaluator abstract base class."""
    
    def test_base_evaluator_is_abstract(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()
    
    def test_base_evaluator_concrete_implementation(self):
        """Test concrete implementation of BaseEvaluator."""
        class ConcreteEvaluator(BaseEvaluator):
            def __init__(self, data_interface):
                self.data_interface = data_interface
            
            def evaluate(self, y_true: pd.Series, y_pred: pd.Series, **kwargs) -> Dict[str, float]:
                return {"accuracy": 0.95, "f1": 0.92}
            
            def get_supported_metrics(self) -> List[str]:
                return ["accuracy", "f1", "precision", "recall"]
            
            def validate_predictions(self, y_pred: pd.Series) -> bool:
                return len(y_pred) > 0
        
        data_interface = Mock()
        evaluator = ConcreteEvaluator(data_interface)
        
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(evaluator, 'get_supported_metrics')
        assert hasattr(evaluator, 'validate_predictions')
        assert evaluator.data_interface == data_interface
        
        # Test functionality
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 0])
        
        results = evaluator.evaluate(y_true, y_pred)
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "f1" in results
        
        metrics = evaluator.get_supported_metrics()
        assert isinstance(metrics, list)
        assert "accuracy" in metrics
        
        assert evaluator.validate_predictions(y_pred) is True


class TestBasePreprocessor:
    """Test BasePreprocessor abstract base class."""
    
    def test_base_preprocessor_is_abstract(self):
        """Test that BasePreprocessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePreprocessor()
    
    def test_base_preprocessor_concrete_implementation(self):
        """Test concrete implementation of BasePreprocessor."""
        class ConcretePreprocessor(BasePreprocessor):
            def __init__(self, settings):
                self.settings = settings
                self.is_fitted = False
            
            def fit(self, data: pd.DataFrame) -> 'ConcretePreprocessor':
                self.is_fitted = True
                return self
            
            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                if not self.is_fitted:
                    raise ValueError("Preprocessor not fitted")
                return data * 2  # Simple transformation
            
            def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return self.fit(data).transform(data)
            
            def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
                if input_features:
                    return [f"{name}_transformed" for name in input_features]
                return ["feature_transformed"]
        
        settings = Mock(spec=Settings)
        preprocessor = ConcretePreprocessor(settings)
        
        assert hasattr(preprocessor, 'fit')
        assert hasattr(preprocessor, 'transform')
        assert hasattr(preprocessor, 'fit_transform')
        assert hasattr(preprocessor, 'get_feature_names_out')
        assert preprocessor.settings == settings
        
        # Test functionality
        test_data = pd.DataFrame({'feature': [1, 2, 3]})
        
        # Test fit
        fitted = preprocessor.fit(test_data)
        assert fitted is preprocessor
        assert preprocessor.is_fitted is True
        
        # Test transform
        transformed = preprocessor.transform(test_data)
        assert isinstance(transformed, pd.DataFrame)
        assert transformed['feature'].tolist() == [2, 4, 6]
        
        # Test feature names
        features = preprocessor.get_feature_names_out(['col1', 'col2'])
        assert features == ['col1_transformed', 'col2_transformed']


class TestBaseTrainer:
    """Test BaseTrainer abstract base class."""
    
    def test_base_trainer_is_abstract(self):
        """Test that BaseTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTrainer()
    
    def test_base_trainer_concrete_implementation(self):
        """Test concrete implementation of BaseTrainer."""
        class ConcreteTrainer(BaseTrainer):
            def __init__(self, settings):
                self.settings = settings
                self.model = None
            
            def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
                # Mock training
                self.model = Mock()
                self.model.predict = Mock(return_value=[0, 1, 1])
                return self.model
            
            def validate(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
                return {"accuracy": 0.89, "loss": 0.23}
            
            def save_model(self, model: Any, path: str) -> None:
                # Mock saving
                pass
            
            def load_model(self, path: str) -> Any:
                # Mock loading
                return Mock()
        
        settings = Mock(spec=Settings)
        trainer = ConcreteTrainer(settings)
        
        assert hasattr(trainer, 'train')
        assert hasattr(trainer, 'validate')
        assert hasattr(trainer, 'save_model')
        assert hasattr(trainer, 'load_model')
        assert trainer.settings == settings
        
        # Test functionality
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([0, 1, 1])
        
        model = trainer.train(X, y)
        assert model is not None
        assert trainer.model == model
        
        validation_results = trainer.validate(X, y)
        assert isinstance(validation_results, dict)
        assert "accuracy" in validation_results


class TestBaseModel:
    """Test BaseModel abstract base class."""
    
    def test_base_model_is_abstract(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_base_model_concrete_implementation(self):
        """Test concrete implementation of BaseModel."""
        class ConcreteModel(BaseModel):
            def __init__(self):
                self.is_fitted = False
            
            def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ConcreteModel':
                self.is_fitted = True
                return self
            
            def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                return pd.Series([1] * len(X))
            
            def predict_proba(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                probs = [[0.3, 0.7]] * len(X)
                return pd.DataFrame(probs, columns=['class_0', 'class_1'])
            
            def get_params(self) -> Dict[str, Any]:
                return {"param1": "value1", "param2": 42}
            
            def set_params(self, **params) -> 'ConcreteModel':
                # Mock parameter setting
                return self
        
        model = ConcreteModel()
        
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'get_params')
        assert hasattr(model, 'set_params')
        
        # Test functionality
        X = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        y = pd.Series([0, 1])
        
        fitted = model.fit(X, y)
        assert fitted is model
        assert model.is_fitted is True
        
        predictions = model.predict(X)
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 2
        
        probabilities = model.predict_proba(X)
        assert isinstance(probabilities, pd.DataFrame)
        assert probabilities.shape == (2, 2)
        
        params = model.get_params()
        assert isinstance(params, dict)
        assert "param1" in params


class TestBaseFactory:
    """Test BaseFactory abstract base class."""
    
    def test_base_factory_is_abstract(self):
        """Test that BaseFactory cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFactory()
    
    def test_base_factory_concrete_implementation(self):
        """Test concrete implementation of BaseFactory."""
        class ConcreteFactory(BaseFactory):
            def __init__(self, settings):
                self.settings = settings
            
            def create_component(self, component_type: str, **kwargs) -> Any:
                if component_type == "adapter":
                    return Mock(spec=BaseAdapter)
                elif component_type == "fetcher":
                    return Mock(spec=BaseFetcher)
                else:
                    raise ValueError(f"Unknown component type: {component_type}")
            
            def list_available_components(self) -> List[str]:
                return ["adapter", "fetcher", "evaluator"]
            
            def validate_configuration(self, config: Dict[str, Any]) -> bool:
                return "component_type" in config
        
        settings = Mock(spec=Settings)
        factory = ConcreteFactory(settings)
        
        assert hasattr(factory, 'create_component')
        assert hasattr(factory, 'list_available_components')
        assert hasattr(factory, 'validate_configuration')
        assert factory.settings == settings
        
        # Test functionality
        adapter = factory.create_component("adapter")
        assert adapter is not None
        
        components = factory.list_available_components()
        assert isinstance(components, list)
        assert "adapter" in components
        
        config_valid = factory.validate_configuration({"component_type": "adapter"})
        assert config_valid is True
        
        config_invalid = factory.validate_configuration({"invalid": "config"})
        assert config_invalid is False


class TestInterfaceInheritance:
    """Test interface inheritance patterns."""
    
    def test_all_interfaces_are_abstract(self):
        """Test that all interface classes are abstract."""
        interfaces = [
            BaseAdapter,
            BaseFetcher, 
            BaseEvaluator,
            BaseFactory,
            BasePreprocessor,
            BaseTrainer,
            BaseModel
        ]
        
        for interface in interfaces:
            assert issubclass(interface, ABC)
            with pytest.raises(TypeError):
                interface()
    
    def test_interface_method_contracts(self):
        """Test that interfaces define expected method contracts."""
        # BaseAdapter contract
        assert hasattr(BaseAdapter, '__abstractmethods__')
        
        # BaseFetcher contract
        assert hasattr(BaseFetcher, '__abstractmethods__')
        
        # BaseEvaluator contract
        assert hasattr(BaseEvaluator, '__abstractmethods__')
        
        # Test that abstract methods are properly defined
        for interface in [BaseAdapter, BaseFetcher, BaseEvaluator, BaseFactory, 
                         BasePreprocessor, BaseTrainer, BaseModel]:
            abstract_methods = getattr(interface, '__abstractmethods__', set())
            assert len(abstract_methods) > 0  # Each interface should have abstract methods


class TestInterfaceDocumentation:
    """Test interface documentation and metadata."""
    
    def test_interfaces_have_docstrings(self):
        """Test that interface classes have docstrings."""
        interfaces = [
            BaseAdapter,
            BaseFetcher,
            BaseEvaluator,
            BaseFactory,
            BasePreprocessor,
            BaseTrainer,
            BaseModel
        ]
        
        for interface in interfaces:
            assert interface.__doc__ is not None
            assert len(interface.__doc__.strip()) > 0