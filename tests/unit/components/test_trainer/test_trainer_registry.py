"""
Unit tests for TrainerRegistry.
Tests registry pattern with self-registration mechanism for trainers.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from src.components.trainer.registry import TrainerRegistry
from src.interface import BaseTrainer
from src.settings import Settings


class TestTrainerRegistryBasicOperations:
    """Test TrainerRegistry basic CRUD operations."""
    
    def test_register_valid_trainer(self):
        """Test registering a valid trainer class."""
        # Arrange
        class MockTrainer(BaseTrainer):
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        # Act
        TrainerRegistry.register("test_trainer", MockTrainer)
        
        # Assert
        assert "test_trainer" in TrainerRegistry.trainers
        assert TrainerRegistry.trainers["test_trainer"] == MockTrainer
    
    def test_register_invalid_trainer_type_error(self):
        """Test registering non-BaseTrainer class raises TypeError."""
        # Arrange
        class InvalidTrainer:
            pass
        
        # Act & Assert
        with pytest.raises(TypeError, match="must be a subclass of BaseTrainer"):
            TrainerRegistry.register("invalid", InvalidTrainer)
    
    def test_get_trainer_class_existing(self):
        """Test getting existing trainer class."""
        # Arrange
        class MockTrainer(BaseTrainer):
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        TrainerRegistry.register("existing_trainer", MockTrainer)
        
        # Act
        result = TrainerRegistry.get_trainer_class("existing_trainer")
        
        # Assert
        assert result == MockTrainer
    
    def test_create_trainer_instance(self):
        """Test creating trainer instance."""
        # Arrange
        class MockTrainer(BaseTrainer):
            def __init__(self, test_arg=None):
                self.test_arg = test_arg
            
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        TrainerRegistry.register("creatable_trainer", MockTrainer)
        
        # Act
        instance = TrainerRegistry.create("creatable_trainer", test_arg="test_value")
        
        # Assert
        assert isinstance(instance, MockTrainer)
        assert instance.test_arg == "test_value"
    
    def test_create_nonexistent_trainer_error(self):
        """Test creating non-existent trainer raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown trainer type"):
            TrainerRegistry.create("nonexistent_trainer")


class TestTrainerRegistryInstanceCreation:
    """Test TrainerRegistry instance creation functionality."""
    
    def test_create_error_message_includes_available_types(self):
        """Test that ValueError includes available trainer types."""
        # Arrange
        class AvailableTrainer(BaseTrainer):
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        TrainerRegistry.register("available_trainer", AvailableTrainer)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            TrainerRegistry.create("missing_trainer")
        
        error_message = str(exc_info.value)
        assert "available_trainer" in error_message
        assert "Available types:" in error_message
    
    def test_create_with_kwargs(self):
        """Test creating trainer with keyword arguments."""
        # Arrange
        class ConfigurableTrainer(BaseTrainer):
            def __init__(self, param1=None, param2=None, **kwargs):
                self.param1 = param1
                self.param2 = param2
                self.kwargs = kwargs
            
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        TrainerRegistry.register("configurable", ConfigurableTrainer)
        
        # Act
        instance = TrainerRegistry.create(
            "configurable", 
            param1="value1", 
            param2="value2", 
            extra_param="extra"
        )
        
        # Assert
        assert instance.param1 == "value1"
        assert instance.param2 == "value2"
        assert instance.kwargs["extra_param"] == "extra"


class TestTrainerRegistryAvailableTypes:
    """Test TrainerRegistry available types functionality."""
    
    def test_get_available_types_empty(self):
        """Test getting available types when registry is empty."""
        # Arrange - registry is empty after cleanup
        
        # Act
        types = TrainerRegistry.get_available_types()
        
        # Assert
        assert isinstance(types, list)
    
    def test_get_available_types_with_trainers(self):
        """Test getting available types with registered trainers."""
        # Arrange
        class TrainerA(BaseTrainer):
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        class TrainerB(BaseTrainer):
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        TrainerRegistry.register("trainer_a", TrainerA)
        TrainerRegistry.register("trainer_b", TrainerB)
        
        # Act
        types = TrainerRegistry.get_available_types()
        
        # Assert
        assert "trainer_a" in types
        assert "trainer_b" in types
        assert len(types) >= 2
    
    def test_get_trainer_class_error_message_includes_available_types(self):
        """Test that get_trainer_class error includes available types."""
        # Arrange
        class AvailableTrainer(BaseTrainer):
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        TrainerRegistry.register("available", AvailableTrainer)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            TrainerRegistry.get_trainer_class("nonexistent")
        
        error_message = str(exc_info.value)
        assert "available" in error_message
        assert "Available types:" in error_message


class TestTrainerRegistryIsolation:
    """Test TrainerRegistry isolation and cleanup mechanisms."""
    
    def test_registry_state_isolation(self):
        """Test that registry state is properly isolated between tests."""
        # Arrange
        initial_count = len(TrainerRegistry.trainers)
        
        class TestTrainer(BaseTrainer):
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        # Act
        TrainerRegistry.register("isolation_test", TestTrainer)
        registered_count = len(TrainerRegistry.trainers)
        
        # Assert
        assert registered_count == initial_count + 1
        # The clean_registries fixture should restore state after test


class TestTrainerRegistryEdgeCases:
    """Test TrainerRegistry edge cases and error scenarios."""
    
    def test_register_duplicate_trainer_overwrites(self):
        """Test registering duplicate trainer type overwrites previous."""
        # Arrange
        class Trainer1(BaseTrainer):
            def train(self, *args, **kwargs):
                return "trainer1", Mock(), {}, {}
        
        class Trainer2(BaseTrainer):
            def train(self, *args, **kwargs):
                return "trainer2", Mock(), {}, {}
        
        # Act
        TrainerRegistry.register("duplicate_trainer", Trainer1)
        first_trainer = TrainerRegistry.trainers["duplicate_trainer"]
        
        TrainerRegistry.register("duplicate_trainer", Trainer2)
        second_trainer = TrainerRegistry.trainers["duplicate_trainer"]
        
        # Assert
        assert first_trainer == Trainer1
        assert second_trainer == Trainer2
        assert first_trainer != second_trainer
    
    def test_empty_trainer_type_registration(self):
        """Test registering with empty string trainer type."""
        # Arrange
        class EmptyTypeTrainer(BaseTrainer):
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        # Act
        TrainerRegistry.register("", EmptyTypeTrainer)
        
        # Assert
        assert "" in TrainerRegistry.trainers
        assert TrainerRegistry.trainers[""] == EmptyTypeTrainer


class TestTrainerSelfRegistration:
    """Test trainer self-registration mechanism."""
    
    def test_default_trainer_self_registration(self):
        """Test that default trainer automatically registers itself on import."""
        # Act - Import triggers self-registration
        from src.components.trainer.modules import trainer
        
        # Assert
        registered_types = list(TrainerRegistry.trainers.keys())
        assert "default" in registered_types
        
        # Verify trainer class can be retrieved and instantiated
        trainer_class = TrainerRegistry.get_trainer_class("default")
        assert trainer_class is not None
        assert issubclass(trainer_class, BaseTrainer)


class TestTrainerRegistryRobustness:
    """Test TrainerRegistry robustness and error recovery."""
    
    def test_multiple_registrations_stability(self):
        """Test that multiple rapid registrations maintain stability."""
        # Arrange
        trainers_to_register = []
        for i in range(5):
            class TestTrainer(BaseTrainer):
                def __init__(self, index=i):
                    self.index = index
                def train(self, *args, **kwargs):
                    return f"trainer_{i}", Mock(), {}, {}
            trainers_to_register.append((f"test_trainer_{i}", TestTrainer))
        
        # Act
        for trainer_type, trainer_class in trainers_to_register:
            TrainerRegistry.register(trainer_type, trainer_class)
        
        # Assert
        for i in range(5):
            trainer_type = f"test_trainer_{i}"
            assert trainer_type in TrainerRegistry.trainers
            retrieved_class = TrainerRegistry.trainers[trainer_type]
            instance = TrainerRegistry.create(trainer_type)
            result, _, _, _ = instance.train()
            assert result == f"trainer_{i}"
    
    def test_registry_consistency_after_errors(self):
        """Test that registry remains consistent after registration errors."""
        # Arrange
        class ValidTrainer(BaseTrainer):
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        class InvalidTrainer:
            pass
        
        initial_count = len(TrainerRegistry.trainers)
        
        # Act
        TrainerRegistry.register("valid_trainer", ValidTrainer)
        
        try:
            TrainerRegistry.register("invalid_trainer", InvalidTrainer)
        except TypeError:
            pass  # Expected error
        
        # Assert
        assert len(TrainerRegistry.trainers) == initial_count + 1
        assert "valid_trainer" in TrainerRegistry.trainers
        assert "invalid_trainer" not in TrainerRegistry.trainers
    
    def test_trainer_type_case_sensitivity(self):
        """Test that trainer types are case sensitive."""
        # Arrange
        class LowerCaseTrainer(BaseTrainer):
            def train(self, *args, **kwargs):
                return "lowercase", Mock(), {}, {}
        
        class UpperCaseTrainer(BaseTrainer):
            def train(self, *args, **kwargs):
                return "uppercase", Mock(), {}, {}
        
        # Act
        TrainerRegistry.register("trainer", LowerCaseTrainer)
        TrainerRegistry.register("TRAINER", UpperCaseTrainer)
        
        # Assert
        lower_trainer = TrainerRegistry.trainers["trainer"]
        upper_trainer = TrainerRegistry.trainers["TRAINER"]
        
        assert lower_trainer == LowerCaseTrainer
        assert upper_trainer == UpperCaseTrainer
        assert lower_trainer != upper_trainer


class TestTrainerRegistryIntegration:
    """Test TrainerRegistry integration scenarios."""
    
    def test_registry_preserves_inheritance_structure(self):
        """Test that registry preserves class inheritance information."""
        # Arrange
        class SpecializedTrainer(BaseTrainer):
            special_attribute = "special"
            
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
                
            def special_method(self):
                return "special_functionality"
        
        # Act
        TrainerRegistry.register("specialized", SpecializedTrainer)
        retrieved_class = TrainerRegistry.trainers["specialized"]
        instance = TrainerRegistry.create("specialized")
        
        # Assert
        assert retrieved_class == SpecializedTrainer
        assert isinstance(instance, BaseTrainer)
        assert isinstance(instance, SpecializedTrainer)
        assert hasattr(instance, "special_attribute")
        assert instance.special_attribute == "special"
        assert hasattr(instance, "special_method")
        assert instance.special_method() == "special_functionality"
    
    def test_registry_handles_complex_initialization(self):
        """Test that registry handles trainers with complex initialization."""
        # Arrange
        class ComplexTrainer(BaseTrainer):
            def __init__(self, settings, factory_provider=None, *args, **kwargs):
                if not settings:
                    raise ValueError("settings is mandatory")
                self.settings = settings
                self.factory_provider = factory_provider
                self.args = args
                self.kwargs = kwargs
            
            def train(self, *args, **kwargs):
                return Mock(), Mock(), {}, {}
        
        TrainerRegistry.register("complex", ComplexTrainer)
        
        # Act & Assert - should work with required param
        mock_settings = Mock(spec=Settings)
        instance1 = TrainerRegistry.create("complex", mock_settings)
        assert instance1.settings == mock_settings
        assert instance1.factory_provider is None
        
        # Act & Assert - should work with all params
        mock_factory = Mock()
        instance2 = TrainerRegistry.create(
            "complex", 
            mock_settings,
            factory_provider=mock_factory,
            extra_arg="extra",
            kwarg1="kw1"
        )
        assert instance2.settings == mock_settings
        assert instance2.factory_provider == mock_factory
        assert instance2.kwargs["kwarg1"] == "kw1"
        
        # Act & Assert - should fail without required param
        with pytest.raises((ValueError, TypeError)):
            TrainerRegistry.create("complex")