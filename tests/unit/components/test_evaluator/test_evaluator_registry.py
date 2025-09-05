"""
Unit tests for EvaluatorRegistry.
Tests registry pattern with self-registration mechanism for model evaluators.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from src.components.evaluator.registry import EvaluatorRegistry
from src.interface import BaseEvaluator
from src.settings import Settings


class TestEvaluatorRegistryBasicOperations:
    """Test EvaluatorRegistry basic CRUD operations."""
    
    def test_register_valid_evaluator(self):
        """Test registering a valid evaluator class."""
        # Arrange
        class MockEvaluator(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        # Act
        EvaluatorRegistry.register("test_task", MockEvaluator)
        
        # Assert
        assert "test_task" in EvaluatorRegistry.evaluators
        assert EvaluatorRegistry.evaluators["test_task"] == MockEvaluator
    
    def test_register_invalid_evaluator_type_error(self):
        """Test registering non-BaseEvaluator class raises TypeError."""
        # Arrange
        class InvalidEvaluator:
            pass
        
        # Act & Assert
        with pytest.raises(TypeError, match="must be a subclass of BaseEvaluator"):
            EvaluatorRegistry.register("invalid", InvalidEvaluator)
    
    def test_get_evaluator_class_existing(self):
        """Test getting existing evaluator class."""
        # Arrange
        class MockEvaluator(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        EvaluatorRegistry.register("existing_task", MockEvaluator)
        
        # Act
        result = EvaluatorRegistry.get_evaluator_class("existing_task")
        
        # Assert
        assert result == MockEvaluator
    
    def test_get_evaluator_class_nonexistent_value_error(self):
        """Test getting non-existent evaluator raises ValueError with available options."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown task type: 'nonexistent'"):
            EvaluatorRegistry.get_evaluator_class("nonexistent")
    
    def test_get_available_tasks(self):
        """Test getting all registered task types."""
        # Arrange
        class MockEvaluator1(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        class MockEvaluator2(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        EvaluatorRegistry.register("task1", MockEvaluator1)
        EvaluatorRegistry.register("task2", MockEvaluator2)
        
        # Act
        tasks = EvaluatorRegistry.get_available_tasks()
        
        # Assert
        assert "task1" in tasks
        assert "task2" in tasks
        assert isinstance(tasks, list)


class TestEvaluatorRegistryInstanceCreation:
    """Test EvaluatorRegistry instance creation functionality."""
    
    def test_create_evaluator_instance(self):
        """Test creating evaluator instance with arguments."""
        # Arrange
        class MockEvaluator(BaseEvaluator):
            def __init__(self, data_interface, test_arg=None):
                self.data_interface = data_interface
                self.test_arg = test_arg
            
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        EvaluatorRegistry.register("creatable_task", MockEvaluator)
        mock_data_interface = Mock()
        
        # Act
        instance = EvaluatorRegistry.create("creatable_task", mock_data_interface, test_arg="test_value")
        
        # Assert
        assert isinstance(instance, MockEvaluator)
        assert instance.data_interface == mock_data_interface
        assert instance.test_arg == "test_value"
    
    def test_create_nonexistent_evaluator_error(self):
        """Test creating non-existent evaluator raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown task type for evaluator"):
            EvaluatorRegistry.create("nonexistent_task", Mock())
    
    def test_create_error_message_includes_available_types(self):
        """Test that ValueError includes available task types."""
        # Arrange
        class AvailableEvaluator(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        EvaluatorRegistry.register("available_task", AvailableEvaluator)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            EvaluatorRegistry.create("missing_task", Mock())
        
        error_message = str(exc_info.value)
        assert "available_task" in error_message
        assert "Available types:" in error_message


class TestEvaluatorRegistryIsolation:
    """Test EvaluatorRegistry isolation and cleanup mechanisms."""
    
    def test_registry_state_isolation(self):
        """Test that registry state is properly isolated between tests."""
        # Arrange
        initial_count = len(EvaluatorRegistry.evaluators)
        
        class TestEvaluator(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        # Act
        EvaluatorRegistry.register("isolation_test", TestEvaluator)
        registered_count = len(EvaluatorRegistry.evaluators)
        
        # Assert
        assert registered_count == initial_count + 1
        # The clean_registries fixture should restore state after test


class TestEvaluatorRegistryEdgeCases:
    """Test EvaluatorRegistry edge cases and error scenarios."""
    
    def test_register_duplicate_evaluator_overwrites(self):
        """Test registering duplicate task type overwrites previous."""
        # Arrange
        class Evaluator1(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return "evaluator1"
        
        class Evaluator2(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return "evaluator2"
        
        # Act
        EvaluatorRegistry.register("duplicate_task", Evaluator1)
        first_evaluator = EvaluatorRegistry.get_evaluator_class("duplicate_task")
        
        EvaluatorRegistry.register("duplicate_task", Evaluator2)
        second_evaluator = EvaluatorRegistry.get_evaluator_class("duplicate_task")
        
        # Assert
        assert first_evaluator == Evaluator1
        assert second_evaluator == Evaluator2
        assert first_evaluator != second_evaluator
    
    def test_empty_task_type_registration(self):
        """Test registering with empty string task type."""
        # Arrange
        class EmptyTaskEvaluator(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        # Act
        EvaluatorRegistry.register("", EmptyTaskEvaluator)
        
        # Assert
        assert "" in EvaluatorRegistry.evaluators
        assert EvaluatorRegistry.get_evaluator_class("") == EmptyTaskEvaluator


class TestEvaluatorSelfRegistration:
    """Test evaluator self-registration mechanism."""
    
    def test_classification_evaluator_self_registration(self):
        """Test that classification evaluator automatically registers itself on import."""
        # Act - Import triggers self-registration
        from src.components.evaluator.modules import classification_evaluator
        
        # Assert
        assert "classification" in EvaluatorRegistry.evaluators
        evaluator_class = EvaluatorRegistry.get_evaluator_class("classification")
        assert evaluator_class.__name__ == "ClassificationEvaluator"
    
    def test_regression_evaluator_self_registration(self):
        """Test that regression evaluator automatically registers itself on import."""
        # Act - Import triggers self-registration  
        from src.components.evaluator.modules import regression_evaluator
        
        # Assert
        assert "regression" in EvaluatorRegistry.evaluators
        evaluator_class = EvaluatorRegistry.get_evaluator_class("regression")
        assert evaluator_class.__name__ == "RegressionEvaluator"
    
    def test_clustering_evaluator_self_registration(self):
        """Test that clustering evaluator automatically registers itself on import."""
        # Act - Import triggers self-registration
        from src.components.evaluator.modules import clustering_evaluator
        
        # Assert
        assert "clustering" in EvaluatorRegistry.evaluators
        evaluator_class = EvaluatorRegistry.get_evaluator_class("clustering")
        assert evaluator_class.__name__ == "ClusteringEvaluator"
    
    def test_causal_evaluator_self_registration(self):
        """Test that causal evaluator automatically registers itself on import."""
        # Act - Import triggers self-registration
        from src.components.evaluator.modules import causal_evaluator
        
        # Assert
        assert "causal" in EvaluatorRegistry.evaluators
        evaluator_class = EvaluatorRegistry.get_evaluator_class("causal")
        assert evaluator_class.__name__ == "CausalEvaluator"


class TestEvaluatorRegistryRobustness:
    """Test EvaluatorRegistry robustness and error recovery."""
    
    def test_multiple_registrations_stability(self):
        """Test that multiple rapid registrations maintain stability."""
        # Arrange
        evaluators_to_register = []
        for i in range(5):
            class TestEvaluator(BaseEvaluator):
                def __init__(self, index=i):
                    self.index = index
                def evaluate(self, *args, **kwargs):
                    return f"evaluator_{i}"
            evaluators_to_register.append((f"test_task_{i}", TestEvaluator))
        
        # Act
        for task_type, evaluator_class in evaluators_to_register:
            EvaluatorRegistry.register(task_type, evaluator_class)
        
        # Assert
        for i in range(5):
            task_type = f"test_task_{i}"
            assert task_type in EvaluatorRegistry.evaluators
            retrieved_class = EvaluatorRegistry.get_evaluator_class(task_type)
            instance = EvaluatorRegistry.create(task_type, Mock())
            assert instance.evaluate() == f"evaluator_{i}"
    
    def test_registry_consistency_after_errors(self):
        """Test that registry remains consistent after registration errors."""
        # Arrange
        class ValidEvaluator(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        class InvalidEvaluator:
            pass
        
        initial_count = len(EvaluatorRegistry.evaluators)
        
        # Act
        EvaluatorRegistry.register("valid_task", ValidEvaluator)
        
        try:
            EvaluatorRegistry.register("invalid_task", InvalidEvaluator)
        except TypeError:
            pass  # Expected error
        
        # Assert
        assert len(EvaluatorRegistry.evaluators) == initial_count + 1
        assert "valid_task" in EvaluatorRegistry.evaluators
        assert "invalid_task" not in EvaluatorRegistry.evaluators
    
    def test_get_available_tasks_returns_copy(self):
        """Test that get_available_tasks returns a copy, not reference."""
        # Arrange
        class TestEvaluator(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return Mock()
        
        EvaluatorRegistry.register("copy_test_task", TestEvaluator)
        
        # Act
        tasks1 = EvaluatorRegistry.get_available_tasks()
        tasks2 = EvaluatorRegistry.get_available_tasks()
        
        # Assert
        assert tasks1 == tasks2
        assert tasks1 is not tasks2  # Different list objects
        assert "copy_test_task" in tasks1
        
        # Modify returned list shouldn't affect registry
        tasks1.append("fake_task")
        original_tasks = EvaluatorRegistry.get_available_tasks()
        assert "fake_task" not in original_tasks
    
    def test_task_type_case_sensitivity(self):
        """Test that task types are case sensitive."""
        # Arrange
        class LowerCaseEvaluator(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return "lowercase"
        
        class UpperCaseEvaluator(BaseEvaluator):
            def evaluate(self, *args, **kwargs):
                return "uppercase"
        
        # Act
        EvaluatorRegistry.register("classification", LowerCaseEvaluator)
        EvaluatorRegistry.register("CLASSIFICATION", UpperCaseEvaluator)
        
        # Assert
        lower_evaluator = EvaluatorRegistry.get_evaluator_class("classification")
        upper_evaluator = EvaluatorRegistry.get_evaluator_class("CLASSIFICATION")
        
        assert lower_evaluator == LowerCaseEvaluator
        assert upper_evaluator == UpperCaseEvaluator
        assert lower_evaluator != upper_evaluator