"""
Unit tests for PreprocessorStepRegistry.
Tests registry pattern with self-registration mechanism for preprocessing steps.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from src.components.preprocessor.registry import PreprocessorStepRegistry
from src.interface import BasePreprocessor
from src.settings import Settings


class TestPreprocessorStepRegistryBasicOperations:
    """Test PreprocessorStepRegistry basic CRUD operations."""
    
    def test_register_valid_preprocessor_step(self):
        """Test registering a valid preprocessor step class."""
        # Arrange
        class MockPreprocessorStep(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return Mock()
        
        # Act
        PreprocessorStepRegistry.register("test_step", MockPreprocessorStep)
        
        # Assert
        assert "test_step" in PreprocessorStepRegistry.preprocessor_steps
        assert PreprocessorStepRegistry.preprocessor_steps["test_step"] == MockPreprocessorStep
    
    def test_register_invalid_preprocessor_step_type_error(self):
        """Test registering non-BasePreprocessor class raises TypeError."""
        # Arrange
        class InvalidPreprocessorStep:
            pass
        
        # Act & Assert
        with pytest.raises(TypeError, match="must be a subclass of BasePreprocessor"):
            PreprocessorStepRegistry.register("invalid", InvalidPreprocessorStep)
    
    def test_get_preprocessor_step_existing(self):
        """Test getting existing preprocessor step class."""
        # Arrange
        class MockPreprocessorStep(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return Mock()
        
        PreprocessorStepRegistry.register("existing_step", MockPreprocessorStep)
        
        # Act
        # Note: PreprocessorStepRegistry might not have get_step_class method, using dict access
        result = PreprocessorStepRegistry.preprocessor_steps.get("existing_step")
        
        # Assert
        assert result == MockPreprocessorStep
    
    def test_create_preprocessor_step_instance(self):
        """Test creating preprocessor step instance."""
        # Arrange
        class MockPreprocessorStep(BasePreprocessor):
            def __init__(self, test_arg=None):
                self.test_arg = test_arg
            
            def fit(self, *args, **kwargs):
                return self
                
            def transform(self, *args, **kwargs):
                return Mock()
        
        PreprocessorStepRegistry.register("creatable_step", MockPreprocessorStep)
        
        # Act
        instance = PreprocessorStepRegistry.create("creatable_step", test_arg="test_value")
        
        # Assert
        assert isinstance(instance, MockPreprocessorStep)
        assert instance.test_arg == "test_value"
    
    def test_create_nonexistent_preprocessor_step_error(self):
        """Test creating non-existent preprocessor step raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown preprocessor step type"):
            PreprocessorStepRegistry.create("nonexistent_step")


class TestPreprocessorStepRegistryInstanceCreation:
    """Test PreprocessorStepRegistry instance creation functionality."""
    
    def test_create_error_message_includes_available_types(self):
        """Test that ValueError includes available step types."""
        # Arrange
        class AvailableStep(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return Mock()
        
        PreprocessorStepRegistry.register("available_step", AvailableStep)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            PreprocessorStepRegistry.create("missing_step")
        
        error_message = str(exc_info.value)
        assert "available_step" in error_message
        assert "Available types:" in error_message
    
    def test_create_with_kwargs(self):
        """Test creating step with keyword arguments."""
        # Arrange
        class ConfigurableStep(BasePreprocessor):
            def __init__(self, param1=None, param2=None, **kwargs):
                self.param1 = param1
                self.param2 = param2
                self.kwargs = kwargs
            
            def fit(self, *args, **kwargs):
                return self
                
            def transform(self, *args, **kwargs):
                return Mock()
        
        PreprocessorStepRegistry.register("configurable", ConfigurableStep)
        
        # Act
        instance = PreprocessorStepRegistry.create(
            "configurable", 
            param1="value1", 
            param2="value2", 
            extra_param="extra"
        )
        
        # Assert
        assert instance.param1 == "value1"
        assert instance.param2 == "value2"
        assert instance.kwargs["extra_param"] == "extra"


class TestPreprocessorStepRegistryIsolation:
    """Test PreprocessorStepRegistry isolation and cleanup mechanisms."""
    
    def test_registry_state_isolation(self):
        """Test that registry state is properly isolated between tests."""
        # Arrange
        initial_count = len(PreprocessorStepRegistry.preprocessor_steps)
        
        class TestStep(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return Mock()
        
        # Act
        PreprocessorStepRegistry.register("isolation_test", TestStep)
        registered_count = len(PreprocessorStepRegistry.preprocessor_steps)
        
        # Assert
        assert registered_count == initial_count + 1
        # The clean_registries fixture should restore state after test


class TestPreprocessorStepRegistryEdgeCases:
    """Test PreprocessorStepRegistry edge cases and error scenarios."""
    
    def test_register_duplicate_step_overwrites(self):
        """Test registering duplicate step type overwrites previous."""
        # Arrange
        class Step1(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return "step1"
        
        class Step2(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return "step2"
        
        # Act
        PreprocessorStepRegistry.register("duplicate_step", Step1)
        first_step = PreprocessorStepRegistry.preprocessor_steps["duplicate_step"]
        
        PreprocessorStepRegistry.register("duplicate_step", Step2)
        second_step = PreprocessorStepRegistry.preprocessor_steps["duplicate_step"]
        
        # Assert
        assert first_step == Step1
        assert second_step == Step2
        assert first_step != second_step
    
    def test_empty_step_type_registration(self):
        """Test registering with empty string step type."""
        # Arrange
        class EmptyStepType(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return Mock()
        
        # Act
        PreprocessorStepRegistry.register("", EmptyStepType)
        
        # Assert
        assert "" in PreprocessorStepRegistry.preprocessor_steps
        assert PreprocessorStepRegistry.preprocessor_steps[""] == EmptyStepType


class TestPreprocessorStepSelfRegistration:
    """Test preprocessor step self-registration mechanism."""
    
    def test_scaler_step_self_registration(self):
        """Test that scaler step automatically registers itself on import."""
        # Act - Import triggers self-registration
        from src.components.preprocessor.modules import scaler
        
        # Assert
        # Check for common scaler types that might be registered
        registered_steps = list(PreprocessorStepRegistry.preprocessor_steps.keys())
        scaler_steps = [step for step in registered_steps if 'scaler' in step.lower() or 'standard' in step.lower()]
        assert len(scaler_steps) > 0, f"No scaler steps found in: {registered_steps}"
    
    def test_encoder_step_self_registration(self):
        """Test that encoder step automatically registers itself on import."""
        try:
            # Act - Import triggers self-registration  
            from src.components.preprocessor.modules import encoder
            
            # Assert
            registered_steps = list(PreprocessorStepRegistry.preprocessor_steps.keys())
            encoder_steps = [step for step in registered_steps if 'encoder' in step.lower() or 'encode' in step.lower()]
            assert len(encoder_steps) > 0, f"No encoder steps found in: {registered_steps}"
        except ImportError as e:
            # Skip if optional dependency (category_encoders) is not available
            import pytest
            pytest.skip(f"Skipping encoder test due to missing dependency: {e}")
    
    def test_imputer_step_self_registration(self):
        """Test that imputer step automatically registers itself on import."""
        # Act - Import triggers self-registration
        from src.components.preprocessor.modules import imputer
        
        # Assert
        registered_steps = list(PreprocessorStepRegistry.preprocessor_steps.keys())
        imputer_steps = [step for step in registered_steps if 'imputer' in step.lower() or 'impute' in step.lower()]
        assert len(imputer_steps) > 0, f"No imputer steps found in: {registered_steps}"


class TestPreprocessorStepRegistryRobustness:
    """Test PreprocessorStepRegistry robustness and error recovery."""
    
    def test_multiple_registrations_stability(self):
        """Test that multiple rapid registrations maintain stability."""
        # Arrange
        steps_to_register = []
        for i in range(5):
            class TestStep(BasePreprocessor):
                def __init__(self, index=i):
                    self.index = index
                def fit(self, *args, **kwargs):
                    return self
                def transform(self, *args, **kwargs):
                    return f"step_{i}"
            steps_to_register.append((f"test_step_{i}", TestStep))
        
        # Act
        for step_type, step_class in steps_to_register:
            PreprocessorStepRegistry.register(step_type, step_class)
        
        # Assert
        for i in range(5):
            step_type = f"test_step_{i}"
            assert step_type in PreprocessorStepRegistry.preprocessor_steps
            retrieved_class = PreprocessorStepRegistry.preprocessor_steps[step_type]
            instance = PreprocessorStepRegistry.create(step_type)
            assert instance.transform() == f"step_{i}"
    
    def test_registry_consistency_after_errors(self):
        """Test that registry remains consistent after registration errors."""
        # Arrange
        class ValidStep(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return Mock()
        
        class InvalidStep:
            pass
        
        initial_count = len(PreprocessorStepRegistry.preprocessor_steps)
        
        # Act
        PreprocessorStepRegistry.register("valid_step", ValidStep)
        
        try:
            PreprocessorStepRegistry.register("invalid_step", InvalidStep)
        except TypeError:
            pass  # Expected error
        
        # Assert
        assert len(PreprocessorStepRegistry.preprocessor_steps) == initial_count + 1
        assert "valid_step" in PreprocessorStepRegistry.preprocessor_steps
        assert "invalid_step" not in PreprocessorStepRegistry.preprocessor_steps
    
    def test_step_type_case_sensitivity(self):
        """Test that step types are case sensitive."""
        # Arrange
        class LowerCaseStep(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return "lowercase"
        
        class UpperCaseStep(BasePreprocessor):
            def fit(self, *args, **kwargs):
                return self
            def transform(self, *args, **kwargs):
                return "uppercase"
        
        # Act
        PreprocessorStepRegistry.register("scaler", LowerCaseStep)
        PreprocessorStepRegistry.register("SCALER", UpperCaseStep)
        
        # Assert
        lower_step = PreprocessorStepRegistry.preprocessor_steps["scaler"]
        upper_step = PreprocessorStepRegistry.preprocessor_steps["SCALER"]
        
        assert lower_step == LowerCaseStep
        assert upper_step == UpperCaseStep
        assert lower_step != upper_step


class TestPreprocessorStepRegistryIntegration:
    """Test PreprocessorStepRegistry integration scenarios."""
    
    def test_registry_preserves_inheritance_structure(self):
        """Test that registry preserves class inheritance information."""
        # Arrange
        class SpecializedStep(BasePreprocessor):
            special_attribute = "special"
            
            def fit(self, *args, **kwargs):
                return self
                
            def transform(self, *args, **kwargs):
                return Mock()
                
            def special_method(self):
                return "special_functionality"
        
        # Act
        PreprocessorStepRegistry.register("specialized", SpecializedStep)
        retrieved_class = PreprocessorStepRegistry.preprocessor_steps["specialized"]
        instance = PreprocessorStepRegistry.create("specialized")
        
        # Assert
        assert retrieved_class == SpecializedStep
        assert isinstance(instance, BasePreprocessor)
        assert isinstance(instance, SpecializedStep)
        assert hasattr(instance, "special_attribute")
        assert instance.special_attribute == "special"
        assert hasattr(instance, "special_method")
        assert instance.special_method() == "special_functionality"
    
    def test_registry_handles_complex_initialization(self):
        """Test that registry handles steps with complex initialization."""
        # Arrange
        class ComplexStep(BasePreprocessor):
            def __init__(self, required_param, optional_param=None, *args, **kwargs):
                if not required_param:
                    raise ValueError("required_param is mandatory")
                self.required_param = required_param
                self.optional_param = optional_param
                self.args = args
                self.kwargs = kwargs
            
            def fit(self, *args, **kwargs):
                return self
                
            def transform(self, *args, **kwargs):
                return Mock()
        
        PreprocessorStepRegistry.register("complex", ComplexStep)
        
        # Act & Assert - should work with required param
        instance1 = PreprocessorStepRegistry.create("complex", required_param="required_value")
        assert instance1.required_param == "required_value"
        assert instance1.optional_param is None
        
        # Act & Assert - should work with all params
        instance2 = PreprocessorStepRegistry.create(
            "complex", 
            required_param="req_val", 
            optional_param="opt_val",
            extra_arg="extra",
            kwarg1="kw1"
        )
        assert instance2.required_param == "req_val"
        assert instance2.optional_param == "opt_val"
        assert instance2.kwargs["kwarg1"] == "kw1"
        
        # Act & Assert - should fail without required param
        with pytest.raises((ValueError, TypeError)):
            PreprocessorStepRegistry.create("complex")