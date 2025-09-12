"""
Calibration Registry Unit Tests
Testing calibration method registration and creation
"""

import pytest
import numpy as np
from src.components.calibration.registry import CalibrationRegistry
from src.interface import BaseCalibrator


class MockCalibrator(BaseCalibrator):
    """Mock calibrator for testing"""
    
    def __init__(self):
        super().__init__()
        self._is_fitted = False
    
    def fit(self, y_prob, y_true):
        self._is_fitted = True
        return self
    
    def transform(self, y_prob):
        if not self._is_fitted:
            raise ValueError("Not fitted")
        return y_prob  # Identity transformation for testing
    
    @property
    def supports_multiclass(self):
        return True


class InvalidCalibrator:
    """Invalid calibrator (doesn't inherit BaseCalibrator)"""
    pass


class TestCalibrationRegistry:
    """Test CalibrationRegistry functionality"""
    
    def setup_method(self):
        """Reset registry before each test"""
        CalibrationRegistry.calibrators.clear()
    
    def test_register_valid_calibrator(self):
        """Test registering a valid calibrator class"""
        # When: Register a valid calibrator
        CalibrationRegistry.register("mock", MockCalibrator)
        
        # Then: Calibrator should be registered
        assert "mock" in CalibrationRegistry.calibrators
        assert CalibrationRegistry.calibrators["mock"] == MockCalibrator
    
    def test_register_invalid_calibrator_raises_error(self):
        """Test that registering invalid calibrator raises TypeError"""
        # When/Then: Register an invalid calibrator should raise error
        with pytest.raises(TypeError, match="must be a subclass of BaseCalibrator"):
            CalibrationRegistry.register("invalid", InvalidCalibrator)
    
    def test_create_registered_calibrator(self):
        """Test creating instance of registered calibrator"""
        # Given: Register a calibrator
        CalibrationRegistry.register("mock", MockCalibrator)
        
        # When: Create instance
        calibrator = CalibrationRegistry.create("mock")
        
        # Then: Instance should be of correct type
        assert isinstance(calibrator, MockCalibrator)
        assert isinstance(calibrator, BaseCalibrator)
    
    def test_create_unregistered_calibrator_raises_error(self):
        """Test that creating unregistered calibrator raises ValueError"""
        # When/Then: Create unregistered calibrator should raise error
        with pytest.raises(ValueError, match="Unknown calibration method: 'nonexistent'"):
            CalibrationRegistry.create("nonexistent")
    
    def test_get_available_methods(self):
        """Test getting list of available methods"""
        # Given: Register multiple calibrators
        CalibrationRegistry.register("mock1", MockCalibrator)
        CalibrationRegistry.register("mock2", MockCalibrator)
        
        # When: Get available methods
        methods = CalibrationRegistry.get_available_methods()
        
        # Then: Should return all registered methods
        assert set(methods) == {"mock1", "mock2"}
    
    def test_get_calibrator_class(self):
        """Test getting calibrator class by method name"""
        # Given: Register a calibrator
        CalibrationRegistry.register("mock", MockCalibrator)
        
        # When: Get calibrator class
        calibrator_class = CalibrationRegistry.get_calibrator_class("mock")
        
        # Then: Should return correct class
        assert calibrator_class == MockCalibrator
    
    def test_get_calibrator_class_raises_error_for_unknown_method(self):
        """Test that getting unknown calibrator class raises ValueError"""
        # When/Then: Get unknown calibrator class should raise error
        with pytest.raises(ValueError, match="Unknown calibration method: 'unknown'"):
            CalibrationRegistry.get_calibrator_class("unknown")
    
    def test_auto_registration_beta_calibration(self):
        """Test that Beta Calibration is auto-registered"""
        # Re-register after setup clearing for this test
        import src.components.calibration.modules.beta_calibration
        
        # Then: Beta Calibration should be registered
        assert "beta" in CalibrationRegistry.get_available_methods()
        
        # And: Should be able to create instance
        calibrator = CalibrationRegistry.create("beta")
        assert calibrator is not None
        assert calibrator.supports_multiclass  # Beta supports multiclass
    
    def test_auto_registration_isotonic_regression(self):
        """Test that Isotonic Regression is auto-registered"""
        # Re-register after setup clearing for this test
        import src.components.calibration.modules.isotonic_regression
        
        # Then: Isotonic Regression should be registered
        assert "isotonic" in CalibrationRegistry.get_available_methods()
        
        # And: Should be able to create instance
        calibrator = CalibrationRegistry.create("isotonic")
        assert calibrator is not None
        assert calibrator.supports_multiclass  # Isotonic supports multiclass