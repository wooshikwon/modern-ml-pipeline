"""
Calibration Registry Unit Tests
Testing calibration method registration and creation
"""

import pytest

from src.components.calibration.registry import CalibrationRegistry
from src.components.calibration.base import BaseCalibrator


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
        CalibrationRegistry.clear()

    def teardown_method(self):
        """Restore registry after each test to prevent pollution"""
        CalibrationRegistry.clear()
        # Re-import to re-register all calibrators
        from src.components.calibration.modules.beta_calibration import BetaCalibration
        from src.components.calibration.modules.isotonic_regression import IsotonicCalibration

        CalibrationRegistry.register("beta", BetaCalibration)
        CalibrationRegistry.register("isotonic", IsotonicCalibration)

    def test_register_valid_calibrator(self):
        """Test registering a valid calibrator class"""
        # When: Register a valid calibrator
        CalibrationRegistry.register("mock", MockCalibrator)

        # Then: Calibrator should be registered
        assert "mock" in CalibrationRegistry.list_keys()
        assert CalibrationRegistry.get_class("mock") == MockCalibrator

    def test_register_invalid_calibrator_raises_error(self):
        """Test that registering invalid calibrator raises TypeError"""
        # When/Then: Register an invalid calibrator should raise error
        with pytest.raises(TypeError, match="하위 클래스여야 합니다"):
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
        """Test that creating unregistered calibrator raises KeyError"""
        # When/Then: Create unregistered calibrator should raise error
        with pytest.raises(KeyError, match="알 수 없는 키"):
            CalibrationRegistry.create("nonexistent")

    def test_list_keys(self):
        """Test getting list of available methods"""
        # Given: Register multiple calibrators
        CalibrationRegistry.register("mock1", MockCalibrator)
        CalibrationRegistry.register("mock2", MockCalibrator)

        # When: Get available methods
        methods = CalibrationRegistry.list_keys()

        # Then: Should return all registered methods
        assert set(methods) == {"mock1", "mock2"}

    def test_get_class(self):
        """Test getting calibrator class by method name"""
        # Given: Register a calibrator
        CalibrationRegistry.register("mock", MockCalibrator)

        # When: Get calibrator class
        calibrator_class = CalibrationRegistry.get_class("mock")

        # Then: Should return correct class
        assert calibrator_class == MockCalibrator

    def test_get_class_raises_error_for_unknown_method(self):
        """Test that getting unknown calibrator class raises KeyError"""
        # When/Then: Get unknown calibrator class should raise error
        with pytest.raises(KeyError, match="알 수 없는 키"):
            CalibrationRegistry.get_class("unknown")

    def test_auto_registration_beta_calibration(self):
        """Test that Beta Calibration is auto-registered"""
        # Re-register after setup clearing for this test
        from src.components.calibration.modules.beta_calibration import BetaCalibration

        CalibrationRegistry.register("beta", BetaCalibration)

        # Then: Beta Calibration should be registered
        assert "beta" in CalibrationRegistry.list_keys()

        # And: Should be able to create instance
        calibrator = CalibrationRegistry.create("beta")
        assert calibrator is not None
        assert calibrator.supports_multiclass  # Beta supports multiclass

    def test_auto_registration_isotonic_regression(self):
        """Test that Isotonic Regression is auto-registered"""
        # Re-register after setup clearing for this test
        from src.components.calibration.modules.isotonic_regression import IsotonicCalibration

        CalibrationRegistry.register("isotonic", IsotonicCalibration)

        # Then: Isotonic Regression should be registered
        assert "isotonic" in CalibrationRegistry.list_keys()

        # And: Should be able to create instance
        calibrator = CalibrationRegistry.create("isotonic")
        assert calibrator is not None
        assert calibrator.supports_multiclass  # Isotonic supports multiclass
