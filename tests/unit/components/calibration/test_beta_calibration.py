"""
Beta Calibration Unit Tests
Testing Beta Calibration implementation for binary and multiclass classification
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.components.calibration.modules.beta_calibration import BetaCalibration


class TestBetaCalibration:
    """Test BetaCalibration calibrator functionality"""

    def setup_method(self):
        """Setup test data for binary and multiclass scenarios"""
        # Binary classification data
        X_bin, y_bin = make_classification(
            n_samples=1000, n_features=10, n_classes=2, random_state=42
        )
        X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(
            X_bin, y_bin, test_size=0.3, random_state=42
        )

        model_bin = RandomForestClassifier(n_estimators=50, random_state=42)
        model_bin.fit(X_bin_train, y_bin_train)
        y_prob_bin = model_bin.predict_proba(X_bin_test)[:, 1]  # Positive class only

        # Multiclass classification data
        X_multi, y_multi = make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=3,
            n_informative=10,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
            X_multi, y_multi, test_size=0.3, random_state=42
        )

        model_multi = RandomForestClassifier(n_estimators=50, random_state=42)
        model_multi.fit(X_multi_train, y_multi_train)
        y_prob_multi = model_multi.predict_proba(X_multi_test)  # All class probabilities

        # Store test data
        self.y_true_bin = y_bin_test
        self.y_prob_bin = y_prob_bin
        self.y_true_multi = y_multi_test
        self.y_prob_multi = y_prob_multi

    def test_beta_calibration_initialization(self):
        """Test BetaCalibration initialization"""
        # When: Create BetaCalibration instance
        calibrator = BetaCalibration()

        # Then: Should be properly initialized
        assert not calibrator._is_fitted
        assert calibrator.supports_multiclass
        assert calibrator.parameters is None
        assert calibrator._n_classes is None

    def test_fit_with_binary_data_1d(self):
        """Test fitting with binary classification data (1D probabilities)"""
        # Given: BetaCalibration instance
        calibrator = BetaCalibration()

        # When: Fit with binary data
        result = calibrator.fit(self.y_prob_bin, self.y_true_bin)

        # Then: Should be fitted and return self
        assert calibrator._is_fitted
        assert calibrator._n_classes == 2
        assert result is calibrator
        assert isinstance(calibrator.parameters, dict)
        assert "a" in calibrator.parameters
        assert "b" in calibrator.parameters

    def test_fit_with_multiclass_data_2d(self):
        """Test fitting with multiclass classification data (2D probabilities)"""
        # Given: BetaCalibration instance
        calibrator = BetaCalibration()

        # When: Fit with multiclass data
        result = calibrator.fit(self.y_prob_multi, self.y_true_multi)

        # Then: Should be fitted and return self
        assert calibrator._is_fitted
        assert calibrator._n_classes == 3
        assert result is calibrator
        assert isinstance(calibrator.parameters, list)
        assert len(calibrator.parameters) == 3  # One calibrator per class
        for params in calibrator.parameters:
            assert "a" in params
            assert "b" in params

    def test_fit_with_invalid_probability_range_raises_error(self):
        """Test that fitting with invalid probability range raises ValueError"""
        # Given: Invalid probability range
        y_prob_invalid = np.array([0.5, 1.2, -0.1, 0.8])
        y_true_binary = np.array([1, 0, 0, 1])
        calibrator = BetaCalibration()

        # When/Then: Fit with invalid probabilities should raise error
        with pytest.raises(ValueError, match="확률값은 \\[0, 1\\] 범위에 있어야 합니다"):
            calibrator.fit(y_prob_invalid, y_true_binary)

    def test_fit_with_mismatched_lengths_raises_error(self):
        """Test that fitting with mismatched input lengths raises ValueError"""
        # Given: Mismatched lengths
        y_prob_short = np.array([0.3, 0.7])
        y_true_long = np.array([1, 0, 1])
        calibrator = BetaCalibration()

        # When/Then: Fit with mismatched lengths should raise error
        with pytest.raises(ValueError, match="y_prob와 y_true의 길이가 다릅니다"):
            calibrator.fit(y_prob_short, y_true_long)

    def test_fit_with_single_class_raises_error(self):
        """Test that fitting with single class raises ValueError"""
        # Given: Single class data
        y_prob_single = np.array([0.3, 0.7, 0.5])
        y_true_single = np.array([1, 1, 1])  # All same class
        calibrator = BetaCalibration()

        # When/Then: Fit with single class should raise error
        with pytest.raises(ValueError, match="최소 2개 클래스가 필요합니다"):
            calibrator.fit(y_prob_single, y_true_single)

    def test_transform_binary_after_fitting(self):
        """Test transforming binary probabilities after fitting"""
        # Given: Fitted binary calibrator
        calibrator = BetaCalibration()
        calibrator.fit(self.y_prob_bin, self.y_true_bin)

        # When: Transform probabilities
        calibrated_probs = calibrator.transform(self.y_prob_bin)

        # Then: Should return calibrated probabilities
        assert isinstance(calibrated_probs, np.ndarray)
        assert calibrated_probs.ndim == 1  # Binary should return 1D
        assert len(calibrated_probs) == len(self.y_prob_bin)
        assert np.all((calibrated_probs >= 0) & (calibrated_probs <= 1))

    def test_transform_multiclass_after_fitting(self):
        """Test transforming multiclass probabilities after fitting"""
        # Given: Fitted multiclass calibrator
        calibrator = BetaCalibration()
        calibrator.fit(self.y_prob_multi, self.y_true_multi)

        # When: Transform probabilities
        calibrated_probs = calibrator.transform(self.y_prob_multi)

        # Then: Should return calibrated probabilities
        assert isinstance(calibrated_probs, np.ndarray)
        assert calibrated_probs.ndim == 2  # Multiclass should return 2D
        assert calibrated_probs.shape == self.y_prob_multi.shape
        assert np.all((calibrated_probs >= 0) & (calibrated_probs <= 1))

        # And: Probabilities should sum to 1 for each sample (after normalization)
        row_sums = calibrated_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    def test_transform_before_fitting_raises_error(self):
        """Test that transforming before fitting raises ValueError"""
        # Given: Unfitted calibrator
        calibrator = BetaCalibration()

        # When/Then: Transform before fit should raise error
        with pytest.raises(ValueError, match="fit\\(\\)을 먼저 호출해야 합니다"):
            calibrator.transform(self.y_prob_bin)

    def test_fit_transform_convenience_method(self):
        """Test fit_transform convenience method"""
        # Given: BetaCalibration instance
        calibrator = BetaCalibration()

        # When: Use fit_transform with binary data
        calibrated_probs = calibrator.fit_transform(self.y_prob_bin, self.y_true_bin)

        # Then: Should be fitted and return calibrated probabilities
        assert calibrator._is_fitted
        assert isinstance(calibrated_probs, np.ndarray)
        assert len(calibrated_probs) == len(self.y_prob_bin)
        assert np.all((calibrated_probs >= 0) & (calibrated_probs <= 1))

    def test_supports_multiclass_property(self):
        """Test supports_multiclass property"""
        # Given: BetaCalibration instance
        calibrator = BetaCalibration()

        # Then: Should support multiclass
        assert calibrator.supports_multiclass

    def test_serialization_support(self):
        """Test MLflow serialization support"""
        # Given: Fitted calibrator
        calibrator = BetaCalibration()
        calibrator.fit(self.y_prob_bin, self.y_true_bin)

        # When: Serialize and deserialize
        state = calibrator.__getstate__()
        new_calibrator = BetaCalibration()
        new_calibrator.__setstate__(state)

        # Then: Should maintain fitted state
        assert new_calibrator._is_fitted
        assert new_calibrator._n_classes == calibrator._n_classes
        assert new_calibrator.parameters == calibrator.parameters

        # And: Should produce same results
        original_result = calibrator.transform(self.y_prob_bin)
        restored_result = new_calibrator.transform(self.y_prob_bin)
        np.testing.assert_array_almost_equal(original_result, restored_result)

    def test_beta_parameters_reasonable(self):
        """Test that Beta parameters are reasonable"""
        # Given: Fitted calibrator
        calibrator = BetaCalibration()
        calibrator.fit(self.y_prob_bin, self.y_true_bin)

        # Then: Parameters should be reasonable
        params = calibrator.parameters
        assert params["a"] > 0  # Scale parameter should be positive
        assert -10 <= params["b"] <= 10  # Shift parameter should be bounded

        # And: When a=1, b=0, should be close to identity transformation
        identity_calibrator = BetaCalibration()
        identity_calibrator.parameters = {"a": 1.0, "b": 0.0}
        identity_calibrator._is_fitted = True
        identity_calibrator._n_classes = 2

        test_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        identity_result = identity_calibrator.transform(test_probs)

        # Should be very close to original (sigmoid(1*logit(p) + 0) ≈ p)
        np.testing.assert_allclose(identity_result, test_probs, rtol=1e-10)
