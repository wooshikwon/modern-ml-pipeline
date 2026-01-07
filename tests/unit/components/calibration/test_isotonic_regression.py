"""
Isotonic Regression Calibrator Unit Tests
Testing Isotonic Regression implementation for binary and multiclass classification
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.components.calibration.modules.isotonic_regression import IsotonicCalibration


class TestIsotonicCalibration:
    """Test IsotonicCalibration calibrator functionality"""

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

    def test_isotonic_calibration_initialization(self):
        """Test IsotonicCalibration initialization"""
        # When: Create IsotonicCalibration instance
        calibrator = IsotonicCalibration()

        # Then: Should be properly initialized
        assert not calibrator._is_fitted
        assert calibrator.supports_multiclass
        assert calibrator.calibrator is None
        assert calibrator._n_classes is None

    def test_fit_with_binary_data_1d(self):
        """Test fitting with binary classification data (1D probabilities)"""
        # Given: IsotonicCalibration instance
        calibrator = IsotonicCalibration()

        # When: Fit with binary data
        result = calibrator.fit(self.y_prob_bin, self.y_true_bin)

        # Then: Should be fitted and return self
        assert calibrator._is_fitted
        assert calibrator._n_classes == 2
        assert result is calibrator
        assert hasattr(calibrator.calibrator, "transform")  # Single IsotonicRegression

    def test_fit_with_multiclass_data_2d(self):
        """Test fitting with multiclass classification data (2D probabilities)"""
        # Given: IsotonicCalibration instance
        calibrator = IsotonicCalibration()

        # When: Fit with multiclass data
        result = calibrator.fit(self.y_prob_multi, self.y_true_multi)

        # Then: Should be fitted and return self
        assert calibrator._is_fitted
        assert calibrator._n_classes == 3
        assert result is calibrator
        assert isinstance(calibrator.calibrator, list)
        assert len(calibrator.calibrator) == 3  # One calibrator per class

    def test_fit_with_invalid_probability_range_raises_error(self):
        """Test that fitting with invalid probability range raises ValueError"""
        # Given: Invalid probability range
        y_prob_invalid = np.array([0.5, 1.2, -0.1, 0.8])
        y_true_binary = np.array([1, 0, 0, 1])
        calibrator = IsotonicCalibration()

        # When/Then: Fit with invalid probabilities should raise error
        with pytest.raises(ValueError, match="확률값은 \\[0, 1\\] 범위에 있어야 합니다"):
            calibrator.fit(y_prob_invalid, y_true_binary)

    def test_fit_with_mismatched_lengths_raises_error(self):
        """Test that fitting with mismatched input lengths raises ValueError"""
        # Given: Mismatched lengths
        y_prob_short = np.array([0.3, 0.7])
        y_true_long = np.array([1, 0, 1])
        calibrator = IsotonicCalibration()

        # When/Then: Fit with mismatched lengths should raise error
        with pytest.raises(ValueError, match="y_prob와 y_true의 길이가 다릅니다"):
            calibrator.fit(y_prob_short, y_true_long)

    def test_fit_with_single_class_raises_error(self):
        """Test that fitting with single class raises ValueError"""
        # Given: Single class data
        y_prob_single = np.array([0.3, 0.7, 0.5])
        y_true_single = np.array([1, 1, 1])  # All same class
        calibrator = IsotonicCalibration()

        # When/Then: Fit with single class should raise error
        with pytest.raises(ValueError, match="최소 2개 클래스가 필요합니다"):
            calibrator.fit(y_prob_single, y_true_single)

    def test_fit_with_binary_data_but_multiclass_labels_raises_error(self):
        """Test that fitting with 1D probabilities but multiclass labels raises ValueError"""
        # Given: 1D probabilities but 3 classes
        y_prob_1d = np.array([0.3, 0.7, 0.5, 0.8])
        y_true_multiclass = np.array([0, 1, 2, 1])  # 3 classes
        calibrator = IsotonicCalibration()

        # When/Then: Should raise error
        with pytest.raises(
            ValueError, match="이진 확률값\\(1D\\)이지만 3개 클래스가 발견되었습니다"
        ):
            calibrator.fit(y_prob_1d, y_true_multiclass)

    def test_fit_with_wrong_number_of_probability_columns_raises_error(self):
        """Test that fitting with wrong number of probability columns raises ValueError"""
        # Given: 2D probabilities with 2 columns but y_true has 3 classes
        y_prob_2_cols = np.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]])  # 2 columns, 3 samples
        y_true_3_class = np.array([0, 1, 2])  # 3 classes (0, 1, 2), 3 samples
        calibrator = IsotonicCalibration()

        # When/Then: Should raise error (2 probability columns but 3 actual classes)
        with pytest.raises(
            ValueError, match="확률 행렬의 클래스 수\\(2\\)와 실제 클래스 수\\(3\\)가 다릅니다"
        ):
            calibrator.fit(y_prob_2_cols, y_true_3_class)

    def test_transform_binary_after_fitting(self):
        """Test transforming binary probabilities after fitting"""
        # Given: Fitted binary calibrator
        calibrator = IsotonicCalibration()
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
        calibrator = IsotonicCalibration()
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
        calibrator = IsotonicCalibration()

        # When/Then: Transform before fit should raise error
        with pytest.raises(ValueError, match="fit\\(\\)을 먼저 호출해야 합니다"):
            calibrator.transform(self.y_prob_bin)

    def test_transform_wrong_dimensionality_raises_error(self):
        """Test that transforming with wrong dimensionality raises ValueError"""
        # Given: Binary-fitted calibrator
        calibrator = IsotonicCalibration()
        calibrator.fit(self.y_prob_bin, self.y_true_bin)

        # When/Then: Transform with 2D probabilities should raise error
        with pytest.raises(
            ValueError, match="다중 클래스 확률값이지만 이진 분류 calibrator로 학습되었습니다"
        ):
            calibrator.transform(self.y_prob_multi)

        # And: Given multiclass-fitted calibrator
        calibrator_multi = IsotonicCalibration()
        calibrator_multi.fit(self.y_prob_multi, self.y_true_multi)

        # When/Then: Transform with 1D probabilities should raise error
        with pytest.raises(
            ValueError, match="이진 확률값이지만 다중 클래스 calibrator로 학습되었습니다"
        ):
            calibrator_multi.transform(self.y_prob_bin)

    def test_transform_wrong_number_of_classes_raises_error(self):
        """Test that transforming with wrong number of classes raises ValueError"""
        # Given: Multiclass-fitted calibrator (3 classes)
        calibrator = IsotonicCalibration()
        calibrator.fit(self.y_prob_multi, self.y_true_multi)

        # When/Then: Transform with wrong number of classes should raise error
        wrong_shape_probs = np.array([[0.5, 0.5], [0.3, 0.7]])  # Only 2 classes
        with pytest.raises(
            ValueError, match="확률 행렬의 클래스 수\\(2\\)와 학습된 클래스 수\\(3\\)가 다릅니다"
        ):
            calibrator.transform(wrong_shape_probs)

    def test_fit_transform_convenience_method(self):
        """Test fit_transform convenience method"""
        # Given: IsotonicCalibration instance
        calibrator = IsotonicCalibration()

        # When: Use fit_transform with binary data
        calibrated_probs = calibrator.fit_transform(self.y_prob_bin, self.y_true_bin)

        # Then: Should be fitted and return calibrated probabilities
        assert calibrator._is_fitted
        assert isinstance(calibrated_probs, np.ndarray)
        assert len(calibrated_probs) == len(self.y_prob_bin)
        assert np.all((calibrated_probs >= 0) & (calibrated_probs <= 1))

    def test_supports_multiclass_property(self):
        """Test supports_multiclass property"""
        # Given: IsotonicCalibration instance
        calibrator = IsotonicCalibration()

        # Then: Should support multiclass
        assert calibrator.supports_multiclass

    def test_calibration_preserves_monotonicity(self):
        """Test that isotonic calibration preserves monotonic relationship"""
        # Given: Fitted binary calibrator
        calibrator = IsotonicCalibration()
        calibrator.fit(self.y_prob_bin, self.y_true_bin)

        # When: Apply calibration to sorted probabilities
        sorted_probs = np.sort(self.y_prob_bin)
        calibrated_sorted = calibrator.transform(sorted_probs)

        # Then: Calibrated probabilities should maintain monotonic order
        # (Isotonic regression ensures this)
        diff = np.diff(calibrated_sorted)
        assert np.all(diff >= 0), "Isotonic calibration should preserve monotonic order"

    def test_serialization_support(self):
        """Test MLflow serialization support"""
        # Given: Fitted calibrator
        calibrator = IsotonicCalibration()
        calibrator.fit(self.y_prob_bin, self.y_true_bin)

        # When: Serialize and deserialize
        state = calibrator.__getstate__()
        new_calibrator = IsotonicCalibration()
        new_calibrator.__setstate__(state)

        # Then: Should maintain fitted state
        assert new_calibrator._is_fitted
        assert new_calibrator._n_classes == calibrator._n_classes

        # And: Should produce same results
        original_result = calibrator.transform(self.y_prob_bin)
        restored_result = new_calibrator.transform(self.y_prob_bin)
        np.testing.assert_array_almost_equal(original_result, restored_result)
