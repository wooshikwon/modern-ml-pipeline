"""
Unit tests for Time Series Wrappers.
Tests ARIMA and Exponential Smoothing wrappers with sklearn interface.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock, MagicMock

from src.models.custom.timeseries_wrappers import ARIMAWrapper, ExponentialSmoothingWrapper


class TestARIMAWrapperInitialization:
    """Test ARIMA wrapper initialization."""
    
    def test_arima_wrapper_default_initialization(self):
        """Test default parameter initialization."""
        model = ARIMAWrapper()
        
        # Assert default parameters
        assert model.order_p == 1
        assert model.order_d == 1
        assert model.order_q == 1
        
        # Assert internal state
        assert model.model_ is None
        assert model.fitted_model_ is None
        assert model._is_fitted is False
    
    def test_arima_wrapper_custom_initialization(self):
        """Test initialization with custom parameters."""
        model = ARIMAWrapper(order_p=2, order_d=0, order_q=3)
        
        assert model.order_p == 2
        assert model.order_d == 0
        assert model.order_q == 3
        assert model.model_ is None
        assert model.fitted_model_ is None
        assert model._is_fitted is False


class TestARIMAWrapperFit:
    """Test ARIMA wrapper fitting process."""
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_arima_fit_success(self, mock_arima_class, mock_logger):
        """Test successful ARIMA fitting."""
        # Setup
        model = ARIMAWrapper(order_p=1, order_d=1, order_q=1)
        
        X = pd.DataFrame(np.random.randn(100, 3))  # Features (ignored)
        y = pd.Series(np.random.randn(100))        # Time series data
        
        # Mock ARIMA model and fitting
        mock_arima_instance = Mock()
        mock_fitted_model = Mock()
        mock_arima_instance.fit.return_value = mock_fitted_model
        mock_arima_class.return_value = mock_arima_instance
        
        # Execute
        result = model.fit(X, y)
        
        # Assertions
        assert result is model  # Returns self
        assert model._is_fitted is True
        assert model.model_ == mock_arima_instance
        assert model.fitted_model_ == mock_fitted_model
        
        # Check ARIMA was called with correct parameters
        mock_arima_class.assert_called_once_with(y, order=(1, 1, 1))
        mock_arima_instance.fit.assert_called_once()
        
        # Check logging
        mock_logger.info.assert_called()
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    def test_arima_fit_statsmodels_import_error(self, mock_logger):
        """Test ARIMA fit when statsmodels is not available."""
        model = ARIMAWrapper()
        
        X = pd.DataFrame(np.random.randn(50, 2))
        y = pd.Series(np.random.randn(50))
        
        # Mock import error
        with patch('builtins.__import__', side_effect=ImportError("No module named 'statsmodels'")):
            with pytest.raises(ImportError, match="statsmodels is required"):
                model.fit(X, y)
        
        # Should log error
        mock_logger.error.assert_called()
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_arima_fit_model_error(self, mock_arima_class, mock_logger):
        """Test ARIMA fit when model fitting fails."""
        model = ARIMAWrapper()
        
        X = pd.DataFrame(np.random.randn(10, 1))
        y = pd.Series(np.random.randn(10))
        
        # Mock ARIMA fitting error
        mock_arima_instance = Mock()
        mock_arima_instance.fit.side_effect = ValueError("ARIMA fitting failed")
        mock_arima_class.return_value = mock_arima_instance
        
        with pytest.raises(ValueError, match="ARIMA fitting failed"):
            model.fit(X, y)
        
        # Should log error and not mark as fitted
        mock_logger.error.assert_called()
        assert model._is_fitted is False
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_arima_fit_different_orders(self, mock_arima_class, mock_logger):
        """Test ARIMA fitting with different order parameters."""
        model = ARIMAWrapper(order_p=3, order_d=2, order_q=1)
        
        X = pd.DataFrame(np.random.randn(200, 5))
        y = pd.Series(np.random.randn(200))
        
        # Mock ARIMA components
        mock_arima_instance = Mock()
        mock_fitted_model = Mock()
        mock_arima_instance.fit.return_value = mock_fitted_model
        mock_arima_class.return_value = mock_arima_instance
        
        # Execute
        result = model.fit(X, y)
        
        # Check order was passed correctly
        mock_arima_class.assert_called_once_with(y, order=(3, 2, 1))
        assert result is model
        assert model._is_fitted is True


class TestARIMAWrapperPredict:
    """Test ARIMA wrapper prediction."""
    
    def test_arima_predict_not_fitted_error(self):
        """Test prediction error when model is not fitted."""
        model = ARIMAWrapper()
        
        X = pd.DataFrame(np.random.randn(10, 2))
        
        with pytest.raises(ValueError, match="모델이 학습되지 않았습니다"):
            model.predict(X)
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    def test_arima_predict_success(self, mock_logger):
        """Test successful ARIMA prediction."""
        # Setup fitted model
        model = ARIMAWrapper()
        model._is_fitted = True
        
        # Mock fitted model with forecast method
        mock_fitted_model = Mock()
        mock_forecast_result = np.array([1.5, 2.3, 1.8, 2.1, 1.9])
        mock_fitted_model.forecast.return_value = mock_forecast_result
        model.fitted_model_ = mock_fitted_model
        
        # Test data
        X_test = pd.DataFrame(np.random.randn(5, 3))  # 5 steps forecast
        
        # Execute
        predictions = model.predict(X_test)
        
        # Assertions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5
        np.testing.assert_array_equal(predictions, mock_forecast_result)
        
        # Check forecast was called with correct steps
        mock_fitted_model.forecast.assert_called_once_with(steps=5)
        
        # Check logging
        mock_logger.info.assert_called()
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    def test_arima_predict_forecast_error(self, mock_logger):
        """Test ARIMA prediction when forecast fails."""
        # Setup fitted model
        model = ARIMAWrapper()
        model._is_fitted = True
        
        # Mock fitted model with forecast error
        mock_fitted_model = Mock()
        mock_fitted_model.forecast.side_effect = RuntimeError("Forecast failed")
        model.fitted_model_ = mock_fitted_model
        
        X_test = pd.DataFrame(np.random.randn(3, 2))
        
        with pytest.raises(RuntimeError, match="Forecast failed"):
            model.predict(X_test)
        
        # Should log error
        mock_logger.error.assert_called()
    
    def test_arima_predict_different_lengths(self):
        """Test ARIMA prediction with different forecast lengths."""
        # Setup fitted model
        model = ARIMAWrapper()
        model._is_fitted = True
        
        # Mock fitted model
        mock_fitted_model = Mock()
        model.fitted_model_ = mock_fitted_model
        
        # Test different lengths
        test_cases = [1, 5, 10, 20]
        
        for length in test_cases:
            mock_fitted_model.reset_mock()
            mock_fitted_model.forecast.return_value = np.random.randn(length)
            
            X_test = pd.DataFrame(np.random.randn(length, 2))
            predictions = model.predict(X_test)
            
            assert len(predictions) == length
            mock_fitted_model.forecast.assert_called_once_with(steps=length)


class TestExponentialSmoothingWrapperInitialization:
    """Test Exponential Smoothing wrapper initialization."""
    
    def test_exponential_smoothing_default_initialization(self):
        """Test default parameter initialization."""
        model = ExponentialSmoothingWrapper()
        
        # Assert default parameters
        assert model.trend == "add"
        assert model.seasonal is None
        assert model.seasonal_periods == 12
        
        # Assert internal state
        assert model.fitted_model_ is None
        assert model._is_fitted is False
    
    def test_exponential_smoothing_custom_initialization(self):
        """Test initialization with custom parameters."""
        model = ExponentialSmoothingWrapper(
            trend="mul",
            seasonal="add",
            seasonal_periods=52
        )
        
        assert model.trend == "mul"
        assert model.seasonal == "add"
        assert model.seasonal_periods == 52
        assert model.fitted_model_ is None
        assert model._is_fitted is False
    
    def test_exponential_smoothing_no_trend_no_seasonal(self):
        """Test initialization with no trend and no seasonal components."""
        model = ExponentialSmoothingWrapper(
            trend=None,
            seasonal=None
        )
        
        assert model.trend is None
        assert model.seasonal is None


class TestExponentialSmoothingWrapperFit:
    """Test Exponential Smoothing wrapper fitting."""
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    @patch('statsmodels.tsa.holtwinters.ExponentialSmoothing')
    def test_exponential_smoothing_fit_success(self, mock_es_class, mock_logger):
        """Test successful Exponential Smoothing fitting."""
        # Setup
        model = ExponentialSmoothingWrapper(trend="add", seasonal=None)
        
        X = pd.DataFrame(np.random.randn(50, 2))
        y = pd.Series(np.random.randn(50))
        
        # Mock ExponentialSmoothing model
        mock_es_instance = Mock()
        mock_fitted_model = Mock()
        mock_es_instance.fit.return_value = mock_fitted_model
        mock_es_class.return_value = mock_es_instance
        
        # Execute
        result = model.fit(X, y)
        
        # Assertions
        assert result is model  # Returns self
        assert model._is_fitted is True
        assert model.fitted_model_ == mock_fitted_model
        
        # Check ExponentialSmoothing was called with correct parameters
        mock_es_class.assert_called_once_with(
            y,
            trend="add",
            seasonal=None,
            seasonal_periods=None  # Should be None when seasonal is None
        )
        mock_es_instance.fit.assert_called_once()
        
        # Check logging
        mock_logger.info.assert_called()
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    @patch('statsmodels.tsa.holtwinters.ExponentialSmoothing')
    def test_exponential_smoothing_fit_with_seasonal(self, mock_es_class, mock_logger):
        """Test Exponential Smoothing fitting with seasonal component."""
        model = ExponentialSmoothingWrapper(
            trend="add",
            seasonal="mul",
            seasonal_periods=12
        )
        
        X = pd.DataFrame(np.random.randn(60, 1))
        y = pd.Series(np.random.randn(60))
        
        # Mock components
        mock_es_instance = Mock()
        mock_fitted_model = Mock()
        mock_es_instance.fit.return_value = mock_fitted_model
        mock_es_class.return_value = mock_es_instance
        
        # Execute
        result = model.fit(X, y)
        
        # Check seasonal_periods was passed correctly
        mock_es_class.assert_called_once_with(
            y,
            trend="add",
            seasonal="mul",
            seasonal_periods=12
        )
        assert result is model
        assert model._is_fitted is True
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    def test_exponential_smoothing_fit_statsmodels_import_error(self, mock_logger):
        """Test ExponentialSmoothing fit when statsmodels is not available."""
        model = ExponentialSmoothingWrapper()
        
        X = pd.DataFrame(np.random.randn(30, 2))
        y = pd.Series(np.random.randn(30))
        
        # Mock import error
        with patch('builtins.__import__', side_effect=ImportError("No module named 'statsmodels'")):
            with pytest.raises(ImportError, match="statsmodels is required"):
                model.fit(X, y)
        
        # Should log error
        mock_logger.error.assert_called()
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    @patch('statsmodels.tsa.holtwinters.ExponentialSmoothing')
    def test_exponential_smoothing_fit_model_error(self, mock_es_class, mock_logger):
        """Test ExponentialSmoothing fit when model fitting fails."""
        model = ExponentialSmoothingWrapper()
        
        X = pd.DataFrame(np.random.randn(20, 1))
        y = pd.Series(np.random.randn(20))
        
        # Mock ExponentialSmoothing fitting error
        mock_es_instance = Mock()
        mock_es_instance.fit.side_effect = ValueError("ExponentialSmoothing fitting failed")
        mock_es_class.return_value = mock_es_instance
        
        with pytest.raises(ValueError, match="ExponentialSmoothing fitting failed"):
            model.fit(X, y)
        
        # Should log error and not mark as fitted
        mock_logger.error.assert_called()
        assert model._is_fitted is False


class TestExponentialSmoothingWrapperPredict:
    """Test Exponential Smoothing wrapper prediction."""
    
    def test_exponential_smoothing_predict_not_fitted_error(self):
        """Test prediction error when model is not fitted."""
        model = ExponentialSmoothingWrapper()
        
        X = pd.DataFrame(np.random.randn(10, 2))
        
        with pytest.raises(ValueError, match="모델이 학습되지 않았습니다"):
            model.predict(X)
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    def test_exponential_smoothing_predict_success(self, mock_logger):
        """Test successful ExponentialSmoothing prediction."""
        # Setup fitted model
        model = ExponentialSmoothingWrapper()
        model._is_fitted = True
        
        # Mock fitted model with forecast method
        mock_fitted_model = Mock()
        mock_forecast_result = np.array([10.5, 11.2, 10.8, 11.5])
        mock_fitted_model.forecast.return_value = mock_forecast_result
        model.fitted_model_ = mock_fitted_model
        
        # Test data
        X_test = pd.DataFrame(np.random.randn(4, 2))  # 4 steps forecast
        
        # Execute
        predictions = model.predict(X_test)
        
        # Assertions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 4
        np.testing.assert_array_equal(predictions, mock_forecast_result)
        
        # Check forecast was called with correct steps
        mock_fitted_model.forecast.assert_called_once_with(steps=4)
        
        # Check logging
        mock_logger.info.assert_called()
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    def test_exponential_smoothing_predict_forecast_error(self, mock_logger):
        """Test ExponentialSmoothing prediction when forecast fails."""
        # Setup fitted model
        model = ExponentialSmoothingWrapper()
        model._is_fitted = True
        
        # Mock fitted model with forecast error
        mock_fitted_model = Mock()
        mock_fitted_model.forecast.side_effect = RuntimeError("Forecast failed")
        model.fitted_model_ = mock_fitted_model
        
        X_test = pd.DataFrame(np.random.randn(6, 1))
        
        with pytest.raises(RuntimeError, match="Forecast failed"):
            model.predict(X_test)
        
        # Should log error
        mock_logger.error.assert_called()


class TestTimeSeriesWrappersIntegration:
    """Test time series wrappers integration scenarios."""
    
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_arima_wrapper_fit_predict_pipeline(self, mock_arima_class):
        """Test complete ARIMA fit-predict pipeline."""
        # Setup
        model = ARIMAWrapper(order_p=2, order_d=1, order_q=2)
        
        # Training data
        X_train = pd.DataFrame(np.random.randn(100, 3))
        y_train = pd.Series(np.cumsum(np.random.randn(100)))  # Random walk
        
        # Test data
        X_test = pd.DataFrame(np.random.randn(10, 3))
        
        # Mock ARIMA components
        mock_arima_instance = Mock()
        mock_fitted_model = Mock()
        mock_arima_instance.fit.return_value = mock_fitted_model
        mock_fitted_model.forecast.return_value = np.array([1.1, 1.3, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        mock_arima_class.return_value = mock_arima_instance
        
        # Execute complete pipeline
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Assertions
        assert model._is_fitted is True
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
        
        # Check method calls
        mock_arima_class.assert_called_once_with(y_train, order=(2, 1, 2))
        mock_arima_instance.fit.assert_called_once()
        mock_fitted_model.forecast.assert_called_once_with(steps=10)
    
    @patch('statsmodels.tsa.holtwinters.ExponentialSmoothing')
    def test_exponential_smoothing_wrapper_fit_predict_pipeline(self, mock_es_class):
        """Test complete ExponentialSmoothing fit-predict pipeline."""
        # Setup
        model = ExponentialSmoothingWrapper(
            trend="add",
            seasonal="add",
            seasonal_periods=4
        )
        
        # Training data (quarterly data)
        X_train = pd.DataFrame(np.random.randn(20, 2))
        y_train = pd.Series(10 + np.cumsum(np.random.randn(20)) + np.sin(np.arange(20) * 2 * np.pi / 4))
        
        # Test data
        X_test = pd.DataFrame(np.random.randn(8, 2))
        
        # Mock ExponentialSmoothing components
        mock_es_instance = Mock()
        mock_fitted_model = Mock()
        mock_es_instance.fit.return_value = mock_fitted_model
        mock_fitted_model.forecast.return_value = np.array([12.1, 13.2, 11.8, 10.5, 12.3, 13.1, 11.9, 10.7])
        mock_es_class.return_value = mock_es_instance
        
        # Execute complete pipeline
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Assertions
        assert model._is_fitted is True
        assert len(predictions) == 8
        assert isinstance(predictions, np.ndarray)
        
        # Check method calls
        mock_es_class.assert_called_once_with(
            y_train,
            trend="add",
            seasonal="add",
            seasonal_periods=4
        )
        mock_es_instance.fit.assert_called_once()
        mock_fitted_model.forecast.assert_called_once_with(steps=8)


class TestTimeSeriesWrappersEdgeCases:
    """Test time series wrappers edge cases."""
    
    def test_arima_wrapper_empty_data(self):
        """Test ARIMA wrapper with empty data."""
        model = ARIMAWrapper()
        
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)
        
        # Should raise an error when trying to fit empty data
        with pytest.raises(Exception):  # Could be various types of errors
            model.fit(X_empty, y_empty)
    
    def test_exponential_smoothing_wrapper_empty_data(self):
        """Test ExponentialSmoothing wrapper with empty data."""
        model = ExponentialSmoothingWrapper()
        
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)
        
        # Should raise an error when trying to fit empty data
        with pytest.raises(Exception):  # Could be various types of errors
            model.fit(X_empty, y_empty)
    
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_arima_wrapper_single_forecast_step(self, mock_arima_class):
        """Test ARIMA wrapper with single forecast step."""
        model = ARIMAWrapper()
        
        # Mock components
        mock_arima_instance = Mock()
        mock_fitted_model = Mock()
        mock_arima_instance.fit.return_value = mock_fitted_model
        mock_fitted_model.forecast.return_value = np.array([42.0])
        mock_arima_class.return_value = mock_arima_instance
        
        # Fit model
        X_train = pd.DataFrame(np.random.randn(50, 1))
        y_train = pd.Series(np.random.randn(50))
        model.fit(X_train, y_train)
        
        # Single step prediction
        X_test = pd.DataFrame([[1.0]])  # Single row
        predictions = model.predict(X_test)
        
        assert len(predictions) == 1
        assert predictions[0] == 42.0
    
    def test_arima_wrapper_sklearn_compatibility(self):
        """Test ARIMA wrapper sklearn interface compatibility."""
        from sklearn.base import BaseEstimator, RegressorMixin
        
        model = ARIMAWrapper()
        
        # Should inherit from sklearn base classes
        assert isinstance(model, BaseEstimator)
        assert isinstance(model, RegressorMixin)
        
        # Should have sklearn-compatible methods
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert callable(getattr(model, 'fit'))
        assert callable(getattr(model, 'predict'))
    
    def test_exponential_smoothing_wrapper_sklearn_compatibility(self):
        """Test ExponentialSmoothing wrapper sklearn interface compatibility."""
        from sklearn.base import BaseEstimator, RegressorMixin
        
        model = ExponentialSmoothingWrapper()
        
        # Should inherit from sklearn base classes
        assert isinstance(model, BaseEstimator)
        assert isinstance(model, RegressorMixin)
        
        # Should have sklearn-compatible methods
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert callable(getattr(model, 'fit'))
        assert callable(getattr(model, 'predict'))
    
    @patch('src.models.custom.timeseries_wrappers.logger')
    def test_arima_wrapper_extreme_parameters(self, mock_logger):
        """Test ARIMA wrapper with extreme parameters."""
        # Very high order ARIMA
        model_high = ARIMAWrapper(order_p=10, order_d=3, order_q=10)
        assert model_high.order_p == 10
        assert model_high.order_d == 3
        assert model_high.order_q == 10
        
        # Zero order ARIMA (should be valid)
        model_zero = ARIMAWrapper(order_p=0, order_d=0, order_q=0)
        assert model_zero.order_p == 0
        assert model_zero.order_d == 0
        assert model_zero.order_q == 0
    
    def test_exponential_smoothing_wrapper_parameter_combinations(self):
        """Test ExponentialSmoothing wrapper with various parameter combinations."""
        # All combinations of trend and seasonal
        trend_options = [None, "add", "mul"]
        seasonal_options = [None, "add", "mul"]
        
        for trend in trend_options:
            for seasonal in seasonal_options:
                model = ExponentialSmoothingWrapper(
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=24
                )
                assert model.trend == trend
                assert model.seasonal == seasonal
                assert model.seasonal_periods == 24


class TestTimeSeriesWrappersPerformance:
    """Test time series wrappers performance characteristics."""
    
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_arima_wrapper_large_dataset(self, mock_arima_class):
        """Test ARIMA wrapper with large dataset."""
        model = ARIMAWrapper()
        
        # Large dataset
        X_large = pd.DataFrame(np.random.randn(1000, 5))
        y_large = pd.Series(np.cumsum(np.random.randn(1000)))
        
        # Mock components
        mock_arima_instance = Mock()
        mock_fitted_model = Mock()
        mock_arima_instance.fit.return_value = mock_fitted_model
        mock_arima_class.return_value = mock_arima_instance
        
        # Should handle large dataset without issues
        result = model.fit(X_large, y_large)
        assert result is model
        assert model._is_fitted is True
        
        # Check ARIMA was called with large series
        mock_arima_class.assert_called_once_with(y_large, order=(1, 1, 1))
    
    @patch('statsmodels.tsa.holtwinters.ExponentialSmoothing')
    def test_exponential_smoothing_wrapper_large_forecast(self, mock_es_class):
        """Test ExponentialSmoothing wrapper with large forecast horizon."""
        model = ExponentialSmoothingWrapper(seasonal="add", seasonal_periods=12)
        
        # Mock fitting
        mock_es_instance = Mock()
        mock_fitted_model = Mock()
        mock_es_instance.fit.return_value = mock_fitted_model
        mock_es_class.return_value = mock_es_instance
        
        # Fit model
        X_train = pd.DataFrame(np.random.randn(100, 2))
        y_train = pd.Series(np.random.randn(100))
        model.fit(X_train, y_train)
        
        # Large forecast horizon
        X_forecast = pd.DataFrame(np.random.randn(100, 2))  # 100 steps
        mock_fitted_model.forecast.return_value = np.random.randn(100)
        
        predictions = model.predict(X_forecast)
        
        assert len(predictions) == 100
        mock_fitted_model.forecast.assert_called_once_with(steps=100)


class TestTimeSeriesWrappersExports:
    """Test module exports."""
    
    def test_module_exports(self):
        """Test that module exports expected classes."""
        from src.models.custom.timeseries_wrappers import __all__, ARIMAWrapper, ExponentialSmoothingWrapper
        
        # Check __all__ exports
        assert 'ARIMAWrapper' in __all__
        assert 'ExponentialSmoothingWrapper' in __all__
        assert len(__all__) == 2
        
        # Check classes are accessible
        assert ARIMAWrapper is not None
        assert ExponentialSmoothingWrapper is not None
        assert callable(ARIMAWrapper)
        assert callable(ExponentialSmoothingWrapper)