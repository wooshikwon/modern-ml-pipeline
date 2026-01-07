"""
Model Interface Contract Tests - No Mock Hell Approach
Testing BaseModel contract compliance with real implementations
Following comprehensive testing strategy document principles
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel


class CustomTestModel(BaseModel):
    """Custom model implementation for testing BaseModel contract."""

    def __init__(self):
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs):
        """Implement fit method according to BaseModel contract."""
        self.is_fitted = True
        self.feature_names = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Implement predict method according to BaseModel contract."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Simple prediction: return zeros
        predictions = pd.DataFrame({"predictions": np.zeros(len(X))})
        return predictions


class TestModelInterfaceContract:
    """Test BaseModel interface contract compliance."""

    def test_base_model_interface_definition(self):
        """Test that BaseModel defines required interface."""
        # Given: BaseModel class

        # Then: Required methods are abstract
        assert hasattr(BaseModel, "fit")
        assert hasattr(BaseModel, "predict")
        assert inspect.isabstract(BaseModel)

        # Verify method signatures
        fit_sig = inspect.signature(BaseModel.fit)
        assert "X" in fit_sig.parameters
        assert "y" in fit_sig.parameters

        predict_sig = inspect.signature(BaseModel.predict)
        assert "X" in predict_sig.parameters

    def test_custom_model_implements_base_model(self, test_data_generator):
        """Test custom model properly implements BaseModel interface."""
        # Given: Custom model and data
        model = CustomTestModel()
        X, y = test_data_generator.classification_data(n_samples=50, n_features=3)
        X_df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        y_series = pd.Series(y, name="target")

        # When: Using model through BaseModel interface
        assert isinstance(model, BaseModel)

        # Fit should return self
        returned_model = model.fit(X_df, y_series)
        assert returned_model is model
        assert model.is_fitted

        # Predict should return DataFrame
        predictions = model.predict(X_df)
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(X_df)

    def test_sklearn_model_wrapper_contract(self, test_data_generator):
        """Test that sklearn models can be wrapped to follow BaseModel contract."""

        # Given: Sklearn model wrapper
        class SklearnModelWrapper(BaseModel):
            def __init__(self, sklearn_model):
                self.model = sklearn_model

            def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs):
                self.model.fit(X, y)
                return self

            def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                predictions = self.model.predict(X)
                return pd.DataFrame({"predictions": predictions})

        # When: Wrapping sklearn model
        sklearn_model = RandomForestClassifier(n_estimators=5, random_state=42)
        wrapped_model = SklearnModelWrapper(sklearn_model)

        X, y = test_data_generator.classification_data(n_samples=50)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        # Then: Wrapped model follows BaseModel contract
        assert isinstance(wrapped_model, BaseModel)
        wrapped_model.fit(X_df, y_series)
        predictions = wrapped_model.predict(X_df)
        assert isinstance(predictions, pd.DataFrame)
        assert "predictions" in predictions.columns

    def test_model_fit_method_contract(self, test_data_generator):
        """Test fit method contract requirements."""
        # Given: Model and various data formats
        model = CustomTestModel()
        X, y = test_data_generator.regression_data(n_samples=30)

        # Test 1: DataFrame input
        X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)
        result = model.fit(X_df, y_series)
        assert result is model  # Should return self

        # Test 2: Optional y parameter
        model2 = CustomTestModel()
        result2 = model2.fit(X_df)  # y is optional
        assert result2 is model2

        # Test 3: Additional kwargs
        model3 = CustomTestModel()
        result3 = model3.fit(X_df, y_series, sample_weight=np.ones(len(y)))
        assert result3 is model3

    def test_model_predict_method_contract(self, test_data_generator):
        """Test predict method contract requirements."""
        # Given: Fitted model
        model = CustomTestModel()
        X, y = test_data_generator.classification_data(n_samples=40)
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        model.fit(X_df, y_series)

        # When: Making predictions
        predictions = model.predict(X_df)

        # Then: Output must be DataFrame
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(X_df)

        # Test with different sized input
        X_new = X_df.iloc[:10]
        predictions_new = model.predict(X_new)
        assert len(predictions_new) == 10

    def test_model_error_handling_unfitted(self):
        """Test model behavior when predicting without fitting."""
        # Given: Unfitted model
        model = CustomTestModel()
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # When/Then: Should raise error
        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_model_interface_with_different_tasks(self, test_data_generator):
        """Test model interface works for different ML tasks."""

        # Classification task
        class ClassificationModel(BaseModel):
            def fit(self, X, y=None, **kwargs):
                self.classes_ = np.unique(y) if y is not None else [0, 1]
                return self

            def predict(self, X):
                # Return class predictions
                return pd.DataFrame({"predictions": np.random.choice(self.classes_, size=len(X))})

        # Regression task
        class RegressionModel(BaseModel):
            def fit(self, X, y=None, **kwargs):
                self.mean_ = np.mean(y) if y is not None else 0
                return self

            def predict(self, X):
                # Return continuous predictions
                return pd.DataFrame({"predictions": np.random.randn(len(X)) + self.mean_})

        # Test both models follow same interface
        X, y_cls = test_data_generator.classification_data(n_samples=20)
        _, y_reg = test_data_generator.regression_data(n_samples=20)
        X_df = pd.DataFrame(X)

        cls_model = ClassificationModel()
        reg_model = RegressionModel()

        # Both should work with same interface
        cls_model.fit(X_df, pd.Series(y_cls))
        reg_model.fit(X_df, pd.Series(y_reg))

        cls_pred = cls_model.predict(X_df)
        reg_pred = reg_model.predict(X_df)

        assert isinstance(cls_pred, pd.DataFrame)
        assert isinstance(reg_pred, pd.DataFrame)
