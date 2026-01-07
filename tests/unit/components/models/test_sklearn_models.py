"""
Sklearn Models Unit Tests - No Mock Hell Approach
Real models, real training, real predictions validation
Following comprehensive testing strategy document principles
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class TestSklearnClassificationModels:
    """Test sklearn classification models with real training and prediction."""

    def test_random_forest_classifier_training(self, test_data_generator):
        """Test RandomForestClassifier training with real data."""
        # Given: Real classification data
        X_df, y = test_data_generator.classification_data(n_samples=100, n_features=5)
        # Remove entity_id column for model training
        X_df = X_df.drop(columns=["entity_id"])
        y_series = pd.Series(y, name="target")

        # When: Training RandomForest model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y_series)

        # Then: Model is trained successfully
        assert hasattr(model, "estimators_")
        assert len(model.estimators_) == 10
        predictions = model.predict(X_df)
        accuracy = accuracy_score(y_series, predictions)
        assert accuracy > 0.8  # Should fit training data well

    def test_logistic_regression_training(self, test_data_generator):
        """Test LogisticRegression training with real data."""
        # Given: Real classification data
        X_df, y = test_data_generator.classification_data(n_samples=100, n_features=5)
        # Remove entity_id column for model training
        X_df = X_df.drop(columns=["entity_id"])
        y_series = pd.Series(y, name="target")

        # When: Training LogisticRegression model
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_df, y_series)

        # Then: Model is trained successfully
        assert hasattr(model, "coef_")
        assert model.coef_.shape == (1, X_df.shape[1]) or model.coef_.shape == (2, X_df.shape[1])
        predictions = model.predict(X_df)
        assert len(predictions) == len(y_series)

    def test_decision_tree_classifier_training(self, test_data_generator):
        """Test DecisionTreeClassifier training with real data."""
        # Given: Real classification data
        X_df, y = test_data_generator.classification_data(n_samples=80, n_features=4)
        # Remove entity_id column for model training
        X_df = X_df.drop(columns=["entity_id"])
        y_series = pd.Series(y, name="target")

        # When: Training DecisionTree model
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_df, y_series)

        # Then: Model is trained successfully
        assert hasattr(model, "tree_")
        assert model.tree_.node_count > 0
        predictions = model.predict(X_df)
        unique_predictions = np.unique(predictions)
        assert len(unique_predictions) >= 2  # Binary classification

    def test_svm_classifier_training(self, test_data_generator):
        """Test SVM classifier training with real data."""
        # Given: Small dataset for SVM (computationally expensive)
        X_df, y = test_data_generator.classification_data(n_samples=50, n_features=3)
        # Remove entity_id column for model training
        X_df = X_df.drop(columns=["entity_id"])
        y_series = pd.Series(y, name="target")

        # When: Training SVM model
        model = SVC(kernel="rbf", random_state=42)
        model.fit(X_df, y_series)

        # Then: Model is trained successfully
        assert hasattr(model, "support_vectors_")
        assert len(model.support_vectors_) > 0
        predictions = model.predict(X_df)
        assert len(predictions) == len(y_series)

    def test_model_predict_proba_functionality(self, test_data_generator):
        """Test predict_proba functionality for probabilistic models."""
        # Given: Data and probabilistic model
        X_df, y = test_data_generator.classification_data(n_samples=60, n_features=4)
        # Remove entity_id column for model training
        X_df = X_df.drop(columns=["entity_id"])
        y_series = pd.Series(y, name="target")

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_df, y_series)

        # When: Getting probability predictions
        proba = model.predict_proba(X_df)

        # Then: Probabilities are valid
        assert proba.shape == (len(X_df), 2)  # Binary classification
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all((proba >= 0) & (proba <= 1))  # Valid probability range


class TestSklearnRegressionModels:
    """Test sklearn regression models with real training and prediction."""

    def test_linear_regression_training(self, test_data_generator):
        """Test LinearRegression training with real data."""
        # Given: Real regression data
        X_df, y = test_data_generator.regression_data(n_samples=100, n_features=5)
        # Remove entity_id column for model training
        X_df = X_df.drop(columns=["entity_id"])
        y_series = pd.Series(y, name="target")

        # When: Training LinearRegression model
        model = LinearRegression()
        model.fit(X_df, y_series)

        # Then: Model is trained successfully
        assert hasattr(model, "coef_")
        assert len(model.coef_) == X_df.shape[1]
        predictions = model.predict(X_df)
        mse = mean_squared_error(y_series, predictions)
        assert mse < 1.0  # Should fit synthetic data well

    def test_decision_tree_regressor_training(self, test_data_generator):
        """Test DecisionTreeRegressor training with real data."""
        # Given: Real regression data
        X_df, y = test_data_generator.regression_data(n_samples=80, n_features=4)
        # Remove entity_id column for model training
        X_df = X_df.drop(columns=["entity_id"])
        y_series = pd.Series(y, name="target")

        # When: Training DecisionTreeRegressor
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X_df, y_series)

        # Then: Model is trained successfully
        assert hasattr(model, "tree_")
        predictions = model.predict(X_df)
        assert predictions.shape == y_series.shape
        assert np.std(predictions) > 0  # Model produces varied predictions

    def test_model_feature_importance_extraction(self, test_data_generator):
        """Test extracting feature importance from tree-based models."""
        # Given: Data and tree-based model
        X_df, y = test_data_generator.classification_data(n_samples=100, n_features=5)
        # Remove entity_id column for model training
        X_df = X_df.drop(columns=["entity_id"])
        y_series = pd.Series(y, name="target")

        # When: Training model and extracting feature importance
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y_series)

        # Then: Feature importance is available
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == X_df.shape[1]
        assert np.sum(model.feature_importances_) > 0
        assert np.allclose(np.sum(model.feature_importances_), 1.0, rtol=1e-3)
