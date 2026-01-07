"""
Classification Evaluator Unit Tests - No Mock Hell Approach
Real metrics calculation, real model evaluation
Following comprehensive testing strategy document principles
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.components.evaluator.modules.classification_evaluator import ClassificationEvaluator
from src.components.evaluator.base import BaseEvaluator


class TestClassificationEvaluator:
    """Test ClassificationEvaluator with real models and metrics."""

    def test_classification_evaluator_initialization(self, settings_builder):
        """Test ClassificationEvaluator initialization."""
        # Given: Valid settings for classification
        settings = settings_builder.with_task("classification").build()

        # When: Creating ClassificationEvaluator
        evaluator = ClassificationEvaluator(settings)

        # Then: Evaluator is properly initialized
        assert isinstance(evaluator, ClassificationEvaluator)
        assert isinstance(evaluator, BaseEvaluator)
        assert evaluator.task_choice == "classification"

    def test_evaluate_with_random_forest_model(self, settings_builder):
        """Test evaluation with real RandomForest model."""
        # Given: Trained RandomForest model and test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_train, y_train = X[:70], y[:70]
        X_test, y_test = X[70:], y[70:]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Metrics are calculated correctly
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        # Check for class-specific metrics (binary classification has 2 classes: 0 and 1)
        assert "class_0_precision" in metrics
        assert "class_0_recall" in metrics
        assert "class_0_f1" in metrics
        assert "class_1_precision" in metrics
        assert "class_1_recall" in metrics
        assert "class_1_f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["class_0_precision"] <= 1
        assert 0 <= metrics["class_0_recall"] <= 1
        assert 0 <= metrics["class_0_f1"] <= 1

    def test_evaluate_with_logistic_regression(self, settings_builder):
        """Test evaluation with LogisticRegression model."""
        # Given: Trained LogisticRegression model
        np.random.seed(42)
        X = np.random.randn(80, 4)
        y = np.random.randint(0, 2, 80)
        X_train, y_train = X[:60], y[:60]
        X_test, y_test = X[60:], y[60:]

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: All metrics are present and valid
        y_pred = model.predict(X_test)
        expected_accuracy = accuracy_score(y_test, y_pred)

        assert abs(metrics["accuracy"] - expected_accuracy) < 0.001
        assert metrics["accuracy"] > 0  # Model should have some predictive power

    def test_evaluate_with_multiclass_classification(self, settings_builder):
        """Test evaluation with multiclass classification."""
        # Given: Multiclass data and model
        np.random.seed(42)
        X = np.random.randn(150, 5)
        y = np.random.randint(0, 3, 150)  # 3 classes
        X_train, y_train = X[:100], y[:100]
        X_test, y_test = X[100:], y[100:]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating multiclass model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Metrics handle multiclass correctly
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0
        # For multiclass, precision/recall/f1 might be averaged
        if "precision" in metrics:
            assert 0 <= metrics["precision"] <= 1

    def test_evaluate_with_perfect_predictions(self, settings_builder):
        """Test evaluation when model makes perfect predictions."""

        # Given: Model that makes perfect predictions
        class PerfectModel:
            def predict(self, X):
                return np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        model = PerfectModel()
        X_test = np.random.randn(10, 3)
        y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating perfect model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Metrics should be perfect
        assert metrics["accuracy"] == 1.0
        if "precision" in metrics:
            assert metrics["precision"] == 1.0
        if "recall" in metrics:
            assert metrics["recall"] == 1.0

    def test_evaluate_with_model_without_predict_proba(self, settings_builder):
        """Test evaluation with model that doesn't support predict_proba."""

        # Given: Model without predict_proba method (multiclass case)
        class ModelWithoutProba:
            def predict(self, X):
                # Return multiclass predictions (3 classes: 0, 1, 2)
                return np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        model = ModelWithoutProba()
        X_test = np.random.randn(10, 3)
        y_test = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # Multiclass to trigger ovr ROC AUC path

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating model without predict_proba
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: ROC AUC should be None due to missing predict_proba
        assert metrics["roc_auc"] is None
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0  # Perfect predictions

    def test_evaluate_with_predict_proba_exception(self, settings_builder):
        """Test evaluation when predict_proba raises an exception."""

        # Given: Model with predict_proba that raises exception
        class ProblematicModel:
            def predict(self, X):
                return np.array([0, 1, 2, 0, 1])

            def predict_proba(self, X):
                raise ValueError("Prediction failed")

        model = ProblematicModel()
        X_test = np.random.randn(5, 3)
        y_test = np.array([0, 1, 2, 0, 1])  # Multiclass

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating model with problematic predict_proba
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: ROC AUC should be None due to exception handling
        assert metrics["roc_auc"] is None
        assert "accuracy" in metrics

    def test_evaluate_with_poor_performance_model(self, settings_builder):
        """Test evaluation with very poor performing model to trigger guidance."""

        # Given: Model with very poor accuracy (< 0.7)
        class PoorModel:
            def predict(self, X):
                # Always predict class 0, but most data is class 1
                return np.zeros(len(X), dtype=int)

        model = PoorModel()
        X_test = np.random.randn(20, 3)
        y_test = np.ones(20, dtype=int)  # All true labels are 1, but model predicts 0

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating poor model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Metrics should reflect poor performance
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 0.0  # Worst possible accuracy
        assert metrics["accuracy"] < 0.7  # Triggers poor performance guidance

    def test_evaluate_binary_classification_with_roc_auc(self, settings_builder):
        """Test binary classification ROC AUC calculation."""
        # Given: Binary classification model
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        X_train, y_train = X[:70], y[:70]
        X_test, y_test = X[70:], y[70:]

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating binary classification model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: ROC AUC should be calculated for binary classification
        assert "roc_auc" in metrics
        # ROC AUC might be None if there's an issue with the data
        if metrics["roc_auc"] is not None:
            assert 0 <= metrics["roc_auc"] <= 1
        assert len(np.unique(y_test)) == 2  # Confirm binary classification

    def test_evaluate_multiclass_with_predict_proba_success(self, settings_builder):
        """Test multiclass classification with successful predict_proba."""
        # Given: Multiclass model with working predict_proba
        np.random.seed(42)
        X = np.random.randn(120, 5)
        y = np.random.randint(0, 3, 120)  # 3 classes
        X_train, y_train = X[:90], y[:90]
        X_test, y_test = X[90:], y[90:]

        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating multiclass model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: ROC AUC should be calculated for multiclass with predict_proba
        assert "roc_auc" in metrics
        # ROC AUC might be None if there's an issue with the data
        if metrics["roc_auc"] is not None:
            assert 0 <= metrics["roc_auc"] <= 1
        assert len(np.unique(y_test)) == 3  # Confirm multiclass classification

    def test_evaluate_excellent_performance_guidance(self, settings_builder):
        """Test performance guidance for excellent model (≥0.9 accuracy)."""

        # Given: Model with excellent performance
        class ExcellentModel:
            def predict(self, X):
                # 95% accuracy - excellent performance
                predictions = np.ones(len(X), dtype=int)
                # Make 5% wrong predictions
                wrong_indices = np.random.choice(len(X), size=max(1, len(X) // 20), replace=False)
                predictions[wrong_indices] = 0
                return predictions

        model = ExcellentModel()
        X_test = np.random.randn(20, 3)
        y_test = np.ones(20, dtype=int)

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating excellent model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Should trigger excellent performance guidance
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0.9

    def test_evaluate_good_performance_guidance(self, settings_builder):
        """Test performance guidance for good model (0.8-0.9 accuracy)."""

        # Given: Model with good performance (85% accuracy)
        class GoodModel:
            def predict(self, X):
                predictions = np.ones(len(X), dtype=int)
                # Make 15% wrong predictions for ~85% accuracy
                wrong_indices = np.random.choice(len(X), size=3, replace=False)
                predictions[wrong_indices] = 0
                return predictions

        model = GoodModel()
        X_test = np.random.randn(20, 3)
        y_test = np.ones(20, dtype=int)

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating good model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Should trigger good performance guidance
        assert "accuracy" in metrics
        assert 0.8 <= metrics["accuracy"] < 0.9

    def test_evaluate_moderate_performance_guidance(self, settings_builder):
        """Test performance guidance for moderate model (0.7-0.8 accuracy)."""

        # Given: Model with moderate performance (75% accuracy)
        class ModerateModel:
            def predict(self, X):
                predictions = np.ones(len(X), dtype=int)
                # Make 25% wrong predictions for 75% accuracy
                wrong_indices = np.random.choice(len(X), size=5, replace=False)
                predictions[wrong_indices] = 0
                return predictions

        model = ModerateModel()
        X_test = np.random.randn(20, 3)
        y_test = np.ones(20, dtype=int)

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating moderate model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Should trigger moderate performance guidance
        assert "accuracy" in metrics
        assert 0.7 <= metrics["accuracy"] < 0.8

    def test_evaluate_with_class_specific_metrics_coverage(self, settings_builder):
        """Test comprehensive class-specific metrics calculation."""

        # Given: Model with known predictions for each class
        class DetailedTestModel:
            def predict(self, X):
                # Create predictions that will generate specific precision/recall values
                return np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # 50/50 split

        model = DetailedTestModel()
        X_test = np.random.randn(10, 3)
        y_test = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1])  # Mixed labels

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating with detailed metrics
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: All class-specific metrics should be present
        unique_classes = np.unique(y_test)
        for class_label in unique_classes:
            assert f"class_{class_label}_precision" in metrics
            assert f"class_{class_label}_recall" in metrics
            assert f"class_{class_label}_f1" in metrics
            assert f"class_{class_label}_support" in metrics

            # Verify metrics are valid
            assert 0 <= metrics[f"class_{class_label}_precision"] <= 1
            assert 0 <= metrics[f"class_{class_label}_recall"] <= 1
            assert 0 <= metrics[f"class_{class_label}_f1"] <= 1
            assert metrics[f"class_{class_label}_support"] > 0

    def test_evaluate_with_single_class_data(self, settings_builder):
        """Test evaluation with single class in test data."""

        # Given: Model and test data with only one class
        class SingleClassModel:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)  # Always predict 0

        model = SingleClassModel()
        X_test = np.random.randn(10, 3)
        y_test = np.zeros(10, dtype=int)  # All true labels are 0

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating with single class
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Should handle single class scenario
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0  # Perfect accuracy with single class
        assert "class_0_precision" in metrics
        assert "class_0_support" in metrics

    def test_evaluate_console_logging_coverage(self, settings_builder):
        """Test console logging system coverage."""
        # Given: Model to trigger various logging paths
        from sklearn.ensemble import RandomForestClassifier

        X, y = np.random.randn(50, 4), np.random.randint(0, 2, 50)
        X_train, y_train = X[:35], y[:35]
        X_test, y_test = X[35:], y[35:]

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)

        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating to trigger logging
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Should execute all logging paths including pipeline connection
        assert "accuracy" in metrics
        assert isinstance(metrics, dict)
        # Test passes if no exceptions are thrown during logging
