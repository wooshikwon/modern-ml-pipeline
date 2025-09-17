"""
Classification Evaluator Unit Tests - No Mock Hell Approach
Real metrics calculation, real model evaluation
Following comprehensive testing strategy document principles
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.components.evaluator.modules.classification_evaluator import ClassificationEvaluator
from src.interface.base_evaluator import BaseEvaluator


class TestClassificationEvaluator:
    """Test ClassificationEvaluator with real models and metrics."""
    
    def test_classification_evaluator_initialization(self, settings_builder):
        """Test ClassificationEvaluator initialization."""
        # Given: Valid settings for classification
        settings = settings_builder \
            .with_task("classification") \
            .build()
        
        # When: Creating ClassificationEvaluator
        evaluator = ClassificationEvaluator(settings)
        
        # Then: Evaluator is properly initialized
        assert isinstance(evaluator, ClassificationEvaluator)
        assert isinstance(evaluator, BaseEvaluator)
        assert evaluator.task_choice == "classification"
    
    def test_evaluate_with_random_forest_model(self, settings_builder, test_data_generator):
        """Test evaluation with real RandomForest model."""
        # Given: Trained RandomForest model and test data
        X, y = test_data_generator.classification_data(n_samples=100, n_features=5)
        X_train, y_train = X[:70], y[:70]
        X_test, y_test = X[70:], y[70:]
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        settings = settings_builder \
            .with_task("classification") \
            .build()
        evaluator = ClassificationEvaluator(settings)
        
        # When: Evaluating model
        metrics = evaluator.evaluate(model, X_test, y_test)
        
        # Then: Metrics are calculated correctly
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        # Check for class-specific metrics (binary classification has 2 classes: 0 and 1)
        assert 'class_0_precision' in metrics
        assert 'class_0_recall' in metrics
        assert 'class_0_f1' in metrics
        assert 'class_1_precision' in metrics
        assert 'class_1_recall' in metrics
        assert 'class_1_f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['class_0_precision'] <= 1
        assert 0 <= metrics['class_0_recall'] <= 1
        assert 0 <= metrics['class_0_f1'] <= 1
    
    def test_evaluate_with_logistic_regression(self, settings_builder, test_data_generator):
        """Test evaluation with LogisticRegression model."""
        # Given: Trained LogisticRegression model
        X, y = test_data_generator.classification_data(n_samples=80, n_features=4)
        X_train, y_train = X[:60], y[:60]
        X_test, y_test = X[60:], y[60:]
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        
        settings = settings_builder \
            .with_task("classification") \
            .build()
        evaluator = ClassificationEvaluator(settings)
        
        # When: Evaluating model
        metrics = evaluator.evaluate(model, X_test, y_test)
        
        # Then: All metrics are present and valid
        y_pred = model.predict(X_test)
        expected_accuracy = accuracy_score(y_test, y_pred)
        
        assert abs(metrics['accuracy'] - expected_accuracy) < 0.001
        assert metrics['accuracy'] > 0  # Model should have some predictive power
    
    def test_evaluate_with_multiclass_classification(self, settings_builder):
        """Test evaluation with multiclass classification."""
        # Given: Multiclass data and model
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=150, n_features=5, n_classes=3, 
                                 n_informative=3, n_redundant=0, random_state=42)
        X_train, y_train = X[:100], y[:100]
        X_test, y_test = X[100:], y[100:]
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        settings = settings_builder \
            .with_task("classification") \
            .build()
        evaluator = ClassificationEvaluator(settings)
        
        # When: Evaluating multiclass model
        metrics = evaluator.evaluate(model, X_test, y_test)
        
        # Then: Metrics handle multiclass correctly
        assert 'accuracy' in metrics
        assert metrics['accuracy'] > 0
        # For multiclass, precision/recall/f1 might be averaged
        if 'precision' in metrics:
            assert 0 <= metrics['precision'] <= 1
    
    def test_evaluate_with_perfect_predictions(self, settings_builder):
        """Test evaluation when model makes perfect predictions."""
        # Given: Model that makes perfect predictions
        class PerfectModel:
            def predict(self, X):
                return np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        model = PerfectModel()
        X_test = np.random.randn(10, 3)
        y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        settings = settings_builder \
            .with_task("classification") \
            .build()
        evaluator = ClassificationEvaluator(settings)
        
        # When: Evaluating perfect model
        metrics = evaluator.evaluate(model, X_test, y_test)
        
        # Then: Metrics should be perfect
        assert metrics['accuracy'] == 1.0
        if 'precision' in metrics:
            assert metrics['precision'] == 1.0
        if 'recall' in metrics:
            assert metrics['recall'] == 1.0

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

        settings = settings_builder \
            .with_task("classification") \
            .build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating model without predict_proba
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: ROC AUC should be None due to missing predict_proba
        assert metrics['roc_auc'] is None
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 1.0  # Perfect predictions

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

        settings = settings_builder \
            .with_task("classification") \
            .build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating model with problematic predict_proba
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: ROC AUC should be None due to exception handling
        assert metrics['roc_auc'] is None
        assert 'accuracy' in metrics

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

        settings = settings_builder \
            .with_task("classification") \
            .build()
        evaluator = ClassificationEvaluator(settings)

        # When: Evaluating poor model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Metrics should reflect poor performance
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 0.0  # Worst possible accuracy
        assert metrics['accuracy'] < 0.7  # Triggers poor performance guidance