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
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
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
        X, y = make_classification(n_samples=150, n_features=4, n_classes=3, 
                                 n_informative=3, random_state=42)
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