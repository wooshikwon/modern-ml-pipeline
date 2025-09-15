"""
FT-Transformer model comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/models/custom/ft_transformer.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.models.custom.ft_transformer import (
    FTTransformerWrapperBase,
    FTTransformerClassifier,
    FTTransformerRegressor
)


class TestFTTransformerWrapperBase:
    """FT-Transformer 베이스 클래스 테스트 - Context 클래스 기반"""

    def test_ft_transformer_base_initialization(self, component_test_context):
        """FTTransformerWrapperBase 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test basic initialization with concrete class
            hyperparams = {'d_block': 64, 'n_blocks': 3}
            model = FTTransformerClassifier(**hyperparams)

            # Verify initialization
            assert model.model is None
            assert model._internal_preprocessor is None
            assert model.hyperparams == hyperparams
            assert model.handles_own_preprocessing is True

    def test_ft_transformer_base_empty_hyperparams(self, component_test_context):
        """빈 하이퍼파라미터로 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = FTTransformerClassifier()

            assert model.model is None
            assert model._internal_preprocessor is None
            assert model.hyperparams == {}
            assert model.handles_own_preprocessing is True

    def test_ft_transformer_base_predict_before_fit_error(self, component_test_context):
        """학습 전 예측 시도 시 에러 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = FTTransformerClassifier()

            # Create test data
            X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': ['a', 'b', 'c']})

            # Should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                model.predict(X)

            assert "모델이 학습되지 않았습니다" in str(exc_info.value)
            assert "fit()을 먼저 호출하세요" in str(exc_info.value)


class TestFTTransformerDataPreprocessing:
    """FT-Transformer 데이터 전처리 테스트"""

    def test_ft_transformer_categorical_numerical_separation(self, component_test_context):
        """범주형/수치형 특성 분리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock FTTransformer to avoid actual model training
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                # Create mixed data
                X = pd.DataFrame({
                    'numerical1': [1.0, 2.0, 3.0],
                    'categorical1': ['a', 'b', 'c'],
                    'numerical2': [10, 20, 30],
                    'categorical2': pd.Categorical(['x', 'y', 'z'])
                })
                y = pd.Series([0, 1, 0])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Verify preprocessing was set up correctly
                assert model._internal_preprocessor is not None

                # Check that ColumnTransformer has the right transformers
                transformers = model._internal_preprocessor.transformers
                assert len(transformers) == 2

                # Verify numerical and categorical transformers
                num_transformer_name, num_transformer, num_features = transformers[0]
                cat_transformer_name, cat_transformer, cat_features = transformers[1]

                assert num_transformer_name == 'num'
                assert isinstance(num_transformer, StandardScaler)
                assert set(num_features) == {'numerical1', 'numerical2'}

                assert cat_transformer_name == 'cat'
                assert isinstance(cat_transformer, OrdinalEncoder)
                assert set(cat_features) == {'categorical1', 'categorical2'}

    def test_ft_transformer_only_numerical_features(self, component_test_context):
        """수치형 특성만 있는 데이터 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({
                    'num1': [1.0, 2.0, 3.0],
                    'num2': [10, 20, 30],
                    'num3': [0.1, 0.2, 0.3]
                })
                y = pd.Series([0, 1, 0])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check FTTransformer parameters
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['n_cont_features'] == 3
                assert call_args['cat_cardinalities'] == []

    def test_ft_transformer_only_categorical_features(self, component_test_context):
        """범주형 특성만 있는 데이터 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({
                    'cat1': ['a', 'b', 'c'],
                    'cat2': pd.Categorical(['x', 'y', 'z']),
                    'cat3': ['p', 'q', 'r']
                })
                y = pd.Series([0, 1, 2])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check FTTransformer parameters
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['n_cont_features'] == 0
                assert len(call_args['cat_cardinalities']) == 3
                # Each categorical feature should have cardinality of unique values + 1 for unknown
                assert all(cardinality == 4 for cardinality in call_args['cat_cardinalities'])


class TestFTTransformerCardinalityCalculation:
    """FT-Transformer 범주형 카디널리티 계산 테스트"""

    def test_ft_transformer_cardinality_with_unknown_handling(self, component_test_context):
        """Unknown 값 처리를 위한 카디널리티 계산 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({
                    'cat_small': ['a', 'b'],  # 2 unique values
                    'cat_large': ['x', 'y', 'z', 'w', 'v']  # 5 unique values
                })
                y = pd.Series([0, 1])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check cardinalities include space for unknown values
                call_args = mock_ft_transformer.call_args[1]
                cardinalities = call_args['cat_cardinalities']

                assert len(cardinalities) == 2
                assert cardinalities[0] == 3  # 2 + 1 for unknown
                assert cardinalities[1] == 6  # 5 + 1 for unknown

    def test_ft_transformer_ordinal_encoder_configuration(self, component_test_context):
        """OrdinalEncoder 설정 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'cat1': ['a', 'b', 'c']})
                y = pd.Series([0, 1, 0])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check OrdinalEncoder configuration
                cat_transformer = model._internal_preprocessor.named_transformers_['cat']
                assert isinstance(cat_transformer, OrdinalEncoder)
                assert cat_transformer.handle_unknown == 'use_encoded_value'
                assert cat_transformer.unknown_value == -1


class TestFTTransformerHyperparameterHandling:
    """FT-Transformer 하이퍼파라미터 처리 테스트"""

    def test_ft_transformer_default_hyperparameters(self, component_test_context):
        """기본 하이퍼파라미터 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'num1': [1, 2, 3]})
                y = pd.Series([0, 1, 0])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check default hyperparameters were applied
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['d_block'] == 32
                assert call_args['n_blocks'] == 2
                assert call_args['attention_n_heads'] == 2
                assert call_args['attention_dropout'] == 0.1
                assert call_args['ffn_d_hidden_multiplier'] == 4
                assert call_args['ffn_dropout'] == 0.1
                assert call_args['residual_dropout'] == 0.0

    def test_ft_transformer_custom_hyperparameters(self, component_test_context):
        """커스텀 하이퍼파라미터 적용 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'num1': [1, 2, 3]})
                y = pd.Series([0, 1, 0])

                custom_params = {
                    'd_block': 128,
                    'n_blocks': 4,
                    'attention_n_heads': 8,
                    'attention_dropout': 0.2,
                    'custom_param': 'test_value'
                }

                model = FTTransformerWrapperBase(**custom_params)
                model._initialize_and_fit(X, y, d_out=2)

                # Check custom hyperparameters were applied
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['d_block'] == 128
                assert call_args['n_blocks'] == 4
                assert call_args['attention_n_heads'] == 8
                assert call_args['attention_dropout'] == 0.2
                assert call_args['custom_param'] == 'test_value'

    def test_ft_transformer_n_heads_alias_handling(self, component_test_context):
        """n_heads 별칭 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'num1': [1, 2, 3]})
                y = pd.Series([0, 1, 0])

                # Use n_heads instead of attention_n_heads
                model = FTTransformerWrapperBase(n_heads=6)
                model._initialize_and_fit(X, y, d_out=2)

                # Check that n_heads was converted to attention_n_heads
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['attention_n_heads'] == 6

    def test_ft_transformer_parameter_conflict_resolution(self, component_test_context):
        """파라미터 충돌 해결 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'num1': [1, 2, 3]})
                y = pd.Series([0, 1, 0])

                # Include both n_heads and attention_n_heads
                conflicting_params = {
                    'n_heads': 4,
                    'attention_n_heads': 8,
                    'd_block': 64
                }

                model = FTTransformerWrapperBase(**conflicting_params)
                model._initialize_and_fit(X, y, d_out=2)

                # Check that attention_n_heads takes precedence
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['attention_n_heads'] == 8
                assert 'n_heads' not in call_args  # Should be excluded


class TestFTTransformerPrediction:
    """FT-Transformer 예측 기능 테스트"""

    def test_ft_transformer_predict_proba_binary_classification(self, component_test_context):
        """이진 분류 predict_proba 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
                mock_ft_transformer.return_value = mock_model

                X_train = pd.DataFrame({'num1': [1, 2, 3]})
                y_train = pd.Series([0, 1, 0])
                X_test = pd.DataFrame({'num1': [4, 5, 6]}, index=[10, 11, 12])

                model = FTTransformerWrapperBase()
                model._initialize_and_fit(X_train, y_train, d_out=2)

                predictions = model.predict(X_test)

                # Should return probabilities for positive class
                expected_probs = np.array([0.8, 0.3, 0.6])
                assert isinstance(predictions, pd.DataFrame)
                assert list(predictions.index) == [10, 11, 12]
                assert predictions.columns.tolist() == ['prediction']
                np.testing.assert_array_almost_equal(predictions['prediction'].values, expected_probs)

    def test_ft_transformer_predict_proba_multiclass_classification(self, component_test_context):
        """다중 분류 predict_proba 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_model.predict_proba.return_value = np.array([
                    [0.1, 0.3, 0.6],  # argmax = 2
                    [0.8, 0.1, 0.1],  # argmax = 0
                    [0.2, 0.7, 0.1]   # argmax = 1
                ])
                mock_ft_transformer.return_value = mock_model

                X_train = pd.DataFrame({'num1': [1, 2, 3]})
                y_train = pd.Series([0, 1, 2])
                X_test = pd.DataFrame({'num1': [4, 5, 6]}, index=[10, 11, 12])

                model = FTTransformerClassifier()
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                # Should return argmax for multiclass
                expected_classes = np.array([2, 0, 1])
                assert isinstance(predictions, pd.DataFrame)
                assert list(predictions.index) == [10, 11, 12]
                np.testing.assert_array_equal(predictions['prediction'].values, expected_classes)

    def test_ft_transformer_predict_regression(self, component_test_context):
        """회귀 예측 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                # Mock model without predict_proba (regression)
                delattr(mock_model, 'predict_proba') if hasattr(mock_model, 'predict_proba') else None
                mock_model.predict.return_value = np.array([1.5, 2.3, 0.8])
                mock_ft_transformer.return_value = mock_model

                X_train = pd.DataFrame({'num1': [1, 2, 3]})
                y_train = pd.Series([1.0, 2.0, 1.5])
                X_test = pd.DataFrame({'num1': [4, 5, 6]}, index=[10, 11, 12])

                model = FTTransformerRegressor()
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                # Should return regression predictions
                expected_values = np.array([1.5, 2.3, 0.8])
                assert isinstance(predictions, pd.DataFrame)
                assert list(predictions.index) == [10, 11, 12]
                np.testing.assert_array_almost_equal(predictions['prediction'].values, expected_values)


class TestFTTransformerClassifier:
    """FTTransformerClassifier 특화 테스트"""

    def test_ft_transformer_classifier_binary_fit(self, component_test_context):
        """이진 분류 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'num1': [1, 2, 3, 4]})
                y = pd.Series([0, 1, 0, 1])  # 2 unique classes

                classifier = FTTransformerClassifier()
                result = classifier.fit(X, y)

                # Check that d_out was set correctly for binary classification
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['d_out'] == 2

                # Check that fit returns self
                assert result is classifier
                assert classifier.model is mock_model

    def test_ft_transformer_classifier_multiclass_fit(self, component_test_context):
        """다중 분류 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'num1': [1, 2, 3, 4, 5]})
                y = pd.Series([0, 1, 2, 0, 1])  # 3 unique classes

                classifier = FTTransformerClassifier()
                result = classifier.fit(X, y)

                # Check that d_out was set correctly for multiclass
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['d_out'] == 3

                # Check that fit returns self
                assert result is classifier

    def test_ft_transformer_classifier_single_class_fit(self, component_test_context):
        """단일 클래스 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'num1': [1, 2, 3, 4]})
                y = pd.Series([1, 1, 1, 1])  # Only 1 unique class

                classifier = FTTransformerClassifier()
                classifier.fit(X, y)

                # Check that d_out was set correctly for single class
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['d_out'] == 1


class TestFTTransformerRegressor:
    """FTTransformerRegressor 특화 테스트"""

    def test_ft_transformer_regressor_fit(self, component_test_context):
        """회귀 모델 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'num1': [1, 2, 3, 4]})
                y = pd.Series([1.5, 2.3, 1.8, 2.7])

                regressor = FTTransformerRegressor()
                result = regressor.fit(X, y)

                # Check that d_out was set to 1 for regression
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['d_out'] == 1

                # Check that fit returns self
                assert result is regressor
                assert regressor.model is mock_model

    def test_ft_transformer_regressor_with_hyperparams(self, component_test_context):
        """하이퍼파라미터가 있는 회귀 모델 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model

                X = pd.DataFrame({'num1': [1, 2, 3]})
                y = pd.Series([1.0, 2.0, 1.5])

                hyperparams = {'d_block': 256, 'n_blocks': 6}
                regressor = FTTransformerRegressor(**hyperparams)
                regressor.fit(X, y)

                # Check that custom hyperparameters were applied
                call_args = mock_ft_transformer.call_args[1]
                assert call_args['d_out'] == 1
                assert call_args['d_block'] == 256
                assert call_args['n_blocks'] == 6


class TestFTTransformerIntegration:
    """FT-Transformer 통합 테스트"""

    def test_ft_transformer_complete_workflow_classification(self, component_test_context):
        """분류 모델 완전한 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                # Setup mock model with both training and prediction capabilities
                mock_model = Mock()
                mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
                mock_ft_transformer.return_value = mock_model

                # Mixed feature data
                X_train = pd.DataFrame({
                    'numerical': [1.0, 2.0, 3.0],
                    'categorical': ['a', 'b', 'c']
                })
                y_train = pd.Series([0, 1, 0])

                X_test = pd.DataFrame({
                    'numerical': [1.5, 2.5],
                    'categorical': ['a', 'b']
                })

                # Complete workflow: initialize -> fit -> predict
                classifier = FTTransformerClassifier(d_block=64, n_blocks=3)
                classifier.fit(X_train, y_train)
                predictions = classifier.predict(X_test)

                # Verify complete workflow
                assert mock_ft_transformer.called
                assert mock_model.fit.called
                assert mock_model.predict_proba.called

                # Verify prediction output
                assert isinstance(predictions, pd.DataFrame)
                assert len(predictions) == 2
                assert predictions.columns.tolist() == ['prediction']

    def test_ft_transformer_complete_workflow_regression(self, component_test_context):
        """회귀 모델 완전한 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.models.custom.ft_transformer.FTTransformer') as mock_ft_transformer:
                # Setup mock model for regression
                mock_model = Mock()
                delattr(mock_model, 'predict_proba') if hasattr(mock_model, 'predict_proba') else None
                mock_model.predict.return_value = np.array([1.2, 2.8])
                mock_ft_transformer.return_value = mock_model

                X_train = pd.DataFrame({'feature': [1, 2, 3, 4]})
                y_train = pd.Series([1.1, 2.2, 3.3, 4.4])
                X_test = pd.DataFrame({'feature': [5, 6]})

                # Complete regression workflow
                regressor = FTTransformerRegressor()
                regressor.fit(X_train, y_train)
                predictions = regressor.predict(X_test)

                # Verify workflow and output
                assert mock_model.predict.called
                assert isinstance(predictions, pd.DataFrame)
                assert len(predictions) == 2
                np.testing.assert_array_almost_equal(predictions['prediction'].values, [1.2, 2.8])

    def test_ft_transformer_error_handling_workflow(self, component_test_context):
        """오류 처리 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test unfitted model prediction error
            classifier = FTTransformerClassifier()
            X_test = pd.DataFrame({'feature': [1, 2, 3]})

            with pytest.raises(RuntimeError) as exc_info:
                classifier.predict(X_test)

            assert "모델이 학습되지 않았습니다" in str(exc_info.value)

            # Test that error persists even after partial initialization
            regressor = FTTransformerRegressor()
            regressor.model = None  # Simulate partial state
            regressor._internal_preprocessor = Mock()  # But preprocessor exists

            with pytest.raises(RuntimeError) as exc_info:
                regressor.predict(X_test)

            assert "모델이 학습되지 않았습니다" in str(exc_info.value)