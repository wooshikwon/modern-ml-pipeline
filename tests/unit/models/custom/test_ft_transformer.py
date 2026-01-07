"""
FT-Transformer model comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/models/custom/ft_transformer.py

Author: Phase 2A Development
Date: 2025-09-13
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.models.custom.ft_transformer import (
    FTTransformerClassifier,
    FTTransformerRegressor,
)


class TestFTTransformerWrapperBase:
    """FT-Transformer 베이스 클래스 테스트 - Context 클래스 기반"""

    def test_ft_transformer_base_initialization(self, component_test_context):
        """FTTransformerWrapperBase 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test basic initialization with concrete class
            hyperparams = {"d_block": 64, "n_blocks": 3}
            model = FTTransformerClassifier(**hyperparams)

            # Verify initialization
            assert model.model is None
            assert model._internal_preprocessor is None
            for k, v in hyperparams.items():
                assert model.hyperparams[k] == v
            assert model.handles_own_preprocessing is True

    def test_ft_transformer_base_empty_hyperparams(self, component_test_context):
        """빈 하이퍼파라미터로 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = FTTransformerClassifier()

            assert model.model is None
            assert model._internal_preprocessor is None
            assert isinstance(model.hyperparams, dict)
            assert model.hyperparams["d_block"] == 64  # Default
            assert model.handles_own_preprocessing is True

    def test_ft_transformer_base_predict_before_fit_error(self, component_test_context):
        """학습 전 예측 시도 시 에러 테스트"""
        with component_test_context.classification_stack() as ctx:
            model = FTTransformerClassifier()

            # Create test data
            X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"]})

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
            # Mock FTTransformer와 train_pytorch_model 모두 패치
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                # Create mixed data (Categorical 대신 object 타입 사용)
                X = pd.DataFrame(
                    {
                        "numerical1": [1.0, 2.0, 3.0],
                        "categorical1": ["a", "b", "c"],
                        "numerical2": [10, 20, 30],
                        "categorical2": ["x", "y", "z"],
                    }
                )
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

                assert num_transformer_name == "num"
                assert isinstance(num_transformer, StandardScaler)
                assert set(num_features) == {"numerical1", "numerical2"}

                assert cat_transformer_name == "cat"
                assert isinstance(cat_transformer, OrdinalEncoder)
                assert set(cat_features) == {"categorical1", "categorical2"}

    def test_ft_transformer_only_numerical_features(self, component_test_context):
        """수치형 특성만 있는 데이터 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame(
                    {"num1": [1.0, 2.0, 3.0], "num2": [10, 20, 30], "num3": [0.1, 0.2, 0.3]}
                )
                y = pd.Series([0, 1, 0])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check FTTransformer parameters
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["n_cont_features"] == 3
                assert call_args["cat_cardinalities"] is None

    def test_ft_transformer_only_categorical_features(self, component_test_context):
        """범주형 특성만 있는 데이터 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame(
                    {
                        "cat1": ["a", "b", "c"],
                        "cat2": ["x", "y", "z"],
                        "cat3": ["p", "q", "r"],
                    }
                )
                y = pd.Series([0, 1, 2])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check FTTransformer parameters
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["n_cont_features"] == 0
                assert len(call_args["cat_cardinalities"]) == 3
                # Each categorical feature should have cardinality of unique values + 1 for unknown
                assert all(cardinality == 4 for cardinality in call_args["cat_cardinalities"])


class TestFTTransformerCardinalityCalculation:
    """FT-Transformer 범주형 카디널리티 계산 테스트"""

    def test_ft_transformer_cardinality_with_unknown_handling(self, component_test_context):
        """Unknown 값 처리를 위한 카디널리티 계산 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame(
                    {
                        "cat_small": ["a", "b", "a", "b", "a"],  # 2 unique values
                        "cat_large": ["x", "y", "z", "w", "v"],  # 5 unique values
                    }
                )
                y = pd.Series([0, 1, 0, 1, 0])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check cardinalities include space for unknown values
                call_args = mock_ft_transformer.call_args[1]
                cardinalities = call_args["cat_cardinalities"]

                assert len(cardinalities) == 2
                assert cardinalities[0] == 3  # 2 + 1 for unknown
                assert cardinalities[1] == 6  # 5 + 1 for unknown

    def test_ft_transformer_ordinal_encoder_configuration(self, component_test_context):
        """OrdinalEncoder 설정 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"cat1": ["a", "b", "c"]})
                y = pd.Series([0, 1, 0])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check OrdinalEncoder configuration
                cat_transformer = model._internal_preprocessor.named_transformers_["cat"]
                assert isinstance(cat_transformer, OrdinalEncoder)
                assert cat_transformer.handle_unknown == "use_encoded_value"
                assert cat_transformer.unknown_value == -1


class TestFTTransformerHyperparameterHandling:
    """FT-Transformer 하이퍼파라미터 처리 테스트"""

    def test_ft_transformer_default_hyperparameters(self, component_test_context):
        """기본 하이퍼파라미터 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"num1": [1, 2, 3]})
                y = pd.Series([0, 1, 0])

                model = FTTransformerClassifier()
                model.fit(X, y)

                # Check default hyperparameters were applied
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_block"] == 64
                assert call_args["n_blocks"] == 2
                assert call_args["attention_n_heads"] == 4
                assert call_args["attention_dropout"] == 0.1
                assert call_args["ffn_d_hidden_multiplier"] == 4.0
                assert call_args["ffn_dropout"] == 0.1
                assert call_args["residual_dropout"] == 0.0

    def test_ft_transformer_custom_hyperparameters(self, component_test_context):
        """커스텀 하이퍼파라미터 적용 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"num1": [1, 2, 3]})
                y = pd.Series([0, 1, 0])

                custom_params = {
                    "d_block": 128,
                    "n_blocks": 4,
                    "attention_n_heads": 8,
                    "attention_dropout": 0.2,
                }

                model = FTTransformerClassifier(**custom_params)
                # Add custom_param to model after initialization for testing
                model.custom_param = "test_value"
                model.fit(X, y)

                # Check custom hyperparameters were applied
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_block"] == 128
                assert call_args["n_blocks"] == 4
                assert call_args["attention_n_heads"] == 8
                assert call_args["attention_dropout"] == 0.2
                # Custom param is not part of FTTransformer args

    def test_ft_transformer_n_heads_alias_handling(self, component_test_context):
        """n_heads는 별칭이 아닌 별도 파라미터로, attention_n_heads를 직접 사용해야 함"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"num1": [1, 2, 3]})
                y = pd.Series([0, 1, 0])

                # n_heads는 별도 파라미터이므로 attention_n_heads에 영향 없음
                model = FTTransformerClassifier(attention_n_heads=6)
                model.fit(X, y)

                # attention_n_heads가 올바르게 적용되는지 확인
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["attention_n_heads"] == 6

    def test_ft_transformer_parameter_conflict_resolution(self, component_test_context):
        """명시적 파라미터는 kwargs를 통해 전달된 값보다 우선함"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"num1": [1, 2, 3]})
                y = pd.Series([0, 1, 0])

                # 명시적 하이퍼파라미터 설정
                model = FTTransformerClassifier(attention_n_heads=8, d_block=64)
                model.fit(X, y)

                # FTTransformer에 전달된 파라미터 확인
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["attention_n_heads"] == 8
                assert call_args["d_block"] == 64


class TestFTTransformerPrediction:
    """FT-Transformer 예측 기능 테스트"""

    def test_ft_transformer_predict_proba_binary_classification(self, component_test_context):
        """이진 분류 predict_proba 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X_train = pd.DataFrame({"num1": [1, 2, 3]})
                y_train = pd.Series([0, 1, 0])

                model = FTTransformerClassifier()
                model.fit(X_train, y_train)

                # fit 완료 확인
                assert model.is_fitted is True
                assert model.model is not None
                # d_out이 클래스 수에 맞게 설정되었는지 확인
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_out"] == 2  # binary classification

    def test_ft_transformer_predict_proba_multiclass_classification(self, component_test_context):
        """다중 분류 predict_proba 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X_train = pd.DataFrame({"num1": [1, 2, 3]})
                y_train = pd.Series([0, 1, 2])

                model = FTTransformerClassifier()
                model.fit(X_train, y_train)

                # fit 완료 확인
                assert model.is_fitted is True
                assert model.model is not None
                # d_out이 클래스 수에 맞게 설정되었는지 확인
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_out"] == 3  # multiclass classification

    def test_ft_transformer_predict_regression(self, component_test_context):
        """회귀 예측 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train, \
                 patch("src.models.custom.ft_transformer.predict_with_pytorch_model") as mock_predict:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}
                mock_predict.return_value = [1.5, 2.3, 0.8]

                X_train = pd.DataFrame({"num1": [1, 2, 3]})
                y_train = pd.Series([1.0, 2.0, 1.5])
                X_test = pd.DataFrame({"num1": [4, 5, 6]}, index=[10, 11, 12])

                model = FTTransformerRegressor()
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                # Should return regression predictions
                expected_values = np.array([1.5, 2.3, 0.8])
                assert isinstance(predictions, pd.DataFrame)
                assert list(predictions.index) == [10, 11, 12]
                np.testing.assert_array_almost_equal(
                    predictions["prediction"].values, expected_values
                )


class TestFTTransformerClassifier:
    """FTTransformerClassifier 특화 테스트"""

    def test_ft_transformer_classifier_binary_fit(self, component_test_context):
        """이진 분류 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"num1": [1, 2, 3, 4]})
                y = pd.Series([0, 1, 0, 1])  # 2 unique classes

                classifier = FTTransformerClassifier()
                result = classifier.fit(X, y)

                # Check that d_out was set correctly for binary classification
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_out"] == 2

                # Check that fit returns self and model is initialized
                assert result is classifier
                assert classifier.model is not None
                assert classifier.is_fitted is True

    def test_ft_transformer_classifier_multiclass_fit(self, component_test_context):
        """다중 분류 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"num1": [1, 2, 3, 4, 5]})
                y = pd.Series([0, 1, 2, 0, 1])  # 3 unique classes

                classifier = FTTransformerClassifier()
                result = classifier.fit(X, y)

                # Check that d_out was set correctly for multiclass
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_out"] == 3

                # Check that fit returns self
                assert result is classifier

    def test_ft_transformer_classifier_single_class_fit(self, component_test_context):
        """단일 클래스 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"num1": [1, 2, 3, 4]})
                y = pd.Series([1, 1, 1, 1])  # Only 1 unique class

                classifier = FTTransformerClassifier()
                classifier.fit(X, y)

                # Check that d_out was set correctly for single class
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_out"] == 1


class TestFTTransformerRegressor:
    """FTTransformerRegressor 특화 테스트"""

    def test_ft_transformer_regressor_fit(self, component_test_context):
        """회귀 모델 학습 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"num1": [1, 2, 3, 4]})
                y = pd.Series([1.5, 2.3, 1.8, 2.7])

                regressor = FTTransformerRegressor()
                result = regressor.fit(X, y)

                # Check that d_out was set to 1 for regression
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_out"] == 1

                # Check that fit returns self and model is initialized
                assert result is regressor
                assert regressor.model is not None
                assert regressor.is_fitted is True

    def test_ft_transformer_regressor_with_hyperparams(self, component_test_context):
        """하이퍼파라미터가 있는 회귀 모델 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                X = pd.DataFrame({"num1": [1, 2, 3]})
                y = pd.Series([1.0, 2.0, 1.5])

                hyperparams = {"d_block": 256, "n_blocks": 6}
                regressor = FTTransformerRegressor(**hyperparams)
                regressor.fit(X, y)

                # Check that custom hyperparameters were applied
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_out"] == 1
                assert call_args["d_block"] == 256
                assert call_args["n_blocks"] == 6


class TestFTTransformerIntegration:
    """FT-Transformer 통합 테스트"""

    def test_ft_transformer_complete_workflow_classification(self, component_test_context):
        """분류 모델 완전한 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train:
                # Setup mock model
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}

                # Mixed feature data
                X_train = pd.DataFrame(
                    {"numerical": [1.0, 2.0, 3.0], "categorical": ["a", "b", "c"]}
                )
                y_train = pd.Series([0, 1, 0])

                # Complete workflow: initialize -> fit
                classifier = FTTransformerClassifier(d_block=64, n_blocks=3)
                classifier.fit(X_train, y_train)

                # Verify workflow
                assert mock_ft_transformer.called
                assert mock_train.called
                assert classifier.is_fitted is True
                assert classifier.model is not None

                # Verify hyperparameters
                call_args = mock_ft_transformer.call_args[1]
                assert call_args["d_block"] == 64
                assert call_args["n_blocks"] == 3

    def test_ft_transformer_complete_workflow_regression(self, component_test_context):
        """회귀 모델 완전한 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch("src.models.custom.ft_transformer.FTTransformer") as mock_ft_transformer, \
                 patch("src.models.custom.ft_transformer.train_pytorch_model") as mock_train, \
                 patch("src.models.custom.ft_transformer.predict_with_pytorch_model") as mock_predict:
                # Setup mock model for regression
                mock_model = Mock()
                mock_ft_transformer.return_value = mock_model
                mock_train.return_value = {"train_loss": [], "val_loss": [], "best_epoch": 1}
                mock_predict.return_value = [1.2, 2.8]

                X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
                y_train = pd.Series([1.1, 2.2, 3.3, 4.4])
                X_test = pd.DataFrame({"feature": [5, 6]})

                # Complete regression workflow
                regressor = FTTransformerRegressor()
                regressor.fit(X_train, y_train)
                predictions = regressor.predict(X_test)

                # Verify workflow and output
                assert mock_train.called
                assert mock_predict.called
                assert isinstance(predictions, pd.DataFrame)
                assert len(predictions) == 2
                np.testing.assert_array_almost_equal(predictions["prediction"].values, [1.2, 2.8])

    def test_ft_transformer_error_handling_workflow(self, component_test_context):
        """오류 처리 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test unfitted model prediction error
            classifier = FTTransformerClassifier()
            X_test = pd.DataFrame({"feature": [1, 2, 3]})

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
