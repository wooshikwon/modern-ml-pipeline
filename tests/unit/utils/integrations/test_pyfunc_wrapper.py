"""
MLflow PyFunc wrapper comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/utils/integrations/pyfunc_wrapper.py

Author: Phase 2A Development
Date: 2025-09-13
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper


class TestPyfuncWrapperInitialization:
    """PyfuncWrapper 초기화 테스트 - Context 클래스 기반"""

    def test_pyfunc_wrapper_initialization_with_settings(self, component_test_context):
        """Settings 객체를 사용한 PyfuncWrapper 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock trained components
            mock_model = Mock()
            mock_calibrator = Mock()

            # Create PyfuncWrapper
            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_calibrator=mock_calibrator,
                training_results={"accuracy": 0.95},
                data_interface_schema={
                    "feature_columns": ["feature_0", "feature_1"],
                    "target_column": "target",
                },
            )

            # Verify initialization
            assert wrapper.trained_model == mock_model
            assert wrapper.trained_calibrator == mock_calibrator
            assert wrapper.training_results == {"accuracy": 0.95}
            assert wrapper._task_type == "classification"  # From context
            assert "recipe" in wrapper.settings_dict

    def test_pyfunc_wrapper_initialization_with_dict_settings(self, component_test_context):
        """Dict 형태 설정을 사용한 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()

            # Dict-based settings
            settings_dict = {
                "recipe": {
                    "task_choice": "regression",
                    "model": {"class_path": "sklearn.linear_model.LinearRegression"},
                }
            }

            wrapper = PyfuncWrapper(settings=settings_dict, trained_model=mock_model)

            assert wrapper._task_type == "regression"
            assert wrapper.settings_dict == settings_dict

    def test_pyfunc_wrapper_extract_serializable_settings_exception_handling(
        self, component_test_context
    ):
        """설정 추출 시 예외 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()

            # Broken settings that raise exceptions
            class Broken:
                def __getattr__(self, name):
                    raise Exception("Broken")

            broken_settings = Broken()

            wrapper = PyfuncWrapper(settings=broken_settings, trained_model=mock_model)

            # Should fallback to unknown
            assert wrapper._task_type == "unknown"
            assert wrapper.settings_dict == {"recipe": {"task_choice": "unknown"}}


class TestPyfuncWrapperProperties:
    """Property 접근자 테스트"""

    def test_model_class_path_property(self, component_test_context):
        """model_class_path property 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()

            wrapper = PyfuncWrapper(settings=ctx.settings, trained_model=mock_model)

            class_path = wrapper.model_class_path
            # Should extract class_path from settings
            assert isinstance(class_path, str)

    def test_loader_sql_snapshot_property(self, component_test_context):
        """loader_sql_snapshot property 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()

            wrapper = PyfuncWrapper(settings=ctx.settings, trained_model=mock_model)

            sql_snapshot = wrapper.loader_sql_snapshot
            assert isinstance(sql_snapshot, str)

    def test_recipe_yaml_snapshot_property(self, component_test_context):
        """recipe_yaml_snapshot property 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()

            wrapper = PyfuncWrapper(settings=ctx.settings, trained_model=mock_model)

            yaml_snapshot = wrapper.recipe_yaml_snapshot
            assert isinstance(yaml_snapshot, str)
            # Should be valid YAML
            parsed = yaml.safe_load(yaml_snapshot)
            assert isinstance(parsed, dict)

    def test_hyperparameter_optimization_property(self, component_test_context):
        """hyperparameter_optimization property 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()

            training_results = {
                "hyperparameter_optimization": {
                    "best_params": {"n_estimators": 100},
                    "best_score": 0.95,
                }
            }

            wrapper = PyfuncWrapper(
                settings=ctx.settings, trained_model=mock_model, training_results=training_results
            )

            hpo_results = wrapper.hyperparameter_optimization
            assert hpo_results["best_params"] == {"n_estimators": 100}
            assert hpo_results["best_score"] == 0.95

    def test_training_methodology_property(self, component_test_context):
        """training_methodology property 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()

            training_results = {"training_methodology": {"cross_validation": True, "splits": 5}}

            wrapper = PyfuncWrapper(
                settings=ctx.settings, trained_model=mock_model, training_results=training_results
            )

            methodology = wrapper.training_methodology
            assert methodology["cross_validation"] is True
            assert methodology["splits"] == 5


class TestPyfuncWrapperPredict:
    """예측 기능 테스트 - Context 클래스 기반"""

    def test_predict_basic_classification(self, component_test_context):
        """기본 분류 예측 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock model with predictions
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0, 1, 0])

            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                data_interface_schema={
                    "feature_columns": ["feature_0", "feature_1"],
                    "target_column": "target",
                },
            )

            # Test data
            test_input = pd.DataFrame(
                {
                    "feature_0": [1.0, 2.0, 3.0],
                    "feature_1": [0.5, 1.5, 2.5],
                    "target": [0, 1, 0],  # Should be excluded from features
                }
            )

            # Predict
            predictions = wrapper.predict(context=None, model_input=test_input)

            # Verify predictions
            assert len(predictions) == 3
            mock_model.predict.assert_called_once()

            # Verify correct features were used (target excluded)
            call_args = mock_model.predict.call_args[0][0]
            assert list(call_args.columns) == ["feature_0", "feature_1"]

    def test_predict_with_probabilities(self, component_test_context):
        """확률 예측 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock model with probability predictions
            mock_model = Mock()
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])

            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                data_interface_schema={"feature_columns": ["feature_0", "feature_1"]},
            )

            test_input = pd.DataFrame({"feature_0": [1.0, 2.0, 3.0], "feature_1": [0.5, 1.5, 2.5]})

            # Predict with probabilities
            predictions = wrapper.predict(
                context=None, model_input=test_input, params={"return_probabilities": True}
            )

            mock_model.predict_proba.assert_called_once()
            assert hasattr(predictions, "__len__")
            assert len(predictions) == 3

    def test_predict_with_calibration(self, component_test_context):
        """캘리브레이션 적용 예측 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock model and calibrator
            mock_model = Mock()
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])

            mock_calibrator = Mock()
            mock_calibrator.transform.return_value = np.array(
                [[0.4, 0.6], [0.7, 0.3]]
            )  # Calibrated

            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_calibrator=mock_calibrator,
                data_interface_schema={"feature_columns": ["feature_0", "feature_1"]},
            )

            test_input = pd.DataFrame({"feature_0": [1.0, 2.0], "feature_1": [0.5, 1.5]})

            # Test calibrated probability prediction
            predictions = wrapper.predict(
                context=None, model_input=test_input, params={"return_probabilities": True}
            )

            mock_model.predict_proba.assert_called_once()
            mock_calibrator.transform.assert_called_once()

    def test_predict_return_dataframe(self, component_test_context):
        """DataFrame 반환 형태 예측 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0, 1, 0])

            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                data_interface_schema={"feature_columns": ["feature_0", "feature_1"]},
            )

            test_input = pd.DataFrame({"feature_0": [1.0, 2.0, 3.0], "feature_1": [0.5, 1.5, 2.5]})

            # Predict with DataFrame return
            predictions = wrapper.predict(
                context=None, model_input=test_input, params={"return_dataframe": True}
            )

            assert isinstance(predictions, pd.DataFrame)
            assert "prediction" in predictions.columns
            assert len(predictions) == 3

    def test_predict_without_feature_columns_fallback(self, component_test_context):
        """feature_columns 없을 때 target 제외 fallback 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0, 1])

            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                data_interface_schema={
                    "target_column": "target"
                    # feature_columns 없음
                },
            )

            test_input = pd.DataFrame(
                {"feature_0": [1.0, 2.0], "feature_1": [0.5, 1.5], "target": [0, 1]}
            )

            predictions = wrapper.predict(context=None, model_input=test_input)

            # Should use all columns except target
            call_args = mock_model.predict.call_args[0][0]
            assert "target" not in call_args.columns
            assert "feature_0" in call_args.columns
            assert "feature_1" in call_args.columns


class TestPyfuncWrapperErrorHandling:
    """예측 오류 처리 테스트"""

    def test_predict_exception_propagation(self, component_test_context):
        """예측 실패 시 예외가 명확히 전파되는지 테스트"""
        with component_test_context.classification_stack() as ctx:
            # 항상 실패하는 Mock 모델
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Prediction failed")

            wrapper = PyfuncWrapper(settings=ctx.settings, trained_model=mock_model)

            test_input = pd.DataFrame({"feature_0": [1.0, 2.0], "feature_1": [0.5, 1.5]})

            # 예외가 전파되어야 함 (Silent Failure 방지)
            with pytest.raises(Exception, match="Prediction failed"):
                wrapper.predict(context=None, model_input=test_input)

    def test_predict_non_dataframe_input_conversion(self, component_test_context):
        """DataFrame이 아닌 입력의 변환 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0, 1])

            wrapper = PyfuncWrapper(settings=ctx.settings, trained_model=mock_model)

            # List input (not DataFrame)
            test_input = [[1.0, 0.5], [2.0, 1.5]]

            predictions = wrapper.predict(context=None, model_input=test_input)

            # Should convert to DataFrame and predict
            assert len(predictions) == 2


class TestPyfuncWrapperIntegration:
    """통합 시나리오 테스트"""

    def test_complete_prediction_pipeline(self, component_test_context):
        """완전한 예측 파이프라인 통합 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Setup complete wrapper with all components
            mock_model = Mock()
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])

            mock_calibrator = Mock()
            mock_calibrator.transform.return_value = np.array([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])

            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_calibrator=mock_calibrator,
                training_results={
                    "hyperparameter_optimization": {"best_score": 0.95},
                    "training_methodology": {"cv_folds": 5},
                },
                data_interface_schema={
                    "feature_columns": ["feature_0", "feature_1", "feature_2"],
                    "target_column": "target",
                },
            )

            # Test comprehensive input
            test_input = pd.DataFrame(
                {
                    "feature_0": [1.0, 2.0, 3.0],
                    "feature_1": [0.5, 1.5, 2.5],
                    "feature_2": [0.1, 0.2, 0.3],
                    "target": [0, 1, 0],
                    "extra_column": ["a", "b", "c"],  # Should be ignored
                }
            )

            # Test various prediction modes
            # 1. Basic class predictions with calibration
            class_predictions = wrapper.predict(context=None, model_input=test_input)

            # 2. Probability predictions with calibration
            prob_predictions = wrapper.predict(
                context=None, model_input=test_input, params={"return_probabilities": True}
            )

            # 3. DataFrame format
            df_predictions = wrapper.predict(
                context=None,
                model_input=test_input,
                params={"return_dataframe": True, "return_probabilities": True},
            )

            # Verify all modes work
            assert len(class_predictions) == 3
            assert len(prob_predictions) == 3
            assert isinstance(df_predictions, pd.DataFrame)
            assert len(df_predictions) == 3

            # Verify calibration was applied
            mock_calibrator.transform.assert_called()

            # Verify only correct features used
            feature_args = mock_model.predict_proba.call_args[0][0]
            assert list(feature_args.columns) == ["feature_0", "feature_1", "feature_2"]
            assert "target" not in feature_args.columns
            assert "extra_column" not in feature_args.columns
