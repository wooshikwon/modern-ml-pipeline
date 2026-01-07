"""
Optuna integration utilities comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/utils/integrations/optuna_integration.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import sys
from unittest.mock import Mock, patch

import pytest

from src.utils.integrations.optuna_integration import (
    OptunaIntegration,
    _require_optuna,
    logging_callback,
)


class TestOptunaRequirement:
    """Optuna 패키지 요구사항 테스트"""

    @patch("src.utils.integrations.optuna_integration.optuna", create=True)
    def test_require_optuna_success(self, mock_optuna):
        """Optuna 성공적으로 import 되는 경우 테스트"""
        mock_optuna_module = Mock()

        with patch.dict(sys.modules, {"optuna": mock_optuna_module}):
            result = _require_optuna()
            assert result == mock_optuna_module

    def test_require_optuna_missing_package(self):
        """Optuna 패키지가 설치되지 않은 경우 테스트"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "optuna":
                raise ImportError("No module named 'optuna'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError) as exc_info:
                _require_optuna()

        assert "Optuna가 설치되지 않았습니다" in str(exc_info.value)
        assert "hyperparameter tuning을 사용하려면 optuna를 설치하세요" in str(exc_info.value)


class TestOptunaLoggingCallback:
    """Optuna logging callback 테스트"""

    def test_logging_callback_with_valid_trial(self, component_test_context):
        """Valid trial과 study로 logging callback 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock study and trial
            mock_study = Mock()
            mock_study.best_value = 0.95123

            mock_trial = Mock()
            mock_trial.value = 0.87654
            mock_trial.number = 5

            with patch("src.utils.integrations.optuna_integration.logger") as mock_logger:
                logging_callback(mock_study, mock_trial)

                # Verify logging call
                mock_logger.info.assert_called_once()
                call_args = str(mock_logger.info.call_args)
                assert "Trial 5 완료" in call_args
                assert "0.87654" in call_args  # Current value
                assert "0.95123" in call_args  # Best value

    def test_logging_callback_with_pruned_trial(self, component_test_context):
        """Pruned trial (value=None)에 대한 logging callback 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_study = Mock()
            mock_study.best_value = 0.95123

            # Pruned trial has no value
            mock_trial = Mock()
            mock_trial.value = None
            mock_trial.number = 3

            with patch("src.utils.integrations.optuna_integration.logger") as mock_logger:
                logging_callback(mock_study, mock_trial)

                call_args = str(mock_logger.info.call_args)
                assert "Trial 3 완료" in call_args
                assert "N/A (pruned)" in call_args  # Pruned value
                assert "0.95123" in call_args  # Best value

    def test_logging_callback_with_no_best_value(self, component_test_context):
        """Best value가 없는 경우 logging callback 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_study = Mock()
            mock_study.best_value = None  # No best value yet

            mock_trial = Mock()
            mock_trial.value = 0.75000
            mock_trial.number = 1

            with patch("src.utils.integrations.optuna_integration.logger") as mock_logger:
                logging_callback(mock_study, mock_trial)

                call_args = str(mock_logger.info.call_args)
                assert "Trial 1 완료" in call_args
                assert "0.75000" in call_args  # Current value
                assert "N/A" in call_args  # No best value

    def test_logging_callback_exception_handling(self, component_test_context):
        """Callback에서 exception 발생 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create broken mock objects
            mock_study = Mock()
            mock_study.best_value = Mock(side_effect=Exception("Broken"))

            mock_trial = Mock()
            mock_trial.value = Mock(side_effect=Exception("Broken"))

            with patch("src.utils.integrations.optuna_integration.logger") as mock_logger:
                # Should not raise exception
                logging_callback(mock_study, mock_trial)

                # Should log fallback message with warning level
                mock_logger.warning.assert_called_with("[TRAIN:HPO] Optuna 콜백에서 정보를 읽을 수 없습니다.")


class TestOptunaIntegrationClass:
    """OptunaIntegration 클래스 테스트 - Context 클래스 기반"""

    def test_optuna_integration_initialization(self, component_test_context):
        """OptunaIntegration 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            tuning_config = {"max_trials": 10, "timeout": 300}

            optuna_integration = OptunaIntegration(
                tuning_config=tuning_config, seed=42, timeout=600, n_jobs=2, pruning={"patience": 5}
            )

            assert optuna_integration.tuning_config == tuning_config
            assert optuna_integration.seed == 42
            assert optuna_integration.timeout == 600
            assert optuna_integration.n_jobs == 2
            assert optuna_integration.pruning == {"patience": 5}

    def test_optuna_integration_default_pruning(self, component_test_context):
        """OptunaIntegration 기본 pruning 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            optuna_integration = OptunaIntegration(tuning_config={}, seed=42)

            assert optuna_integration.pruning == {}

    @patch("src.utils.integrations.optuna_integration._require_optuna")
    def test_create_study_with_seed(self, mock_require_optuna, component_test_context):
        """Seed가 있는 study 생성 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock optuna module
            mock_optuna = Mock()
            mock_sampler = Mock()
            mock_study = Mock()

            mock_optuna.samplers.TPESampler.return_value = mock_sampler
            mock_optuna.create_study.return_value = mock_study
            mock_require_optuna.return_value = mock_optuna

            optuna_integration = OptunaIntegration(tuning_config={}, seed=42)

            result_study = optuna_integration.create_study(
                direction="maximize", study_name="test_study"
            )

            # Verify study creation
            assert result_study == mock_study
            mock_optuna.samplers.TPESampler.assert_called_once_with(seed=42)
            mock_optuna.create_study.assert_called_once_with(
                direction="maximize", study_name="test_study", sampler=mock_sampler, pruner=None
            )

    @patch("src.utils.integrations.optuna_integration._require_optuna")
    def test_create_study_without_seed(self, mock_require_optuna, component_test_context):
        """Seed가 없는 study 생성 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_optuna = Mock()
            mock_sampler = Mock()
            mock_study = Mock()

            mock_optuna.samplers.TPESampler.return_value = mock_sampler
            mock_optuna.create_study.return_value = mock_study
            mock_require_optuna.return_value = mock_optuna

            optuna_integration = OptunaIntegration(tuning_config={}, seed=None)

            result_study = optuna_integration.create_study(
                direction="minimize", study_name="no_seed_study"
            )

            # Verify sampler created without seed
            mock_optuna.samplers.TPESampler.assert_called_once_with()


class TestOptunaHyperparameterSuggestion:
    """하이퍼파라미터 제안 기능 테스트"""

    @patch("src.utils.integrations.optuna_integration._require_optuna")
    def test_suggest_hyperparameters_int_type(self, mock_require_optuna, component_test_context):
        """Integer 타입 하이퍼파라미터 제안 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_require_optuna.return_value = Mock()  # Mock optuna import

            mock_trial = Mock()
            mock_trial.suggest_int.return_value = 100

            optuna_integration = OptunaIntegration(tuning_config={})

            param_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 200, "log": False},
                "fixed_param": "fixed_value",
            }

            result = optuna_integration.suggest_hyperparameters(mock_trial, param_space)

            # Verify int parameter suggestion
            mock_trial.suggest_int.assert_called_once_with("n_estimators", 50, 200, log=False)
            assert result["n_estimators"] == 100
            assert result["fixed_param"] == "fixed_value"

    @patch("src.utils.integrations.optuna_integration._require_optuna")
    def test_suggest_hyperparameters_float_type(self, mock_require_optuna, component_test_context):
        """Float 타입 하이퍼파라미터 제안 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_require_optuna.return_value = Mock()

            mock_trial = Mock()
            mock_trial.suggest_float.return_value = 0.01

            optuna_integration = OptunaIntegration(tuning_config={})

            param_space = {
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True}
            }

            result = optuna_integration.suggest_hyperparameters(mock_trial, param_space)

            mock_trial.suggest_float.assert_called_once_with("learning_rate", 0.001, 0.1, log=True)
            assert result["learning_rate"] == 0.01

    @patch("src.utils.integrations.optuna_integration._require_optuna")
    def test_suggest_hyperparameters_categorical_type(
        self, mock_require_optuna, component_test_context
    ):
        """Categorical 타입 하이퍼파라미터 제안 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_require_optuna.return_value = Mock()

            mock_trial = Mock()
            mock_trial.suggest_categorical.return_value = "gini"

            optuna_integration = OptunaIntegration(tuning_config={})

            param_space = {
                "criterion": {"type": "categorical", "choices": ["gini", "entropy", "log_loss"]}
            }

            result = optuna_integration.suggest_hyperparameters(mock_trial, param_space)

            mock_trial.suggest_categorical.assert_called_once_with(
                "criterion", ["gini", "entropy", "log_loss"]
            )
            assert result["criterion"] == "gini"

    @patch("src.utils.integrations.optuna_integration._require_optuna")
    def test_suggest_hyperparameters_unknown_type(
        self, mock_require_optuna, component_test_context
    ):
        """알 수 없는 타입은 고정값으로 처리하는 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_require_optuna.return_value = Mock()

            mock_trial = Mock()
            optuna_integration = OptunaIntegration(tuning_config={})

            param_space = {"unknown_type_param": {"type": "unknown_type", "value": "some_value"}}

            result = optuna_integration.suggest_hyperparameters(mock_trial, param_space)

            # Should return the entire dict as is
            assert result["unknown_type_param"] == {"type": "unknown_type", "value": "some_value"}

    @patch("src.utils.integrations.optuna_integration._require_optuna")
    def test_suggest_hyperparameters_mixed_params(
        self, mock_require_optuna, component_test_context
    ):
        """여러 타입이 혼재된 파라미터 스페이스 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_require_optuna.return_value = Mock()

            mock_trial = Mock()
            mock_trial.suggest_int.return_value = 150
            mock_trial.suggest_float.return_value = 0.05
            mock_trial.suggest_categorical.return_value = "auto"

            optuna_integration = OptunaIntegration(tuning_config={})

            param_space = {
                "n_estimators": {"type": "int", "low": 100, "high": 200},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": False},
                "max_features": {"type": "categorical", "choices": ["auto", "sqrt", "log2"]},
                "random_state": 42,  # Fixed parameter
                "nested_config": {"param": "value"},  # Non-optuna parameter
            }

            result = optuna_integration.suggest_hyperparameters(mock_trial, param_space)

            # Verify all parameter types handled correctly
            assert result["n_estimators"] == 150
            assert result["learning_rate"] == 0.05
            assert result["max_features"] == "auto"
            assert result["random_state"] == 42
            assert result["nested_config"] == {"param": "value"}


class TestOptunaIntegrationIntegration:
    """OptunaIntegration 통합 시나리오 테스트"""

    @patch("src.utils.integrations.optuna_integration._require_optuna")
    def test_complete_hyperparameter_tuning_workflow(
        self, mock_require_optuna, component_test_context
    ):
        """완전한 하이퍼파라미터 튜닝 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock optuna components
            mock_optuna = Mock()
            mock_study = Mock()
            mock_trial = Mock()

            mock_optuna.samplers.TPESampler.return_value = Mock()
            mock_optuna.create_study.return_value = mock_study
            mock_require_optuna.return_value = mock_optuna

            # Mock trial suggestions
            mock_trial.suggest_int.return_value = 100
            mock_trial.suggest_float.return_value = 0.01
            mock_trial.suggest_categorical.return_value = "gini"

            # Initialize integration
            optuna_integration = OptunaIntegration(
                tuning_config={"max_trials": 10}, seed=42, timeout=300
            )

            # Step 1: Create study
            study = optuna_integration.create_study(
                direction="maximize", study_name="integration_test"
            )
            assert study == mock_study

            # Step 2: Suggest hyperparameters
            param_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 150},
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
                "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
                "random_state": 42,
            }

            suggested_params = optuna_integration.suggest_hyperparameters(mock_trial, param_space)

            # Verify complete workflow
            assert suggested_params["n_estimators"] == 100
            assert suggested_params["learning_rate"] == 0.01
            assert suggested_params["criterion"] == "gini"
            assert suggested_params["random_state"] == 42
