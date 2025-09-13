"""
Tests for Catalog Validator - Component Catalog Validation

Phase 1에서 구현된 CatalogValidator가 컴포넌트 카탈로그 기반으로
Recipe와 Config의 컴포넌트들을 올바르게 검증하는지 테스트합니다.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from src.settings.validation.catalog_validator import CatalogValidator, ValidationError


class TestCatalogValidatorInitialization:
    """CatalogValidator 초기화 및 기본 동작 테스트"""

    def test_validator_initialization(self):
        """CatalogValidator 초기화"""
        validator = CatalogValidator()

        assert validator.registry is not None
        assert validator.errors == []
        assert validator.warnings == []

    def test_clear_messages(self):
        """에러와 경고 메시지 초기화"""
        validator = CatalogValidator()
        validator.errors.append("test error")
        validator.warnings.append("test warning")

        validator.clear_messages()

        assert validator.errors == []
        assert validator.warnings == []

    def test_add_error_with_logging(self):
        """에러 메시지 추가 및 로깅"""
        validator = CatalogValidator()

        with patch('src.settings.validation.catalog_validator.logger') as mock_logger:
            validator.add_error("Test error message")

            assert "Test error message" in validator.errors
            mock_logger.error.assert_called_once_with("[catalog_validator] Test error message")

    def test_add_warning_with_logging(self):
        """경고 메시지 추가 및 로깅"""
        validator = CatalogValidator()

        with patch('src.settings.validation.catalog_validator.logger') as mock_logger:
            validator.add_warning("Test warning message")

            assert "Test warning message" in validator.warnings
            mock_logger.warning.assert_called_once_with("[catalog_validator] Test warning message")


class TestCatalogValidatorPreprocessorValidation:
    """전처리 스텝 검증 테스트"""

    def test_validate_preprocessor_steps_none_config(self):
        """전처리 설정이 None인 경우"""
        validator = CatalogValidator()

        result = validator.validate_preprocessor_steps(None)

        assert result is True
        assert validator.errors == []

    def test_validate_preprocessor_steps_empty_steps(self):
        """전처리 스텝이 빈 경우"""
        validator = CatalogValidator()
        config = {"steps": []}

        result = validator.validate_preprocessor_steps(config)

        assert result is True
        assert validator.errors == []

    def test_validate_preprocessor_steps_invalid_structure(self):
        """전처리 스텝이 리스트가 아닌 경우"""
        validator = CatalogValidator()
        config = {"steps": "invalid"}

        result = validator.validate_preprocessor_steps(config)

        assert result is False
        assert any("must be a list" in error for error in validator.errors)

    def test_validate_preprocessor_steps_valid_steps(self):
        """유효한 전처리 스텝들"""
        validator = CatalogValidator()
        config = {
            "steps": [
                {"type": "standard_scaler", "columns": ["feature1"]},
                {"type": "one_hot_encoder", "columns": ["category"]}
            ]
        }

        with patch.object(validator.registry, 'get_available_preprocessor_steps',
                         return_value=['standard_scaler', 'one_hot_encoder']):
            with patch.object(validator.registry, 'is_preprocessor_step_available',
                             side_effect=lambda x: x in ['standard_scaler', 'one_hot_encoder']):
                result = validator.validate_preprocessor_steps(config)

                assert result is True
                assert validator.errors == []

    def test_validate_preprocessor_steps_invalid_type(self):
        """등록되지 않은 전처리 스텝 타입"""
        validator = CatalogValidator()
        config = {
            "steps": [
                {"type": "invalid_scaler"}
            ]
        }

        with patch.object(validator.registry, 'get_available_preprocessor_steps',
                         return_value=['standard_scaler']):
            with patch.object(validator.registry, 'is_preprocessor_step_available',
                             return_value=False):
                result = validator.validate_preprocessor_steps(config)

                assert result is False
                assert any("is not registered" in error for error in validator.errors)
                assert any("invalid_scaler" in error for error in validator.errors)

    def test_validate_preprocessor_steps_missing_type(self):
        """type 필드가 없는 전처리 스텝"""
        validator = CatalogValidator()
        config = {
            "steps": [
                {"columns": ["feature1"]}  # type 필드 없음
            ]
        }

        result = validator.validate_preprocessor_steps(config)

        assert result is False
        assert any("is missing 'type' field" in error for error in validator.errors)

    def test_validate_preprocessor_steps_no_registry_warning(self):
        """전처리 Registry가 비어있는 경우 경고"""
        validator = CatalogValidator()
        config = {
            "steps": [
                {"type": "standard_scaler"}
            ]
        }

        with patch.object(validator.registry, 'get_available_preprocessor_steps',
                         return_value=[]):
            result = validator.validate_preprocessor_steps(config)

            assert any("No preprocessor steps are registered" in warning
                      for warning in validator.warnings)


class TestCatalogValidatorTaskValidation:
    """태스크 타입 검증 테스트"""

    def test_validate_task_type_valid(self):
        """유효한 태스크 타입"""
        validator = CatalogValidator()

        with patch.object(validator.registry, 'get_available_task_types',
                         return_value=['classification', 'regression']):
            with patch.object(validator.registry, 'is_task_type_available',
                             return_value=True):
                result = validator.validate_task_type('classification')

                assert result is True
                assert validator.errors == []

    def test_validate_task_type_invalid(self):
        """유효하지 않은 태스크 타입"""
        validator = CatalogValidator()

        with patch.object(validator.registry, 'get_available_task_types',
                         return_value=['classification', 'regression']):
            with patch.object(validator.registry, 'is_task_type_available',
                             return_value=False):
                result = validator.validate_task_type('invalid_task')

                assert result is False
                assert any("is not supported" in error for error in validator.errors)
                assert any("invalid_task" in error for error in validator.errors)

    def test_validate_task_type_no_registry_warning(self):
        """태스크 Registry가 비어있는 경우 경고"""
        validator = CatalogValidator()

        with patch.object(validator.registry, 'get_available_task_types',
                         return_value=[]):
            result = validator.validate_task_type('classification')

            assert result is True  # Registry가 비어있어도 True 반환
            assert any("No task types are registered" in warning
                      for warning in validator.warnings)


class TestCatalogValidatorFetcherValidation:
    """Fetcher 검증 테스트"""

    def test_validate_fetcher_type_valid(self):
        """유효한 Fetcher 타입"""
        validator = CatalogValidator()
        fetcher_config = {"type": "feature_store"}

        with patch.object(validator.registry, 'get_available_fetchers',
                         return_value=['feature_store', 'pass_through']):
            with patch.object(validator.registry, 'is_fetcher_available',
                             return_value=True):
                result = validator.validate_fetcher_type(fetcher_config)

                assert result is True
                assert validator.errors == []

    def test_validate_fetcher_type_missing_type(self):
        """type 필드가 없는 Fetcher 설정"""
        validator = CatalogValidator()
        fetcher_config = {"feature_views": {}}

        result = validator.validate_fetcher_type(fetcher_config)

        assert result is False
        assert any("is missing 'type' field" in error for error in validator.errors)

    def test_validate_fetcher_type_invalid(self):
        """유효하지 않은 Fetcher 타입"""
        validator = CatalogValidator()
        fetcher_config = {"type": "invalid_fetcher"}

        with patch.object(validator.registry, 'get_available_fetchers',
                         return_value=['feature_store']):
            with patch.object(validator.registry, 'is_fetcher_available',
                             return_value=False):
                result = validator.validate_fetcher_type(fetcher_config)

                assert result is False
                assert any("is not registered" in error for error in validator.errors)
                assert any("invalid_fetcher" in error for error in validator.errors)


class TestCatalogValidatorAdapterValidation:
    """Adapter 검증 테스트"""

    def test_validate_adapter_type_valid(self):
        """유효한 Adapter 타입"""
        validator = CatalogValidator()

        with patch.object(validator.registry, 'get_available_adapters',
                         return_value=['sql', 'storage']):
            with patch.object(validator.registry, 'is_adapter_available',
                             return_value=True):
                result = validator.validate_adapter_type('sql')

                assert result is True
                assert validator.errors == []

    def test_validate_adapter_type_invalid(self):
        """유효하지 않은 Adapter 타입"""
        validator = CatalogValidator()

        with patch.object(validator.registry, 'get_available_adapters',
                         return_value=['sql', 'storage']):
            with patch.object(validator.registry, 'is_adapter_available',
                             return_value=False):
                result = validator.validate_adapter_type('invalid_adapter')

                assert result is False
                assert any("is not registered" in error for error in validator.errors)


class TestCatalogValidatorCalibrationValidation:
    """Calibration 검증 테스트"""

    def test_validate_calibration_method_disabled(self):
        """캘리브레이션이 비활성화된 경우"""
        validator = CatalogValidator()
        calibration_config = {"enabled": False}

        result = validator.validate_calibration_method(calibration_config)

        assert result is True
        assert validator.errors == []

    def test_validate_calibration_method_none_config(self):
        """캘리브레이션 설정이 None인 경우"""
        validator = CatalogValidator()

        result = validator.validate_calibration_method(None)

        assert result is True
        assert validator.errors == []

    def test_validate_calibration_method_valid(self):
        """유효한 캘리브레이션 방법"""
        validator = CatalogValidator()
        calibration_config = {"enabled": True, "method": "beta"}

        with patch.object(validator.registry, 'get_available_calibration_methods',
                         return_value=['beta', 'isotonic']):
            with patch.object(validator.registry, 'is_calibration_method_available',
                             return_value=True):
                result = validator.validate_calibration_method(calibration_config)

                assert result is True
                assert validator.errors == []

    def test_validate_calibration_method_missing_method(self):
        """캘리브레이션이 활성화되었지만 method가 없는 경우"""
        validator = CatalogValidator()
        calibration_config = {"enabled": True}

        result = validator.validate_calibration_method(calibration_config)

        assert result is False
        assert any("'method' is not specified" in error for error in validator.errors)

    def test_validate_calibration_method_invalid(self):
        """유효하지 않은 캘리브레이션 방법"""
        validator = CatalogValidator()
        calibration_config = {"enabled": True, "method": "invalid_method"}

        with patch.object(validator.registry, 'get_available_calibration_methods',
                         return_value=['beta', 'isotonic']):
            with patch.object(validator.registry, 'is_calibration_method_available',
                             return_value=False):
                result = validator.validate_calibration_method(calibration_config)

                assert result is False
                assert any("is not registered" in error for error in validator.errors)


class TestCatalogValidatorRecipeValidation:
    """Recipe 전체 검증 테스트"""

    def test_validate_recipe_components_success(self):
        """Recipe 컴포넌트 검증 성공"""
        validator = CatalogValidator()
        recipe_data = {
            "task_choice": "classification",
            "preprocessor": {
                "steps": [
                    {"type": "standard_scaler", "columns": ["feature1"]}
                ]
            },
            "data": {
                "fetcher": {"type": "feature_store"}
            },
            "model": {
                "calibration": {"enabled": True, "method": "beta"}
            }
        }

        # Mock all registry calls to return valid results
        with patch.object(validator, 'validate_task_type', return_value=True), \
             patch.object(validator, 'validate_preprocessor_steps', return_value=True), \
             patch.object(validator, 'validate_fetcher_type', return_value=True), \
             patch.object(validator, 'validate_calibration_method', return_value=True):

            result = validator.validate_recipe_components(recipe_data)

            assert result is True
            assert validator.errors == []

    def test_validate_recipe_components_partial_failure(self):
        """Recipe 컴포넌트 검증 부분 실패"""
        validator = CatalogValidator()
        recipe_data = {
            "task_choice": "invalid_task",
            "preprocessor": {
                "steps": [
                    {"type": "invalid_step"}
                ]
            }
        }

        # Mock some validations to fail
        with patch.object(validator, 'validate_task_type', return_value=False), \
             patch.object(validator, 'validate_preprocessor_steps', return_value=False):

            result = validator.validate_recipe_components(recipe_data)

            assert result is False

    def test_validate_recipe_components_empty_data(self):
        """빈 Recipe 데이터"""
        validator = CatalogValidator()
        recipe_data = {}

        result = validator.validate_recipe_components(recipe_data)

        assert result is True  # 빈 데이터는 유효


class TestCatalogValidatorConfigValidation:
    """Config 전체 검증 테스트"""

    def test_validate_config_components_success(self):
        """Config 컴포넌트 검증 성공"""
        validator = CatalogValidator()
        config_data = {
            "data_source": {
                "adapter_type": "sql"
            },
            "output": {
                "inference": {
                    "adapter_type": "storage"
                },
                "preprocessed": {
                    "adapter_type": "sql"
                }
            }
        }

        with patch.object(validator, 'validate_adapter_type', return_value=True):
            result = validator.validate_config_components(config_data)

            assert result is True
            assert validator.errors == []

    def test_validate_config_components_adapter_failure(self):
        """Config의 어댑터 검증 실패"""
        validator = CatalogValidator()
        config_data = {
            "data_source": {
                "adapter_type": "invalid_adapter"
            }
        }

        with patch.object(validator, 'validate_adapter_type', return_value=False):
            result = validator.validate_config_components(config_data)

            assert result is False


class TestCatalogValidatorUtilityMethods:
    """유틸리티 메서드 테스트"""

    def test_get_validation_summary_success(self):
        """검증 성공 요약"""
        validator = CatalogValidator()

        summary = validator.get_validation_summary()

        assert summary['is_valid'] is True
        assert summary['error_count'] == 0
        assert summary['warning_count'] == 0
        assert summary['errors'] == []
        assert summary['warnings'] == []

    def test_get_validation_summary_with_errors(self):
        """에러가 있는 검증 요약"""
        validator = CatalogValidator()
        validator.add_error("Test error")
        validator.add_warning("Test warning")

        summary = validator.get_validation_summary()

        assert summary['is_valid'] is False
        assert summary['error_count'] == 1
        assert summary['warning_count'] == 1
        assert "Test error" in summary['errors']
        assert "Test warning" in summary['warnings']

    def test_get_available_components_summary(self):
        """사용 가능한 컴포넌트 요약"""
        validator = CatalogValidator()
        expected_catalog = {
            'preprocessor_steps': ['scaler'],
            'task_types': ['classification'],
            'adapters': ['sql']
        }

        with patch.object(validator.registry, 'get_component_catalog',
                         return_value=expected_catalog):
            summary = validator.get_available_components_summary()

            assert summary == expected_catalog

    def test_validate_component_exists(self):
        """특정 컴포넌트 존재 여부 검증"""
        validator = CatalogValidator()

        with patch.object(validator.registry, 'validate_component_availability',
                         return_value=True):
            result = validator.validate_component_exists('preprocessor', 'standard_scaler')

            assert result is True

    def test_suggest_alternatives_exact_match(self):
        """대안 제안 - 정확한 매치"""
        validator = CatalogValidator()
        catalog = {
            'preprocessor_steps': ['standard_scaler', 'min_max_scaler', 'robust_scaler']
        }

        with patch.object(validator, 'get_available_components_summary',
                         return_value=catalog):
            suggestions = validator.suggest_alternatives('preprocessor', 'scaler')

            # 'scaler'를 포함하는 모든 항목 반환
            assert 'standard_scaler' in suggestions
            assert 'min_max_scaler' in suggestions
            assert 'robust_scaler' in suggestions
            assert len(suggestions) <= 3  # 최대 3개

    def test_suggest_alternatives_unknown_type(self):
        """대안 제안 - 알 수 없는 타입"""
        validator = CatalogValidator()

        suggestions = validator.suggest_alternatives('unknown_type', 'invalid_name')

        assert suggestions == []

    def test_suggest_alternatives_no_matches(self):
        """대안 제안 - 매치 없음"""
        validator = CatalogValidator()
        catalog = {
            'preprocessor_steps': ['standard_scaler']
        }

        with patch.object(validator, 'get_available_components_summary',
                         return_value=catalog):
            suggestions = validator.suggest_alternatives('preprocessor', 'xyz')

            assert suggestions == []


class TestCatalogValidatorEdgeCases:
    """CatalogValidator 경계 사례 테스트"""

    def test_validate_nested_missing_keys(self):
        """중첩된 딕셔너리에서 키가 없는 경우"""
        validator = CatalogValidator()
        recipe_data = {
            "data": {}  # fetcher 키 없음
        }

        result = validator.validate_recipe_components(recipe_data)

        # fetcher 키가 없어도 에러 없이 처리
        assert result is True

    def test_validate_preprocessor_step_dict_not_dict(self):
        """전처리 스텝이 딕셔너리가 아닌 경우"""
        validator = CatalogValidator()
        config = {
            "steps": ["not_a_dict"]
        }

        result = validator.validate_preprocessor_steps(config)

        assert result is False
        assert any("must be a dictionary" in error for error in validator.errors)

    def test_registry_unavailable_graceful_degradation(self):
        """Registry가 사용 불가능할 때 graceful degradation"""
        validator = CatalogValidator()

        # Registry 메서드들이 빈 결과 반환하도록 mock
        with patch.object(validator.registry, 'get_available_preprocessor_steps',
                         return_value=[]), \
             patch.object(validator.registry, 'is_preprocessor_step_available',
                         return_value=False):

            config = {
                "steps": [{"type": "standard_scaler"}]
            }

            result = validator.validate_preprocessor_steps(config)

            # Registry가 비어있어도 경고만 발생하고 검증은 계속
            assert any("No preprocessor steps are registered" in warning
                      for warning in validator.warnings)
            assert result is False  # 스텝을 찾을 수 없으므로 False