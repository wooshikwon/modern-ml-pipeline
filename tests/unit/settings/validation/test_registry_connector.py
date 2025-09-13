"""
Tests for Registry Connector - Component Registry Integration

Phase 1에서 구현된 RegistryConnector가 각 컴포넌트의 Registry 시스템과
올바르게 연동되는지 검증하고, lazy loading과 캐싱 동작을 확인합니다.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from typing import Dict, List, Optional, Type

from src.settings.validation.registry_connector import RegistryConnector


class TestRegistryConnectorLazyLoading:
    """RegistryConnector의 lazy loading 동작 테스트"""

    def setUp(self):
        """각 테스트 전에 레지스트리 캐시 초기화"""
        RegistryConnector._preprocessor_registry = None
        RegistryConnector._evaluator_registry = None
        RegistryConnector._trainer_registry = None
        RegistryConnector._fetcher_registry = None
        RegistryConnector._adapter_registry = None
        RegistryConnector._calibration_registry = None
        RegistryConnector._datahandler_registry = None

    def test_preprocessor_registry_lazy_import_success(self):
        """PreprocessorStepRegistry의 lazy import 성공 케이스"""
        self.setUp()

        mock_registry = MagicMock()
        mock_registry.preprocessor_steps = {
            'standard_scaler': MagicMock(),
            'one_hot_encoder': MagicMock(),
            'simple_imputer': MagicMock()
        }

        with patch('src.settings.validation.registry_connector.logger') as mock_logger:
            with patch.dict('sys.modules', {
                'src.components.preprocessor.registry': MagicMock(PreprocessorStepRegistry=mock_registry)
            }):
                registry = RegistryConnector._get_preprocessor_registry()

                assert registry == mock_registry
                assert RegistryConnector._preprocessor_registry == mock_registry
                mock_logger.debug.assert_called_with("[validation] PreprocessorStepRegistry loaded")

    def test_preprocessor_registry_lazy_import_failure(self):
        """PreprocessorStepRegistry의 lazy import 실패 케이스"""
        self.setUp()

        with patch('src.settings.validation.registry_connector.logger') as mock_logger:
            with patch('builtins.__import__', side_effect=ImportError("Module not found")):
                registry = RegistryConnector._get_preprocessor_registry()

                assert registry is None
                assert RegistryConnector._preprocessor_registry is None
                mock_logger.warning.assert_called_with(
                    "[validation] Failed to import PreprocessorStepRegistry: Module not found"
                )

    def test_evaluator_registry_lazy_import_success(self):
        """EvaluatorRegistry의 lazy import 성공 케이스"""
        self.setUp()

        mock_registry = MagicMock()
        mock_registry.evaluators = {
            'classification': MagicMock(),
            'regression': MagicMock(),
            'clustering': MagicMock()
        }

        with patch('src.settings.validation.registry_connector.logger') as mock_logger:
            with patch.dict('sys.modules', {
                'src.components.evaluator.registry': MagicMock(EvaluatorRegistry=mock_registry)
            }):
                registry = RegistryConnector._get_evaluator_registry()

                assert registry == mock_registry
                assert RegistryConnector._evaluator_registry == mock_registry
                mock_logger.debug.assert_called_with("[validation] EvaluatorRegistry loaded")

    def test_registry_caching_behavior(self):
        """Registry가 한 번 로드되면 캐싱되는지 확인"""
        self.setUp()

        mock_registry = MagicMock()

        with patch('src.settings.validation.registry_connector.logger'):
            with patch.dict('sys.modules', {
                'src.components.preprocessor.registry': MagicMock(PreprocessorStepRegistry=mock_registry)
            }) as mock_modules:
                # 첫 번째 호출
                registry1 = RegistryConnector._get_preprocessor_registry()
                # 두 번째 호출
                registry2 = RegistryConnector._get_preprocessor_registry()

                # 같은 객체 반환 (캐싱됨)
                assert registry1 is registry2
                assert registry1 == mock_registry

                # import는 첫 번째 호출에서만 발생
                assert len(mock_modules) == 1


class TestRegistryConnectorPreprocessorMethods:
    """PreprocessorStepRegistry 관련 메서드 테스트"""

    def test_get_available_preprocessor_steps_success(self):
        """사용 가능한 전처리 스텝 목록 조회 성공"""
        mock_registry = MagicMock()
        mock_registry.preprocessor_steps = {
            'standard_scaler': MagicMock(),
            'one_hot_encoder': MagicMock(),
            'simple_imputer': MagicMock()
        }

        with patch.object(RegistryConnector, '_get_preprocessor_registry', return_value=mock_registry):
            steps = RegistryConnector.get_available_preprocessor_steps()

            assert steps == ['standard_scaler', 'one_hot_encoder', 'simple_imputer']

    def test_get_available_preprocessor_steps_registry_none(self):
        """Registry가 None일 때 빈 리스트 반환"""
        with patch.object(RegistryConnector, '_get_preprocessor_registry', return_value=None):
            steps = RegistryConnector.get_available_preprocessor_steps()

            assert steps == []

    def test_is_preprocessor_step_available_true(self):
        """전처리 스텝이 사용 가능한 경우"""
        mock_registry = MagicMock()
        mock_registry.preprocessor_steps = {'standard_scaler': MagicMock()}

        with patch.object(RegistryConnector, '_get_preprocessor_registry', return_value=mock_registry):
            result = RegistryConnector.is_preprocessor_step_available('standard_scaler')

            assert result is True

    def test_is_preprocessor_step_available_false(self):
        """전처리 스텝이 사용 불가능한 경우"""
        mock_registry = MagicMock()
        mock_registry.preprocessor_steps = {'standard_scaler': MagicMock()}

        with patch.object(RegistryConnector, '_get_preprocessor_registry', return_value=mock_registry):
            result = RegistryConnector.is_preprocessor_step_available('invalid_step')

            assert result is False

    def test_get_preprocessor_step_class_success(self):
        """전처리 스텝 클래스 조회 성공"""
        mock_class = MagicMock()
        mock_registry = MagicMock()
        mock_registry.preprocessor_steps = {'standard_scaler': mock_class}

        with patch.object(RegistryConnector, '_get_preprocessor_registry', return_value=mock_registry):
            step_class = RegistryConnector.get_preprocessor_step_class('standard_scaler')

            assert step_class == mock_class

    def test_get_preprocessor_step_class_not_found(self):
        """전처리 스텝 클래스 조회 실패"""
        mock_registry = MagicMock()
        mock_registry.preprocessor_steps = {}

        with patch.object(RegistryConnector, '_get_preprocessor_registry', return_value=mock_registry):
            step_class = RegistryConnector.get_preprocessor_step_class('invalid_step')

            assert step_class is None


class TestRegistryConnectorEvaluatorMethods:
    """EvaluatorRegistry 관련 메서드 테스트"""

    def test_get_available_task_types_success(self):
        """사용 가능한 태스크 타입 목록 조회 성공"""
        mock_registry = MagicMock()
        mock_registry.get_available_tasks.return_value = ['classification', 'regression', 'clustering']

        with patch.object(RegistryConnector, '_get_evaluator_registry', return_value=mock_registry):
            tasks = RegistryConnector.get_available_task_types()

            assert tasks == ['classification', 'regression', 'clustering']
            mock_registry.get_available_tasks.assert_called_once()

    def test_get_available_task_types_registry_none(self):
        """Registry가 None일 때 빈 리스트 반환"""
        with patch.object(RegistryConnector, '_get_evaluator_registry', return_value=None):
            tasks = RegistryConnector.get_available_task_types()

            assert tasks == []

    def test_is_task_type_available_true(self):
        """태스크 타입이 사용 가능한 경우"""
        mock_registry = MagicMock()
        mock_registry.evaluators = {'classification': MagicMock()}

        with patch.object(RegistryConnector, '_get_evaluator_registry', return_value=mock_registry):
            result = RegistryConnector.is_task_type_available('classification')

            assert result is True

    def test_get_evaluator_class_success(self):
        """Evaluator 클래스 조회 성공"""
        mock_class = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_evaluator_class.return_value = mock_class

        with patch.object(RegistryConnector, '_get_evaluator_registry', return_value=mock_registry):
            evaluator_class = RegistryConnector.get_evaluator_class('classification')

            assert evaluator_class == mock_class
            mock_registry.get_evaluator_class.assert_called_once_with('classification')

    def test_get_evaluator_class_value_error(self):
        """Evaluator 클래스 조회시 ValueError 발생"""
        mock_registry = MagicMock()
        mock_registry.get_evaluator_class.side_effect = ValueError("Invalid task type")

        with patch.object(RegistryConnector, '_get_evaluator_registry', return_value=mock_registry):
            evaluator_class = RegistryConnector.get_evaluator_class('invalid_task')

            assert evaluator_class is None


class TestRegistryConnectorGenericMethods:
    """범용적인 Registry 메서드들 테스트"""

    def test_get_available_trainers_success(self):
        """사용 가능한 트레이너 목록 조회 성공"""
        mock_registry = MagicMock()
        mock_registry.trainers = {
            'sklearn_trainer': MagicMock(),
            'xgboost_trainer': MagicMock()
        }

        with patch.object(RegistryConnector, '_get_trainer_registry', return_value=mock_registry):
            trainers = RegistryConnector.get_available_trainers()

            assert trainers == ['sklearn_trainer', 'xgboost_trainer']

    def test_get_available_trainers_no_trainers_attribute(self):
        """트레이너 Registry에 trainers 속성이 없는 경우"""
        mock_registry = MagicMock(spec=[])  # trainers 속성 없음

        with patch.object(RegistryConnector, '_get_trainer_registry', return_value=mock_registry):
            trainers = RegistryConnector.get_available_trainers()

            assert trainers == []

    def test_get_available_fetchers_success(self):
        """사용 가능한 페처 목록 조회 성공"""
        mock_registry = MagicMock()
        mock_registry.fetchers = {
            'feature_store': MagicMock(),
            'pass_through': MagicMock()
        }

        with patch.object(RegistryConnector, '_get_fetcher_registry', return_value=mock_registry):
            fetchers = RegistryConnector.get_available_fetchers()

            assert fetchers == ['feature_store', 'pass_through']

    def test_get_available_adapters_success(self):
        """사용 가능한 어댑터 목록 조회 성공"""
        mock_registry = MagicMock()
        mock_registry.adapters = {
            'sql': MagicMock(),
            'storage': MagicMock()
        }

        with patch.object(RegistryConnector, '_get_adapter_registry', return_value=mock_registry):
            adapters = RegistryConnector.get_available_adapters()

            assert adapters == ['sql', 'storage']

    def test_get_available_calibration_methods_success(self):
        """사용 가능한 캘리브레이션 방법 목록 조회 성공"""
        mock_registry = MagicMock()
        mock_registry.calibrators = {
            'beta': MagicMock(),
            'isotonic': MagicMock()
        }

        with patch.object(RegistryConnector, '_get_calibration_registry', return_value=mock_registry):
            methods = RegistryConnector.get_available_calibration_methods()

            assert methods == ['beta', 'isotonic']

    def test_get_available_data_handlers_success(self):
        """사용 가능한 데이터 핸들러 목록 조회 성공"""
        mock_registry = MagicMock()
        mock_registry.handlers = {
            'classification': MagicMock(),
            'regression': MagicMock()
        }

        with patch.object(RegistryConnector, '_get_datahandler_registry', return_value=mock_registry):
            handlers = RegistryConnector.get_available_data_handlers()

            assert handlers == ['classification', 'regression']


class TestRegistryConnectorValidationMethods:
    """종합적인 검증 메서드 테스트"""

    def test_validate_component_availability_preprocessor(self):
        """Preprocessor 컴포넌트 검증"""
        with patch.object(RegistryConnector, 'is_preprocessor_step_available', return_value=True) as mock_method:
            result = RegistryConnector.validate_component_availability('preprocessor', 'standard_scaler')

            assert result is True
            mock_method.assert_called_once_with('standard_scaler')

    def test_validate_component_availability_evaluator(self):
        """Evaluator 컴포넌트 검증"""
        with patch.object(RegistryConnector, 'is_task_type_available', return_value=False) as mock_method:
            result = RegistryConnector.validate_component_availability('evaluator', 'invalid_task')

            assert result is False
            mock_method.assert_called_once_with('invalid_task')

    def test_validate_component_availability_unknown_type(self):
        """알 수 없는 컴포넌트 타입"""
        with patch('src.settings.validation.registry_connector.logger') as mock_logger:
            result = RegistryConnector.validate_component_availability('unknown_type', 'some_component')

            assert result is False
            mock_logger.warning.assert_called_with(
                "[validation] Unknown component type: unknown_type"
            )

    def test_get_component_catalog_success(self):
        """전체 컴포넌트 카탈로그 조회 성공"""
        with patch.object(RegistryConnector, 'get_available_preprocessor_steps', return_value=['scaler']) as mock1, \
             patch.object(RegistryConnector, 'get_available_task_types', return_value=['classification']) as mock2, \
             patch.object(RegistryConnector, 'get_available_trainers', return_value=['sklearn']) as mock3, \
             patch.object(RegistryConnector, 'get_available_fetchers', return_value=['feature_store']) as mock4, \
             patch.object(RegistryConnector, 'get_available_adapters', return_value=['sql']) as mock5, \
             patch.object(RegistryConnector, 'get_available_calibration_methods', return_value=['beta']) as mock6, \
             patch.object(RegistryConnector, 'get_available_data_handlers', return_value=['classification']) as mock7:

            catalog = RegistryConnector.get_component_catalog()

            expected = {
                'preprocessor_steps': ['scaler'],
                'task_types': ['classification'],
                'trainers': ['sklearn'],
                'fetchers': ['feature_store'],
                'adapters': ['sql'],
                'calibration_methods': ['beta'],
                'data_handlers': ['classification'],
            }

            assert catalog == expected
            mock1.assert_called_once()
            mock2.assert_called_once()
            mock3.assert_called_once()
            mock4.assert_called_once()
            mock5.assert_called_once()
            mock6.assert_called_once()
            mock7.assert_called_once()


class TestRegistryConnectorEdgeCases:
    """Registry Connector의 경계 사례 테스트"""

    def test_all_registries_none_graceful_handling(self):
        """모든 Registry가 None일 때 graceful 처리"""
        with patch.object(RegistryConnector, '_get_preprocessor_registry', return_value=None), \
             patch.object(RegistryConnector, '_get_evaluator_registry', return_value=None), \
             patch.object(RegistryConnector, '_get_trainer_registry', return_value=None), \
             patch.object(RegistryConnector, '_get_fetcher_registry', return_value=None), \
             patch.object(RegistryConnector, '_get_adapter_registry', return_value=None), \
             patch.object(RegistryConnector, '_get_calibration_registry', return_value=None), \
             patch.object(RegistryConnector, '_get_datahandler_registry', return_value=None):

            catalog = RegistryConnector.get_component_catalog()

            expected = {
                'preprocessor_steps': [],
                'task_types': [],
                'trainers': [],
                'fetchers': [],
                'adapters': [],
                'calibration_methods': [],
                'data_handlers': [],
            }

            assert catalog == expected

    def test_registry_attribute_missing_graceful_handling(self):
        """Registry 객체에 예상 속성이 없을 때 graceful 처리"""
        mock_registry = MagicMock(spec=[])  # 모든 속성 없음

        with patch.object(RegistryConnector, '_get_trainer_registry', return_value=mock_registry):
            trainers = RegistryConnector.get_available_trainers()
            is_available = RegistryConnector.is_trainer_available('some_trainer')

            assert trainers == []
            assert is_available is False

    def test_partial_registry_loading(self):
        """일부 Registry만 성공적으로 로드된 경우"""
        # Preprocessor는 성공, Evaluator는 실패
        mock_preprocessor = MagicMock()
        mock_preprocessor.preprocessor_steps = {'scaler': MagicMock()}

        with patch.object(RegistryConnector, '_get_preprocessor_registry', return_value=mock_preprocessor), \
             patch.object(RegistryConnector, '_get_evaluator_registry', return_value=None):

            # Preprocessor 작동
            steps = RegistryConnector.get_available_preprocessor_steps()
            assert steps == ['scaler']

            # Evaluator 빈 결과
            tasks = RegistryConnector.get_available_task_types()
            assert tasks == []

            # 카탈로그에는 각각 반영
            catalog = RegistryConnector.get_component_catalog()
            assert catalog['preprocessor_steps'] == ['scaler']
            assert catalog['task_types'] == []