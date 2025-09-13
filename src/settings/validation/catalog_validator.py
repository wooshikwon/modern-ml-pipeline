"""
Catalog Validator - Component Catalog 검증

컴포넌트가 실제로 등록되어 있고 사용 가능한지 검증합니다.
Registry 시스템과 연동하여 실시간으로 사용 가능한 컴포넌트를 확인합니다.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
from .registry_connector import RegistryConnector
from src.utils.core.logger import logger

class ValidationError(Exception):
    """Validation 에러"""
    pass

class CatalogValidator:
    """
    Component Catalog 기반 검증기

    Recipe와 Config에서 참조하는 모든 컴포넌트가
    실제로 시스템에 등록되어 있고 사용 가능한지 검증합니다.
    """

    def __init__(self):
        self.registry = RegistryConnector()
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def clear_messages(self):
        """에러와 경고 메시지 초기화"""
        self.errors.clear()
        self.warnings.clear()

    def add_error(self, message: str):
        """에러 메시지 추가"""
        self.errors.append(message)
        logger.error(f"[catalog_validator] {message}")

    def add_warning(self, message: str):
        """경고 메시지 추가"""
        self.warnings.append(message)
        logger.warning(f"[catalog_validator] {message}")

    def validate_preprocessor_steps(self, preprocessor_config: Optional[Dict[str, Any]]) -> bool:
        """전처리 스텝들이 등록되어 있는지 검증"""
        if not preprocessor_config or not preprocessor_config.get('steps'):
            return True  # 전처리 스텝이 없는 것은 유효함

        steps = preprocessor_config['steps']
        if not isinstance(steps, list):
            self.add_error("Preprocessor steps must be a list")
            return False

        available_steps = self.registry.get_available_preprocessor_steps()
        if not available_steps:
            self.add_warning("No preprocessor steps are registered in the system")

        valid = True
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                self.add_error(f"Preprocessor step {i} must be a dictionary")
                valid = False
                continue

            step_type = step.get('type')
            if not step_type:
                self.add_error(f"Preprocessor step {i} is missing 'type' field")
                valid = False
                continue

            if not self.registry.is_preprocessor_step_available(step_type):
                self.add_error(
                    f"Preprocessor step type '{step_type}' is not registered. "
                    f"Available types: {available_steps}"
                )
                valid = False

        return valid

    def validate_task_type(self, task_choice: str) -> bool:
        """태스크 타입이 지원되는지 검증"""
        available_tasks = self.registry.get_available_task_types()

        if not available_tasks:
            self.add_warning("No task types are registered in the evaluator system")
            return True  # Registry가 아직 로드되지 않았을 수 있음

        if not self.registry.is_task_type_available(task_choice):
            self.add_error(
                f"Task type '{task_choice}' is not supported. "
                f"Available task types: {available_tasks}"
            )
            return False

        return True

    def validate_fetcher_type(self, fetcher_config: Dict[str, Any]) -> bool:
        """피처 페처 타입이 지원되는지 검증"""
        fetcher_type = fetcher_config.get('type')
        if not fetcher_type:
            self.add_error("Fetcher configuration is missing 'type' field")
            return False

        available_fetchers = self.registry.get_available_fetchers()

        if not available_fetchers:
            self.add_warning("No fetcher types are registered in the system")
            return True  # Registry가 아직 로드되지 않았을 수 있음

        if not self.registry.is_fetcher_available(fetcher_type):
            self.add_error(
                f"Fetcher type '{fetcher_type}' is not registered. "
                f"Available fetcher types: {available_fetchers}"
            )
            return False

        return True

    def validate_adapter_type(self, adapter_type: str) -> bool:
        """어댑터 타입이 지원되는지 검증"""
        available_adapters = self.registry.get_available_adapters()

        if not available_adapters:
            self.add_warning("No adapter types are registered in the system")
            return True  # Registry가 아직 로드되지 않았을 수 있음

        if not self.registry.is_adapter_available(adapter_type):
            self.add_error(
                f"Adapter type '{adapter_type}' is not registered. "
                f"Available adapter types: {available_adapters}"
            )
            return False

        return True

    def validate_calibration_method(self, calibration_config: Optional[Dict[str, Any]]) -> bool:
        """캘리브레이션 방법이 지원되는지 검증"""
        if not calibration_config or not calibration_config.get('enabled'):
            return True  # 캘리브레이션이 비활성화된 경우

        method = calibration_config.get('method')
        if not method:
            self.add_error("Calibration is enabled but 'method' is not specified")
            return False

        available_methods = self.registry.get_available_calibration_methods()

        if not available_methods:
            self.add_warning("No calibration methods are registered in the system")
            return True  # Registry가 아직 로드되지 않았을 수 있음

        if not self.registry.is_calibration_method_available(method):
            self.add_error(
                f"Calibration method '{method}' is not registered. "
                f"Available methods: {available_methods}"
            )
            return False

        return True

    def validate_recipe_components(self, recipe_data: Dict[str, Any]) -> bool:
        """Recipe의 모든 컴포넌트 카탈로그 검증"""
        self.clear_messages()
        valid = True

        # Task type 검증
        task_choice = recipe_data.get('task_choice')
        if task_choice:
            if not self.validate_task_type(task_choice):
                valid = False

        # Preprocessor 검증
        preprocessor = recipe_data.get('preprocessor')
        if preprocessor:
            if not self.validate_preprocessor_steps(preprocessor):
                valid = False

        # Fetcher 검증
        data_config = recipe_data.get('data', {})
        fetcher_config = data_config.get('fetcher')
        if fetcher_config:
            if not self.validate_fetcher_type(fetcher_config):
                valid = False

        # Calibration 검증
        model_config = recipe_data.get('model', {})
        calibration_config = model_config.get('calibration')
        if calibration_config:
            if not self.validate_calibration_method(calibration_config):
                valid = False

        return valid

    def validate_config_components(self, config_data: Dict[str, Any]) -> bool:
        """Config의 모든 컴포넌트 카탈로그 검증"""
        self.clear_messages()
        valid = True

        # Data source adapter 검증
        data_source = config_data.get('data_source', {})
        adapter_type = data_source.get('adapter_type')
        if adapter_type:
            if not self.validate_adapter_type(adapter_type):
                valid = False

        # Output adapters 검증
        output_config = config_data.get('output', {})
        if output_config:
            # Inference output adapter
            inference_config = output_config.get('inference', {})
            inference_adapter = inference_config.get('adapter_type')
            if inference_adapter:
                if not self.validate_adapter_type(inference_adapter):
                    valid = False

            # Preprocessed output adapter
            preprocessed_config = output_config.get('preprocessed', {})
            preprocessed_adapter = preprocessed_config.get('adapter_type')
            if preprocessed_adapter:
                if not self.validate_adapter_type(preprocessed_adapter):
                    valid = False

        return valid

    def get_validation_summary(self) -> Dict[str, Any]:
        """검증 결과 요약 반환"""
        return {
            'is_valid': len(self.errors) == 0,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
        }

    def get_available_components_summary(self) -> Dict[str, List[str]]:
        """사용 가능한 컴포넌트 목록 반환"""
        return self.registry.get_component_catalog()

    def validate_component_exists(self, component_type: str, component_name: str) -> bool:
        """특정 컴포넌트의 존재 여부 검증"""
        return self.registry.validate_component_availability(component_type, component_name)

    def suggest_alternatives(self, component_type: str, invalid_name: str) -> List[str]:
        """잘못된 컴포넌트 이름에 대한 대안 제안"""
        catalog = self.get_available_components_summary()

        type_mapping = {
            'preprocessor': 'preprocessor_steps',
            'task': 'task_types',
            'fetcher': 'fetchers',
            'adapter': 'adapters',
            'calibration': 'calibration_methods',
        }

        catalog_key = type_mapping.get(component_type)
        if not catalog_key or catalog_key not in catalog:
            return []

        available = catalog[catalog_key]

        # 간단한 유사도 기반 제안 (이름 포함 여부)
        suggestions = []
        invalid_lower = invalid_name.lower()

        for item in available:
            if invalid_lower in item.lower() or item.lower() in invalid_lower:
                suggestions.append(item)

        # 최대 3개까지만 제안
        return suggestions[:3]