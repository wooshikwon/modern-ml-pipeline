"""src/components Registry 기반 비즈니스 로직 검증"""

from typing import Dict, List, Set
from ..recipe import Preprocessor
from .common import ValidationResult


class BusinessValidator:
    """Components Registry 기반 동적 검증 시스템"""

    def __init__(self):
        self._trigger_component_imports()  # Registry 활성화

    def _trigger_component_imports(self):
        """모든 컴포넌트 Registry 활성화"""
        try:
            import src.components.preprocessor
            import src.components.evaluator
            # import src.components.calibration  # 나중에 추가
            # ... 기타 컴포넌트들
        except ImportError as e:
            # Registry가 아직 없을 수 있음 - 개발 중
            pass

    def get_available_preprocessor_types(self) -> Set[str]:
        """실제 등록된 전처리기 타입들 동적 추출"""
        try:
            from src.components.preprocessor.registry import PreprocessorStepRegistry
            return set(PreprocessorStepRegistry.preprocessor_steps.keys())
        except ImportError:
            # Registry 로드 실패 시 빈 세트 반환
            return set()

    def get_available_calibrators(self) -> Set[str]:
        """실제 등록된 캘리브레이터들 동적 추출"""
        try:
            from src.components.calibration.registry import CalibrationRegistry
            return set(CalibrationRegistry.calibrators.keys())
        except ImportError:
            # Registry 로드 실패 시 빈 세트 반환
            return set()

    def get_available_evaluators_for_task(self, task_type: str) -> List[str]:
        """Task별 사용 가능한 Evaluator들 동적 추출"""
        try:
            from src.components.evaluator.registry import EvaluatorRegistry
            return EvaluatorRegistry.get_available_metrics_for_task(task_type)
        except ImportError:
            # Registry 로드 실패 시 빈 리스트 반환
            return []

    def validate_preprocessor_steps(self, recipe_preprocessor: Preprocessor) -> ValidationResult:
        """Recipe의 전처리 단계를 Registry와 대조 검증"""
        if not recipe_preprocessor or not recipe_preprocessor.steps:
            return ValidationResult(is_valid=True)

        available_types = self.get_available_preprocessor_types()

        for step in recipe_preprocessor.steps:
            step_type = step.type
            if step_type not in available_types:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"알 수 없는 전처리기 타입: '{step_type}'. 사용 가능한 타입: {sorted(available_types)}"
                )

        return ValidationResult(is_valid=True)

    def validate_calibration_settings(self, recipe_calibration: Dict, task_type: str) -> ValidationResult:
        """캘리브레이션 설정 검증"""
        if not recipe_calibration or not recipe_calibration.get('enabled', False):
            return ValidationResult(is_valid=True)

        # Classification 태스크만 캘리브레이션 지원
        if task_type.lower() != 'classification':
            return ValidationResult(
                is_valid=False,
                error_message=f"캘리브레이션은 classification 태스크에서만 지원됩니다. 현재: {task_type}"
            )

        method = recipe_calibration.get('method')
        if not method:
            return ValidationResult(
                is_valid=False,
                error_message="캘리브레이션이 활성화되어 있지만 method가 지정되지 않았습니다."
            )

        available_calibrators = self.get_available_calibrators()
        if method not in available_calibrators:
            return ValidationResult(
                is_valid=False,
                error_message=f"알 수 없는 캘리브레이션 방법: '{method}'. 사용 가능한 방법: {sorted(available_calibrators)}"
            )

        return ValidationResult(is_valid=True)

    def validate_evaluation_metrics(self, recipe_metrics: List[str], task_type: str) -> ValidationResult:
        """평가 메트릭 검증"""
        if not recipe_metrics:
            return ValidationResult(is_valid=True)

        available_metrics = self.get_available_evaluators_for_task(task_type)
        invalid_metrics = [m for m in recipe_metrics if m not in available_metrics]

        if invalid_metrics:
            return ValidationResult(
                is_valid=False,
                error_message=f"{task_type}에서 사용할 수 없는 메트릭: {invalid_metrics}. 사용 가능한 메트릭: {available_metrics}"
            )

        return ValidationResult(is_valid=True)
