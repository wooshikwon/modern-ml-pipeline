"""src/components Registry 기반 비즈니스 로직 검증"""

from typing import Dict, List, Set

from ..recipe import Preprocessor, Recipe
from .common import ValidationResult

# 전처리기 순서 카테고리 정의
# 권장 순서: missing -> encoder -> feature_gen -> scaler
STEP_CATEGORIES = {
    "missing": [
        "simple_imputer",
        "drop_missing",
        "forward_fill",
        "backward_fill",
        "constant_fill",
        "interpolation",
    ],
    "encoder": [
        "ordinal_encoder",
        "one_hot_encoder",
        "catboost_encoder",
    ],
    "feature_gen": [
        "polynomial_features",
        "tree_based_feature_generator",
        "kbins_discretizer",
    ],
    "scaler": [
        "standard_scaler",
        "min_max_scaler",
        "robust_scaler",
    ],
}

# 카테고리별 권장 순서 (낮을수록 먼저)
CATEGORY_ORDER = {
    "missing": 0,
    "encoder": 1,
    "feature_gen": 2,
    "scaler": 3,
}


def _get_step_category(step_type: str) -> str:
    """전처리기 타입의 카테고리 반환"""
    for category, types in STEP_CATEGORIES.items():
        if step_type in types:
            return category
    return "unknown"


class BusinessValidator:
    """Components Registry 기반 동적 검증 시스템"""

    def __init__(self):
        self._trigger_component_imports()  # Registry 활성화

    def _trigger_component_imports(self):
        """모든 컴포넌트 Registry 활성화"""
        try:
            import src.components.evaluator
            import src.components.preprocessor

            # import src.components.calibration  # 나중에 추가
            # ... 기타 컴포넌트들
        except ImportError:
            # Registry가 아직 없을 수 있음 - 개발 중
            pass

    def get_available_preprocessor_types(self) -> Set[str]:
        """실제 등록된 전처리기 타입들 동적 추출"""
        try:
            from src.components.preprocessor.registry import PreprocessorStepRegistry

            return set(PreprocessorStepRegistry.list_keys())
        except ImportError:
            # Registry 로드 실패 시 빈 세트 반환
            return set()

    def get_available_calibrators(self) -> Set[str]:
        """실제 등록된 캘리브레이터들 동적 추출"""
        try:
            from src.components.calibration.registry import CalibrationRegistry

            return set(CalibrationRegistry.list_keys())
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

    def get_default_optimization_metric(self, task_type: str) -> str:
        """Task별 기본 최적화 메트릭 제공"""
        try:
            from src.components.evaluator.registry import EvaluatorRegistry

            return EvaluatorRegistry.get_default_optimization_metric(task_type)
        except ImportError:
            return "accuracy"

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
                    error_message=f"알 수 없는 전처리기 타입: '{step_type}'. 사용 가능한 타입: {sorted(available_types)}",
                )

        return ValidationResult(is_valid=True)

    def validate_preprocessor_ordering(
        self, recipe_preprocessor: Preprocessor
    ) -> ValidationResult:
        """
        전처리기 순서가 권장 순서를 따르는지 검증.

        권장 순서: 결측값 처리 -> 인코딩 -> 피처 생성 -> 스케일링
        잘못된 순서는 에러가 아닌 경고로 처리하여 기존 레시피 호환성 유지.
        """
        if not recipe_preprocessor or not recipe_preprocessor.steps:
            return ValidationResult(is_valid=True)

        warnings = []
        step_positions = []

        # 각 step의 카테고리와 위치 추적
        for idx, step in enumerate(recipe_preprocessor.steps):
            category = _get_step_category(step.type)
            if category != "unknown":
                step_positions.append((idx, step.type, category))

        # 순서 위반 검사
        for i, (idx1, type1, cat1) in enumerate(step_positions):
            for idx2, type2, cat2 in step_positions[i + 1 :]:
                order1 = CATEGORY_ORDER.get(cat1, 99)
                order2 = CATEGORY_ORDER.get(cat2, 99)

                if order1 > order2:
                    # 순서 위반 발견
                    cat1_kr = {"missing": "결측값 처리", "encoder": "인코딩",
                               "feature_gen": "피처 생성", "scaler": "스케일링"}.get(cat1, cat1)
                    cat2_kr = {"missing": "결측값 처리", "encoder": "인코딩",
                               "feature_gen": "피처 생성", "scaler": "스케일링"}.get(cat2, cat2)

                    warning_msg = (
                        f"전처리 순서 주의: '{type1}'({cat1_kr})이 '{type2}'({cat2_kr})보다 뒤에 있습니다. "
                        f"권장 순서: 결측값 처리 → 인코딩 → 피처 생성 → 스케일링"
                    )
                    if warning_msg not in warnings:
                        warnings.append(warning_msg)

        return ValidationResult(is_valid=True, warnings=warnings)

    def validate_calibration_settings(
        self, recipe_calibration: Dict, task_type: str
    ) -> ValidationResult:
        """캘리브레이션 설정 검증"""
        if not recipe_calibration or not recipe_calibration.get("enabled", False):
            return ValidationResult(is_valid=True)

        # Classification 태스크만 캘리브레이션 지원
        if task_type.lower() != "classification":
            return ValidationResult(
                is_valid=False,
                error_message=f"캘리브레이션은 classification 태스크에서만 지원됩니다. 현재: {task_type}",
            )

        method = recipe_calibration.get("method")
        if not method:
            return ValidationResult(
                is_valid=False,
                error_message="캘리브레이션이 활성화되어 있지만 method가 지정되지 않았습니다.",
            )

        available_calibrators = self.get_available_calibrators()
        if method not in available_calibrators:
            return ValidationResult(
                is_valid=False,
                error_message=f"알 수 없는 캘리브레이션 방법: '{method}'. 사용 가능한 방법: {sorted(available_calibrators)}",
            )

        return ValidationResult(is_valid=True)

    def validate_evaluation_metrics(
        self, recipe_metrics: List[str], task_type: str
    ) -> ValidationResult:
        """평가 메트릭 검증"""
        if not recipe_metrics:
            return ValidationResult(is_valid=True)

        available_metrics = self.get_available_evaluators_for_task(task_type)
        invalid_metrics = [m for m in recipe_metrics if m not in available_metrics]

        if invalid_metrics:
            return ValidationResult(
                is_valid=False,
                error_message=f"{task_type}에서 사용할 수 없는 메트릭: {invalid_metrics}. 사용 가능한 메트릭: {available_metrics}",
            )

        return ValidationResult(is_valid=True)

    def validate_optuna_requires_validation_split(self, recipe: Recipe) -> ValidationResult:
        """Optuna 튜닝 사용 시 validation split 필수 검증"""
        tuning_enabled = recipe.model.hyperparameters.tuning_enabled
        validation_split = recipe.data.split.validation

        if tuning_enabled and (validation_split is None or validation_split <= 0):
            return ValidationResult(
                is_valid=False,
                error_message=(
                    "Optuna 튜닝(tuning_enabled=true) 사용 시 validation split이 0보다 커야 합니다. "
                    f"현재 validation split: {validation_split}"
                ),
            )

        return ValidationResult(is_valid=True)
