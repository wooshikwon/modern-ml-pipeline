"""검증 시스템 통합 오케스트레이터"""

from ..config import Config
from ..recipe import Recipe
from .common import ValidationResult
from .catalog_validator import CatalogValidator
from .business_validator import BusinessValidator
from .compatibility_validator import CompatibilityValidator


class ValidationOrchestrator:
    """모든 검증 로직의 중앙 조정자"""

    def __init__(self):
        self.catalog_validator = CatalogValidator()
        self.business_validator = BusinessValidator()
        self.compatibility_validator = CompatibilityValidator()

    def validate_for_training(self, config: Config, recipe: Recipe) -> ValidationResult:
        """학습용 종합 검증"""
        # 1. Catalog 기반 모델/하이퍼파라미터 검증
        catalog_result = self.catalog_validator.validate_model_specification(recipe.model)
        if not catalog_result.is_valid:
            return catalog_result

        # 2. Registry 기반 컴포넌트 검증
        if recipe.preprocessor:
            preprocessor_result = self.business_validator.validate_preprocessor_steps(recipe.preprocessor)
            if not preprocessor_result.is_valid:
                return preprocessor_result

        # 3. Config-Recipe 호환성 검증
        compatibility_result = self.compatibility_validator.validate_feature_store_consistency(config, recipe)
        if not compatibility_result.is_valid:
            return compatibility_result

        # 4. 학습 전용 검증
        # MLflow 설정 검증 등...

        return ValidationResult(is_valid=True)

    def validate_for_serving(self, config: Config, recipe: Recipe) -> ValidationResult:
        """서빙용 종합 검증"""
        # 서빙 환경 특화 검증
        if not config.serving or not config.serving.enabled:
            return ValidationResult(
                is_valid=False,
                error_message="서빙용 설정에서는 serving.enabled가 true여야 합니다."
            )

        # 기본 검증도 실행
        return self.validate_for_training(config, recipe)

    def validate_for_inference(self, config: Config, recipe: Recipe) -> ValidationResult:
        """추론용 종합 검증"""
        # 추론 환경 특화 검증
        if not config.output:
            return ValidationResult(
                is_valid=False,
                error_message="추론용 설정에서는 output 설정이 필요합니다."
            )

        # 기본 검증도 실행
        return self.validate_for_training(config, recipe)
