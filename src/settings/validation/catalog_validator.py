"""models/catalog 기반 동적 모델/하이퍼파라미터 검증"""

from pathlib import Path
from typing import Dict, Optional, Set

import yaml

from ..recipe import Model
from .common import ValidationResult

# 패키지 내부 catalog 경로 (설치 환경에서도 작동)
_DEFAULT_CATALOG_PATH = Path(__file__).parent.parent.parent / "models" / "catalog"


class CatalogValidator:
    """Models Catalog 기반 동적 검증 시스템"""

    def __init__(self, catalog_path: Optional[Path] = None):
        self.catalog_path = catalog_path or _DEFAULT_CATALOG_PATH
        self._task_models_cache = {}  # 성능 최적화용 캐시

    def get_available_tasks(self) -> Set[str]:
        """사용 가능한 Task 목록 동적 추출"""
        # src/models/catalog/ 하위 디렉토리명 기반
        tasks = set()
        for task_dir in self.catalog_path.iterdir():
            if task_dir.is_dir() and not task_dir.name.startswith("."):
                tasks.add(task_dir.name.lower())
        return tasks

    def get_available_models_for_task(self, task_type: str) -> Dict[str, Dict]:
        """특정 Task의 사용 가능한 모델들 동적 추출"""
        task_dir = self.catalog_path / task_type.title()
        if not task_dir.exists():
            return {}

        models = {}
        for model_file in task_dir.glob("*.yaml"):
            with open(model_file, "r") as f:
                model_spec = yaml.safe_load(f)
                models[model_file.stem] = model_spec

        return models

    def validate_model_specification(self, recipe_model: Model) -> ValidationResult:
        """Recipe의 모델 스펙을 Catalog와 대조 검증"""
        class_path = recipe_model.class_path
        # task_choice는 Recipe 레벨에 있으므로 여기서는 기본 검증만

        # 1. 기본적인 model 구조 검증
        if not class_path:
            return ValidationResult(
                is_valid=False, error_message="Model class_path가 지정되지 않았습니다."
            )

        # 2. 하이퍼파라미터 기본 구조 검증
        hyperparams = recipe_model.hyperparameters
        if hyperparams.tuning_enabled:
            if not hyperparams.tunable:
                return ValidationResult(
                    is_valid=False,
                    error_message="튜닝이 활성화되어 있지만 tunable 파라미터가 없습니다.",
                )
        else:
            if not hyperparams.values:
                return ValidationResult(
                    is_valid=False,
                    error_message="튜닝이 비활성화되어 있지만 values 파라미터가 없습니다.",
                )

        return ValidationResult(is_valid=True)

    def validate_task_model_compatibility(self, task_type: str, model: Model) -> ValidationResult:
        """Task 타입과 모델 호환성 검증 (Recipe에서 호출)"""
        # 1. Task 타입 검증
        available_tasks = self.get_available_tasks()
        if task_type.lower() not in available_tasks:
            return ValidationResult(
                is_valid=False,
                error_message=f"알 수 없는 태스크 타입: {task_type}. 사용 가능한 타입: {sorted(available_tasks)}",
            )

        # 2. 모델 존재 검증
        available_models = self.get_available_models_for_task(task_type)
        model_found = None
        for model_name, model_spec in available_models.items():
            if model_spec.get("class_path") == model.class_path:
                model_found = model_spec
                break

        if not model_found:
            return ValidationResult(
                is_valid=False,
                error_message=f"모델 {model.class_path}이 {task_type} catalog에서 찾을 수 없습니다.",
            )

        # 3. 하이퍼파라미터 검증
        return self._validate_hyperparameters(model, model_found)

    def _validate_hyperparameters(
        self, recipe_model: Model, catalog_spec: Dict
    ) -> ValidationResult:
        """하이퍼파라미터 범위 및 타입 검증"""
        recipe_hyperparams = recipe_model.hyperparameters
        catalog_hyperparams = catalog_spec.get("hyperparameters", {})

        if recipe_hyperparams.tuning_enabled:
            # 튜닝 모드: tunable 파라미터 검증
            recipe_tunable = recipe_hyperparams.tunable or {}
            catalog_tunable = catalog_hyperparams.get("tunable", {})

            for param_name, param_spec in recipe_tunable.items():
                if param_name not in catalog_tunable:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"튜닝 파라미터 '{param_name}'이 {catalog_spec['class_path']}에서 지원되지 않습니다.",
                    )

                # 범위 검증
                recipe_range = param_spec.get("range", [])
                catalog_range = catalog_tunable[param_name].get("range", [])

                if (
                    recipe_range
                    and catalog_range
                    and len(recipe_range) >= 2
                    and len(catalog_range) >= 2
                    and (recipe_range[0] < catalog_range[0] or recipe_range[1] > catalog_range[1])
                ):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"파라미터 '{param_name}' 범위 {recipe_range}가 catalog 제한 {catalog_range}을 초과합니다.",
                    )

        return ValidationResult(is_valid=True)
