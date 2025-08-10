"""
Settings Loaders - Public API
이 모듈은 설정 로딩을 위한 최상위 공개 API를 제공합니다.
설정 로딩의 전체 과정을 조율하는 오케스트레이터 역할을 합니다.
"""

from typing import Dict, Any, Optional

from .schema import Settings
from ._recipe_schema import RecipeSettings, JinjaVariable
from src.utils.system.logger import logger
from ._builder import (
    load_config_files,
    load_recipe_file,
    _is_modern_recipe_structure,
    _render_recipe_templates,
    _create_computed_fields,
    _post_process_settings,
    _validate_and_prepare_context_params,
)
from src.utils.system.sql_utils import prevent_select_star
from pathlib import Path
from ._utils import BASE_DIR

__all__ = ["load_settings", "load_settings_by_file", "create_settings_for_inference", "load_config_files"]

def load_settings(model_name: str) -> Settings:
    """
    모델명 기반 설정 로딩 (기존 호환성)
    """
    return load_settings_by_file(f"models/{model_name}")

def load_settings_by_file(recipe_file: str, context_params: Optional[Dict[str, Any]] = None) -> Settings:
    """
    [YAML 로드 → Jinja 변수 검증 → Jinja 렌더링 → Pydantic 검증]의 파이프라인을 조율합니다.
    """
    # 1. 환경별 config와 Recipe 파일 로딩
    config_data = load_config_files()
    recipe_data = load_recipe_file(recipe_file)
    
    if not recipe_data:
        raise ValueError(f"Recipe 파일이 비어있습니다: {recipe_file}")
    
    # 2. Recipe 구조 검증
    if not _is_modern_recipe_structure(recipe_data):
        raise ValueError(f"현대화된 Recipe 구조가 필요합니다: {recipe_file}.")
    
    # 3. (조건부) Jinja 변수 검증 및 렌더링
    if context_params:
        jinja_vars_spec = recipe_data.get("model", {}).get("loader", {}).get("jinja_variables")
        if jinja_vars_spec:
            # Pydantic 모델로 임시 변환하여 명세서 객체 생성
            temp_jinja_vars = [JinjaVariable(**spec) for spec in jinja_vars_spec]
            validated_params = _validate_and_prepare_context_params(temp_jinja_vars, context_params)
            recipe_data = _render_recipe_templates(recipe_data, validated_params)
        else:
            logger.warning("`context_params`가 제공되었지만, 레시피에 `jinja_variables` 명세가 없습니다.")
            recipe_data = _render_recipe_templates(recipe_data, context_params)

    # 4. 정적 SQL에 대한 SELECT * 검증 및 경로 해석 강화
    loader_config = recipe_data.get("model", {}).get("loader", {})
    source_uri = loader_config.get("source_uri")
    if source_uri and source_uri.endswith(".sql"):
        sql_path = Path(source_uri)
        if not sql_path.is_absolute():
            sql_path = BASE_DIR / sql_path
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL 파일을 찾을 수 없습니다: {sql_path}")
        prevent_select_star(sql_path.read_text(encoding="utf-8"))

    # 5. Pydantic 모델로 변환 및 검증
    try:
        recipe_settings = RecipeSettings(**recipe_data)
    except Exception as e:
        raise ValueError(f"Recipe 검증 실패: {e}\n데이터: {recipe_data}")
    
    # 6. 최종 Settings 객체 생성
    final_data = {**config_data, "recipe": recipe_settings.model_dump()}
    
    try:
        settings = Settings(**final_data)
        
        # 7. 동적 필드 생성
        settings.recipe.model.computed = _create_computed_fields(settings.recipe, recipe_file)
        
        # 8. 후처리 위임
        settings = _post_process_settings(settings)

        return settings
        
    except Exception as e:
        raise ValueError(f"Settings 객체 생성 실패: {e}")


def create_settings_for_inference(config_data: Dict[str, Any]) -> Settings:
    """
    추론 파이프라인(배치/서빙)을 위한 Settings 객체를 생성합니다.
    """
    if "recipe" not in config_data:
        config_data["recipe"] = {
            "name": "dummy_inference_recipe",
            "model": {
                "class_path": "dummy.path",
                "loader": {
                    "name": "dummy_loader",
                    "source_uri": "dummy_uri",
                    "entity_schema": {
                        "entity_columns": ["dummy_id"],
                        "timestamp_column": "dummy_timestamp"
                    }
                },
                "data_interface": {"task_type": "dummy"},
                "hyperparameters": {}
            },
            "evaluation": {
                "metrics": ["accuracy"],
                "validation": {"method": "train_test_split"}
            }
        }
    return Settings(**config_data) 