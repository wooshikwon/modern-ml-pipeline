# src/settings/_helpers.py

from pathlib import Path
from typing import Dict, Any, List
from ._utils import _load_yaml_with_env, BASE_DIR
from src.utils.system.logger import logger
from ._recipe_schema import RecipeSettings, JinjaVariable
from src.settings import Settings
import pandas as pd
from src.utils.system.sql_utils import prevent_select_star


def load_config_files() -> Dict[str, Any]:
    """환경별 config 파일 로딩 - base.yaml → {app_env}.yaml 순서로 병합"""
    from ._utils import get_app_env, _recursive_merge
    config_dir = BASE_DIR / "config"
    
    base_config = _load_yaml_with_env(config_dir / "base.yaml")
    
    app_env = get_app_env()
    env_config_file = config_dir / f"{app_env}.yaml"
    env_config = _load_yaml_with_env(env_config_file)
    
    merged_config = _recursive_merge(base_config, env_config)
    
    return merged_config

def load_recipe_file(recipe_file: str) -> Dict[str, Any]:
    """Recipe 파일 로딩 - 절대/상대/recipes 경로 순으로 탐색"""
    path = Path(recipe_file)
    if not path.suffix:
        path = path.with_suffix('.yaml')

    if path.is_absolute():
        final_path = path
    elif path.exists():
        final_path = path
    else:
        final_path = BASE_DIR / "recipes" / path

    if not final_path.exists():
        raise FileNotFoundError(f"Recipe 파일을 찾을 수 없습니다: {final_path}")
    
    return _load_yaml_with_env(final_path)

def _render_recipe_templates(recipe_data: Dict[str, Any], context_params: Dict[str, Any]) -> Dict[str, Any]:
    """Recipe 구조의 Jinja 템플릿 렌더링 및 SQL 안전성 검증"""
    try:
        from src.utils.system.templating_utils import render_template_from_file
        
        model_config = recipe_data.get("model", {})
        loader_config = model_config.get("loader", {})
        loader_uri = loader_config.get("source_uri")
        
        if loader_uri and loader_uri.endswith(".sql.j2"):
            rendered_sql = render_template_from_file(loader_uri, context_params)
            
            # [신규] 렌더링된 SQL에 대해 SELECT * 검증 수행
            prevent_select_star(rendered_sql)

            recipe_data["model"]["loader"]["source_uri"] = rendered_sql
            logger.info(f"Loader SQL template '{loader_uri}' rendered and validated.")
        
        return recipe_data
        
    except Exception as e:
        raise ValueError(f"Jinja 템플릿 렌더링 또는 검증 실패: {e}") from e

def _create_computed_fields(recipe_settings: RecipeSettings, recipe_file: str) -> Dict[str, Any]:
    """Recipe 런타임 필드 생성"""
    from datetime import datetime
    
    class_name = recipe_settings.model.class_path.split('.')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{class_name}_{recipe_settings.name}_{timestamp}"
    
    hpo = recipe_settings.model.hyperparameter_tuning
    hpo_enabled = bool(hpo and hpo.enabled)
    
    computed = {
        "run_name": run_name,
        "timestamp": timestamp,
        "model_class_name": class_name,
        "recipe_file": recipe_file,
        "recipe_name": recipe_settings.name,
        "task_type": recipe_settings.model.data_interface.task_type,
        "hpo_enabled": hpo_enabled
    }
    
    if hpo_enabled:
        computed.update({
            "hpo_trials": hpo.n_trials,
            "hpo_metric": hpo.metric,
            "hpo_direction": hpo.direction
        })
    
    return computed

def _validate_and_prepare_context_params(
    jinja_vars: List[JinjaVariable], context_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    사용자가 전달한 context_params를 레시피에 정의된 jinja_variables 명세서와 비교하여
    유효성을 검증하고, 기본값을 채워넣습니다.
    """
    prepared_params = {}
    declared_vars = {v.name: v for v in jinja_vars}
    
    # 명세서에 정의된 변수들을 순회하며 검증
    for name, spec in declared_vars.items():
        if name in context_params:
            value = context_params[name]
            # 타입 검증 (간단한 예시)
            if spec.type == "date":
                try:
                    pd.to_datetime(value)
                except ValueError:
                    raise TypeError(f"Jinja 변수 '{name}'은(는) 날짜 형식이어야 합니다 (입력값: {value}).")
            elif spec.type == "integer":
                if not isinstance(value, int):
                    raise TypeError(f"Jinja 변수 '{name}'은(는) 정수여야 합니다 (입력값: {value}).")
            # ... 다른 타입 검증 추가 가능 ...
            prepared_params[name] = value
        elif spec.required:
            raise ValueError(f"필수 Jinja 변수 '{name}'이(가) 누락되었습니다.")
        else:
            prepared_params[name] = spec.default

    # 명세서에 정의되지 않은 변수가 있는지 확인
    for name in context_params:
        if name not in declared_vars:
            logger.warning(f"'{name}'은(는) 레시피에 정의되지 않은 Jinja 변수입니다. 무시됩니다.")
            
    return prepared_params


def _post_process_settings(settings: Settings) -> Settings:
    """생성된 Settings 객체에 대한 후처리 작업을 수행합니다."""
    # MLflow 경로 처리 (LOCAL 환경 안정성)
    if settings.environment.app_env == 'local' and settings.mlflow.tracking_uri.startswith("./"):
        uri_path = settings.mlflow.tracking_uri.replace("file://", "")
        absolute_path = Path(uri_path).resolve()
        settings.mlflow.tracking_uri = f"file://{absolute_path}"
        logger.info(f"MLflow relative tracking_uri resolved to absolute path: {settings.mlflow.tracking_uri}")
    return settings

def _is_modern_recipe_structure(recipe_data: Dict[str, Any]) -> bool:
    """현대화된 Recipe 구조인지 검증"""
    required_fields = {"name", "model", "evaluation"}
    return required_fields.issubset(set(recipe_data.keys())) 