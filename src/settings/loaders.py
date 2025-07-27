"""
Settings Loaders & Utils
Blueprint v17.0 - 설정 로딩 및 환경 변수 처리
"""

import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from collections.abc import Mapping

from .models import Settings
from src.utils.system.logger import logger

# 기본 경로 및 환경 변수 로더
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

_env_var_pattern = re.compile(r"\$\{([^}:\s]+)(?::([^}]*))?\}")


def _env_var_replacer(m: re.Match) -> str:
    env_var = m.group(1)
    default_value = m.group(2)
    return os.getenv(env_var, default_value or "")


def _load_yaml_with_env(file_path: Path) -> Dict[str, Any]:
    """YAML 파일 로드 + 환경변수 치환"""
    if not file_path.exists():
        return {}
    text = file_path.read_text(encoding="utf-8")
    substituted_text = re.sub(_env_var_pattern, _env_var_replacer, text)
    return yaml.safe_load(substituted_text) or {}


def _recursive_merge(dict1: Dict, dict2: Dict) -> Dict:
    """딕셔너리 재귀적 병합 (dict2가 dict1을 덮어씀)"""
    for k, v in dict2.items():
        if k in dict1 and isinstance(dict1[k], Mapping) and isinstance(v, Mapping):
            dict1[k] = _recursive_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def load_config_files() -> Dict[str, Any]:
    """환경별 config 파일 로딩 - base.yaml → {app_env}.yaml 순서로 병합"""
    config_dir = BASE_DIR / "config"
    
    # 기본 인프라 설정 로드
    base_config = _load_yaml_with_env(config_dir / "base.yaml")
    
    # 환경별 설정 로드
    app_env = os.getenv("APP_ENV", "local")
    env_config_file = config_dir / f"{app_env}.yaml"
    env_config = _load_yaml_with_env(env_config_file)
    
    # 순차적 병합
    merged_config = _recursive_merge(base_config, env_config)
    
    return merged_config


def load_recipe_file(recipe_file: str) -> Dict[str, Any]:
    """Recipe 파일 로딩 - 절대/상대/recipes 경로 순으로 탐색"""
    path = Path(recipe_file)
    if not path.suffix:
        path = path.with_suffix('.yaml')

    # 우선순위: 절대 경로 → 상대 경로 → recipes/ 경로
    if path.is_absolute():
        final_path = path
    elif path.exists():
        final_path = path
    else:
        final_path = BASE_DIR / "recipes" / path

    if not final_path.exists():
        raise FileNotFoundError(f"Recipe 파일을 찾을 수 없습니다: {final_path}")
    
    return _load_yaml_with_env(final_path)


def load_settings(model_name: str) -> Settings:
    """
    모델명 기반 설정 로딩 (기존 호환성)
    
    Args:
        model_name: 모델명 (recipes/{model_name}.yaml)
        
    Returns:
        완전히 병합된 Settings 객체
    """
    return load_settings_by_file(f"models/{model_name}")


def load_settings_by_file(recipe_file: str, context_params: Optional[Dict[str, Any]] = None) -> Settings:
    """
    Blueprint v17.0 통합 설정 로딩 + Jinja 템플릿 렌더링
    
    현대화된 Recipe 구조 전용 (레거시 지원 제거)
    [YAML 로드 → Jinja 렌더링 → Pydantic 검증]의 3단계 파이프라인
    """
    from .models import RecipeSettings, Settings
    
    # 1. 환경별 config 로딩
    config_data = load_config_files()
    
    # 2. Recipe 파일 로딩
    recipe_data = load_recipe_file(recipe_file)
    
    if not recipe_data:
        raise ValueError(f"Recipe 파일이 비어있습니다: {recipe_file}")
    
    # 3. 현대화된 Recipe 구조 검증
    if not _is_modern_recipe_structure(recipe_data):
        raise ValueError(f"현대화된 Recipe 구조가 필요합니다: {recipe_file}. name, model, evaluation 필드가 있어야 합니다.")
    
    # 4. Jinja 템플릿 렌더링
    if context_params:
        recipe_data = _render_recipe_templates(recipe_data, context_params)
    
    # 5. RecipeSettings 생성 및 검증
    try:
        recipe_settings = RecipeSettings(**recipe_data)
        recipe_settings.validate_recipe_consistency()
    except Exception as e:
        raise ValueError(f"Recipe 검증 실패: {e}\n데이터: {recipe_data}")
    
    # 6. Settings 객체 생성
    final_data = {**config_data, "recipe": recipe_settings.model_dump()}
    
    try:
        settings = Settings(**final_data)
        
        # 7. computed 필드 생성
        settings.recipe.model.computed = _create_computed_fields(settings.recipe, recipe_file)
        
        # 🎯 MLflow 상대 경로를 절대 경로로 변환 (LOCAL 환경 안정성)
        if settings.environment.app_env == 'local' and settings.mlflow.tracking_uri.startswith("./"):
            from pathlib import Path
            uri_path = settings.mlflow.tracking_uri.replace("file://", "")
            absolute_path = Path(uri_path).resolve()
            settings.mlflow.tracking_uri = f"file://{absolute_path}"
            logger.info(f"MLflow relative tracking_uri resolved to absolute path: {settings.mlflow.tracking_uri}")

        return settings
        
    except Exception as e:
        raise ValueError(f"Settings 객체 생성 실패: {e}")


def _is_modern_recipe_structure(recipe_data: Dict[str, Any]) -> bool:
    """현대화된 Recipe 구조인지 검증"""
    required_fields = {"name", "model", "evaluation"}
    return required_fields.issubset(set(recipe_data.keys()))


def _render_recipe_templates(recipe_data: Dict[str, Any], context_params: Dict[str, Any]) -> Dict[str, Any]:
    """Recipe 구조의 Jinja 템플릿 렌더링"""
    try:
        from src.utils.system.templating_utils import render_sql_template
        
        # model.loader.source_uri 템플릿 렌더링
        model_config = recipe_data.get("model", {})
        loader_config = model_config.get("loader", {})
        loader_uri = loader_config.get("source_uri")
        
        if loader_uri and loader_uri.endswith(".sql.j2"):
            rendered_sql = render_sql_template(loader_uri, context_params)
            recipe_data["model"]["loader"]["source_uri"] = rendered_sql
            logger.info(f"Loader SQL template '{loader_uri}' rendered.")
        
        return recipe_data
        
    except Exception as e:
        raise ValueError(f"Jinja 템플릿 렌더링 실패: {e}") from e


def _create_computed_fields(recipe_settings: 'RecipeSettings', recipe_file: str) -> Dict[str, Any]:
    """Recipe 런타임 필드 생성"""
    from datetime import datetime
    
    class_name = recipe_settings.model.class_path.split('.')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{class_name}_{recipe_settings.name}_{timestamp}"
    
    # HPO 정보 추출
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
    
    # HPO 세부 정보 (활성화된 경우만)
    if hpo_enabled:
        computed.update({
            "hpo_trials": hpo.n_trials,
            "hpo_metric": hpo.metric,
            "hpo_direction": hpo.direction
        })
    
    return computed


# 편의 함수들
def get_app_env() -> str:
    """현재 앱 환경 반환"""
    return os.getenv("APP_ENV", "local")


def is_local_env() -> bool:
    """로컬 환경 여부 확인"""
    return get_app_env() == "local"


def is_dev_env() -> bool:
    """개발 환경 여부 확인"""
    return get_app_env() == "dev"


def is_prod_env() -> bool:
    """운영 환경 여부 확인"""
    return get_app_env() == "prod"


def get_feast_config(settings: Settings) -> Dict[str, Any]:
    """
    Blueprint v17.0: config에서 Feast 설정 추출
    
    Args:
        settings: Settings 객체
        
    Returns:
        Feast 초기화에 사용할 수 있는 설정 딕셔너리
    """
    if not settings.feature_store or not settings.feature_store.feast_config:
        raise ValueError("Feature Store 설정이 없습니다.")
    
    return settings.feature_store.feast_config 