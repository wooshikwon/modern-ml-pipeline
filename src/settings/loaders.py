"""
Settings Loaders & Utils
Blueprint v17.0 설정 로딩 및 환경 변수 처리 모듈

이 모듈은 설정 파일 로딩, 환경 변수 치환, 설정 병합 로직을 관리합니다.
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

# --- 기본 경로 및 환경 변수 로더 ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# 환경 변수 패턴: ${VAR_NAME:default_value}
_env_var_pattern = re.compile(r"\$\{([^}:\s]+)(?::([^}]*))?\}")


def _env_var_replacer(m: re.Match) -> str:
    """환경 변수 치환 함수"""
    env_var = m.group(1)
    default_value = m.group(2)
    return os.getenv(env_var, default_value or "")


def _load_yaml_with_env(file_path: Path) -> Dict[str, Any]:
    """환경 변수가 치환된 YAML 파일 로딩"""
    if not file_path.exists():
        return {}
    
    text = file_path.read_text(encoding="utf-8")
    substituted_text = re.sub(_env_var_pattern, _env_var_replacer, text)
    return yaml.safe_load(substituted_text) or {}


def _recursive_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    두 딕셔너리를 재귀적으로 병합합니다. 
    dict2의 값이 dict1의 값을 덮어씁니다.
    """
    for k, v in dict2.items():
        if k in dict1 and isinstance(dict1[k], Mapping) and isinstance(v, Mapping):
            dict1[k] = _recursive_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def load_config_files() -> Dict[str, Any]:
    """
    Blueprint v17.0 환경별 config 파일 로딩
    data_adapters.yaml -> base.yaml -> {app_env}.yaml 순서로 병합
    """
    config_dir = BASE_DIR / "config"
    
    # 1. 어댑터 설정 로드
    adapter_config = _load_yaml_with_env(config_dir / "data_adapters.yaml")
    
    # 2. 기본 인프라 설정 로드
    base_config = _load_yaml_with_env(config_dir / "base.yaml")
    
    # 3. 환경별 설정 로드
    app_env = os.getenv("APP_ENV", "local")
    env_config_file = config_dir / f"{app_env}.yaml"
    env_config = _load_yaml_with_env(env_config_file)
    
    # 4. 순차적 병합 (오른쪽이 왼쪽을 덮어씀)
    merged_config = _recursive_merge(adapter_config, base_config)
    merged_config = _recursive_merge(merged_config, env_config)
    
    return merged_config


def load_recipe_file(recipe_file: str) -> Dict[str, Any]:
    """
    Recipe 파일 로딩.
    절대 경로, 상대 경로, recipes/ 내부 경로 순으로 탐색합니다.
    """
    path = Path(recipe_file)
    if not path.suffix:
        path = path.with_suffix('.yaml')

    # 우선순위 1: 절대 경로
    if path.is_absolute():
        final_path = path
    # 우선순위 2: 현재 작업 디렉토리 기준 상대 경로
    elif path.exists():
        final_path = path
    # 우선순위 3: (하위 호환성) 기존 recipes/ 디렉토리 기준 경로
    else:
        final_path = BASE_DIR / "recipes" / path

    if not final_path.exists():
        raise FileNotFoundError(f"Recipe 파일을 찾을 수 없습니다. 시도한 최종 경로: {final_path}")
    
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
    
    [YAML 로드 → Jinja 렌더링 → Pydantic 검증]의 3단계 파이프라인을 구현합니다.
    """
    # 1. (기존 로직) 환경별 config 로딩
    config_data = load_config_files()
    
    # 2. (기존 로직) Recipe 파일 로딩
    recipe_data = load_recipe_file(recipe_file)
    
    # 3. (기존 로직) Recipe 데이터를 model 키 아래로 감싸기
    if recipe_data and "model" not in recipe_data:
        recipe_data = {"model": recipe_data}
    
    # 4. (기존 로직) 최종 병합: config + recipe
    final_data = _recursive_merge(config_data.copy(), recipe_data)

    # 5. (Jinja 렌더링 단계 추가)
    if context_params:
        try:
            from src.utils.system.templating_utils import render_sql_template
            
            # Loader SQL 템플릿 렌더링
            loader_config = final_data.get("model", {}).get("loader", {})
            loader_uri = loader_config.get("source_uri")
            if loader_uri and loader_uri.endswith(".sql.j2"):
                rendered_sql = render_sql_template(loader_uri, context_params)
                final_data["model"]["loader"]["source_uri"] = rendered_sql
                logger.info(f"Loader SQL template '{loader_uri}' rendered.")

            # Augmenter SQL 템플릿 렌더링 (필요 시)
            # 이 부분은 현재 아키텍처에서는 사용되지 않지만, 확장성을 위해 구조를 남겨둘 수 있습니다.

        except Exception as e:
            raise ValueError(f"Failed to render Jinja2 template: {e}") from e
    
    # 6. Settings 객체 생성 (Pydantic 검증)
    try:
        settings = Settings(**final_data)
        
        # 7. computed 필드 생성
        settings.model.computed = _create_computed_fields(settings, recipe_file)
        
        # 8. 유효성 검증
        if settings.model.augmenter:
            settings.model.augmenter.validate_augmenter_config()
        settings.model.data_interface.validate_required_fields()
        
        return settings
        
    except Exception as e:
        raise ValueError(f"Settings 객체 생성 실패: {e}\n설정 데이터: {final_data}")


def _create_computed_fields(settings: Settings, recipe_file: str) -> Dict[str, Any]:
    """computed 필드 생성"""
    from datetime import datetime
    
    # 모델 클래스에서 간단한 이름 추출
    class_name = settings.model.class_path.split('.')[-1]
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # run_name 생성
    run_name = f"{class_name}_{recipe_file}_{timestamp}"
    
    return {
        "run_name": run_name,
        "timestamp": timestamp,
        "model_class_name": class_name,
        "recipe_file": recipe_file
    }


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