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
from typing import Dict, Any
from collections.abc import Mapping

from .models import Settings

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
    base.yaml + {app_env}.yaml 순서로 병합
    """
    config_dir = BASE_DIR / "config"
    
    # 1. base.yaml 로드 (모든 환경의 기본값)
    base_config = _load_yaml_with_env(config_dir / "base.yaml")
    
    # 2. 현재 환경별 설정 로드
    app_env = os.getenv("APP_ENV", "local")
    env_config_file = config_dir / f"{app_env}.yaml"
    env_config = _load_yaml_with_env(env_config_file)
    
    # 3. 병합: base + env_specific
    merged_config = _recursive_merge(base_config.copy(), env_config)
    
    return merged_config


def load_recipe_file(recipe_file: str) -> Dict[str, Any]:
    """
    Recipe 파일 로딩
    recipes/ 디렉토리에서 YAML 파일을 로드합니다.
    """
    recipes_dir = BASE_DIR / "recipes"
    
    # .yaml 확장자 자동 추가
    if not recipe_file.endswith('.yaml'):
        recipe_file += '.yaml'
    
    recipe_path = recipes_dir / recipe_file
    
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe 파일을 찾을 수 없습니다: {recipe_path}")
    
    return _load_yaml_with_env(recipe_path)


def load_settings(model_name: str) -> Settings:
    """
    모델명 기반 설정 로딩 (기존 호환성)
    
    Args:
        model_name: 모델명 (recipes/{model_name}.yaml)
        
    Returns:
        완전히 병합된 Settings 객체
    """
    return load_settings_by_file(f"models/{model_name}")


def load_settings_by_file(recipe_file: str) -> Settings:
    """
    Blueprint v17.0 통합 설정 로딩
    
    Args:
        recipe_file: Recipe 파일 경로 (recipes/ 기준 상대 경로)
        
    Returns:
        config/*.yaml + recipes/*.yaml이 병합된 Settings 객체
        
    Example:
        settings = load_settings_by_file("models/classification/xgboost_classifier")
        settings = load_settings_by_file("uplift_model_exp3")
    """
    # 1. 환경별 config 로딩
    config_data = load_config_files()
    
    # 2. Recipe 파일 로딩
    recipe_data = load_recipe_file(recipe_file)
    
    # 3. Recipe 데이터를 model 키 아래로 감싸기 (Settings 모델 구조 준수)
    if recipe_data and "model" not in recipe_data:
        # recipe 데이터가 있고 model 키가 없는 경우 자동으로 감싸기
        recipe_data = {"model": recipe_data}
    
    # 4. 최종 병합: config + recipe
    # Recipe의 내용을 직접 병합 (Blueprint 원칙: 레시피는 논리)
    final_data = _recursive_merge(config_data.copy(), recipe_data)
    
    # 5. Settings 객체 생성
    try:
        settings = Settings(**final_data)
        
        # 6. computed 필드 생성
        settings.model.computed = _create_computed_fields(settings, recipe_file)
        
        # 7. 유효성 검증
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