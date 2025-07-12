import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from collections.abc import Mapping

# --- 기본 경로 및 환경 변수 로더 ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

_env_var_pattern = re.compile(r"\$\{([^}:\s]+)(?::([^}]*))?\}")

def _env_var_replacer(m: re.Match) -> str:
    env_var = m.group(1)
    default_value = m.group(2)
    return os.getenv(env_var, default_value or "")

def _load_yaml_with_env(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {}
    text = file_path.read_text(encoding="utf-8")
    return yaml.safe_load(re.sub(_env_var_pattern, _env_var_replacer, text)) or {}

def _recursive_merge(dict1: Dict, dict2: Dict) -> Dict:
    """두 딕셔너리를 재귀적으로 병합합니다. dict2의 값이 dict1의 값을 덮어씁니다."""
    for k, v in dict2.items():
        if k in dict1 and isinstance(dict1[k], Mapping) and isinstance(v, Mapping):
            dict1[k] = _recursive_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1

# --- Pydantic 모델 정의 (이전과 동일) ---
# ... (이전 Pydantic 모델 정의는 여기에 그대로 유지됩니다) ...

# --- 설정 로드 함수 (재설계) ---
def load_settings(model_name: str) -> Settings:
    """
    계층화된 설정 파일을 로드하여 통합된 Settings 객체를 반환합니다.
    base.yaml -> {APP_ENV}.yaml -> local.yaml 순서로 덮어씁니다.
    """
    # 1. 기본 설정 로드
    base_config_path = BASE_DIR / "config" / "base.yaml"
    settings_data = _load_yaml_with_env(base_config_path)

    # 2. 환경별 설정 로드 (덮어쓰기)
    app_env = os.getenv("APP_ENV", "local")
    env_config_path = BASE_DIR / "config" / f"{app_env}.yaml"
    env_config_data = _load_yaml_with_env(env_config_path)
    settings_data = _recursive_merge(settings_data, env_config_data)

    # 3. 로컬 개인 설정 로드 (덮어쓰기, 버전 관리 안 됨)
    local_config_path = BASE_DIR / "config" / "local.yaml"
    local_config_data = _load_yaml_with_env(local_config_path)
    settings_data = _recursive_merge(settings_data, local_config_data)

    # 4. 모델 레시피 로드
    recipe_path = BASE_DIR / "recipes" / f"{model_name}.yaml"
    if not recipe_path.exists():
        raise FileNotFoundError(f"모델 레시피 파일을 찾을 수 없습니다: {recipe_path}")
    recipe_data = _load_yaml_with_env(recipe_path)

    if recipe_data.get("name") != model_name:
        raise ValueError(f"모델 레시피 파일의 모델 이름 불일치")

    # 5. 최종 병합
    combined_data = {**settings_data, "model": recipe_data}

    return Settings(**combined_data)