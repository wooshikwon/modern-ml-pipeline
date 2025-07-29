# src/settings/_utils.py
import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
from collections.abc import Mapping

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