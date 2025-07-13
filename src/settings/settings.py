import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, RootModel
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

# --- Pydantic 모델 정의 ---

# 1. 운영 환경 설정 (config/*.yaml)
class EnvironmentSettings(BaseModel):
    app_env: str
    gcp_project_id: str
    gcp_credential_path: Optional[str] = None

class MlflowSettings(BaseModel):
    tracking_uri: str
    experiment_name: str

class RealtimeFeatureStoreConnectionSettings(BaseModel):
    host: str
    port: int
    db: int = 0

class RealtimeFeatureStoreSettings(BaseModel):
    store_type: str
    connection: RealtimeFeatureStoreConnectionSettings

class ServingSettings(BaseModel):
    model_stage: str
    realtime_feature_store: RealtimeFeatureStoreSettings

class ArtifactStoreSettings(BaseModel):
    enabled: bool
    base_uri: str

# 2. 모델 레시피 설정 (recipes/*.yaml)
class LoaderSettings(BaseModel):
    name: str
    source_uri: str
    local_override_uri: Optional[str] = None

class AugmenterSettings(BaseModel):
    name: str
    source_uri: str
    local_override_uri: Optional[str] = None

class PreprocessorParamsSettings(BaseModel):
    criterion_col: Optional[str] = None
    exclude_cols: List[str]

class PreprocessorSettings(BaseModel):
    name: str
    params: PreprocessorParamsSettings

class DataInterfaceSettings(BaseModel):
    features: Dict[str, str]
    target_col: str
    treatment_col: str
    treatment_value: Any

class ModelHyperparametersSettings(RootModel[Dict[str, Any]]):
    root: Dict[str, Any]

class ModelSettings(BaseModel):
    class_path: str  # 새로 추가: 동적 모델 로딩용
    loader: LoaderSettings
    augmenter: Optional[AugmenterSettings] = None
    preprocessor: Optional[PreprocessorSettings] = None
    data_interface: DataInterfaceSettings
    hyperparameters: ModelHyperparametersSettings
    
    # 내부 계산 필드 (런타임에 생성됨) - 언더스코어 제거로 Pydantic 호환성 확보
    computed: Optional[Dict[str, Any]] = None

# --- 최종 통합 Settings 클래스 ---
class Settings(BaseModel):
    # config/*.yaml에서 오는 필드들
    environment: EnvironmentSettings
    mlflow: MlflowSettings
    serving: ServingSettings
    artifact_stores: Dict[str, ArtifactStoreSettings]
    # recipes/*.yaml에서 오는 필드
    model: ModelSettings

# --- 설정 로드 함수 ---
def load_settings(model_name: str) -> Settings:
    """
    계층화된 설정 파일을 로드하여 통합된 Settings 객체를 반환합니다.
    base.yaml -> {APP_ENV}.yaml -> local.yaml 순서로 덮어씁니다.
    """
    config_dir = BASE_DIR / "config"
    base_config_path = config_dir / "base.yaml"
    settings_data = _load_yaml_with_env(base_config_path)

    app_env = settings_data.get("environment", {}).get("app_env", "local")
    env_config_path = config_dir / f"{app_env}.yaml"
    env_config_data = _load_yaml_with_env(env_config_path)
    settings_data = _recursive_merge(settings_data, env_config_data)

    local_config_path = config_dir / "local.yaml"
    local_config_data = _load_yaml_with_env(local_config_path)
    settings_data = _recursive_merge(settings_data, local_config_data)

    recipe_path = BASE_DIR / "recipes" / f"{model_name}.yaml"
    if not recipe_path.exists():
        raise FileNotFoundError(f"모델 레시피 파일을 찾을 수 없습니다: {recipe_path}")
    recipe_data = _load_yaml_with_env(recipe_path)

    if recipe_data.get("name") != model_name:
        raise ValueError(f"모델 레시피({recipe_path})의 name 속성이 '{model_name}'과 일치하지 않습니다.")

    combined_data = {**settings_data, "model": recipe_data}
    return Settings(**combined_data)


def load_settings_by_file(recipe_file: str) -> Settings:
    """
    recipe_file 기반으로 자유롭게 설정을 로드합니다.
    완전한 실험 자유도를 제공하며, 자동 Run Name 생성을 포함합니다.
    
    Args:
        recipe_file: recipe 파일명 (확장자 제외)
        
    Returns:
        Settings: 완전한 설정 객체
    """
    from datetime import datetime
    
    # 기존 config 로딩 로직 유지
    config_dir = BASE_DIR / "config"
    base_config_path = config_dir / "base.yaml"
    settings_data = _load_yaml_with_env(base_config_path)

    app_env = settings_data.get("environment", {}).get("app_env", "local")
    env_config_path = config_dir / f"{app_env}.yaml"
    env_config_data = _load_yaml_with_env(env_config_path)
    settings_data = _recursive_merge(settings_data, env_config_data)

    local_config_path = config_dir / "local.yaml"
    local_config_data = _load_yaml_with_env(local_config_path)
    settings_data = _recursive_merge(settings_data, local_config_data)

    # Recipe 파일 로딩 (파일명-name 일치 검증 제거!)
    recipe_path = BASE_DIR / "recipes" / f"{recipe_file}.yaml"
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe 파일을 찾을 수 없습니다: {recipe_path}")
    
    recipe_data = _load_yaml_with_env(recipe_path)
    
    # 자동 Run Name 생성 로직
    model_class_name = recipe_data["class_path"].split('.')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_class_name}_{recipe_file}_{timestamp}"
    
    # 내부 계산 필드 추가
    recipe_data["computed"] = {
        "run_name": run_name,
        "model_class_name": model_class_name,
        "recipe_file": recipe_file,
        "timestamp": timestamp
    }
    
    combined_data = {**settings_data, "model": recipe_data}
    return Settings(**combined_data)