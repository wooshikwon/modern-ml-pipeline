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
    # 🔄 기존 필드들 (하위 호환성 유지)
    name: Optional[str] = None
    source_uri: Optional[str] = None
    local_override_uri: Optional[str] = None
    
    # 🆕 Feature Store 방식 필드들 (Blueprint v17.0)
    type: Optional[str] = None  # "feature_store" or "sql" (기본값: sql)
    features: Optional[List[Dict[str, Any]]] = None  # Feature Store 피처 설정
    
    def validate_augmenter_config(self):
        """Augmenter 설정의 유효성 검증"""
        if self.type == "feature_store":
            # Feature Store 방식: features가 필요
            if not self.features:
                raise ValueError("Feature Store 방식 Augmenter에는 features 설정이 필요합니다.")
        else:
            # 기존 SQL 방식: source_uri가 필요 (기본값)
            if not self.source_uri:
                raise ValueError("기존 SQL 방식 Augmenter에는 source_uri가 필요합니다.")

class PreprocessorParamsSettings(BaseModel):
    criterion_col: Optional[str] = None
    exclude_cols: List[str]

class PreprocessorSettings(BaseModel):
    name: str
    params: PreprocessorParamsSettings

# 🆕 하이퍼파라미터 튜닝 설정 (새로 추가)
class HyperparameterTuningSettings(BaseModel):
    enabled: bool = False  # 기본값: 기존 동작 유지
    engine: str = "optuna"
    n_trials: int = 10
    metric: str = "accuracy"
    direction: str = "maximize"
    timeout: Optional[int] = None  # 초 단위, None이면 제한 없음
    pruning: Optional[Dict[str, Any]] = None
    parallelization: Optional[Dict[str, Any]] = None

# 🆕 Feature Store 설정 (새로 추가)  
class FeatureStoreSettings(BaseModel):
    provider: str = "dynamic"
    connection_timeout: int = 5000
    retry_attempts: int = 3
    connection_info: Dict[str, Any] = {}

class MLTaskSettings(BaseModel):
    # 필수 필드
    task_type: str  # "classification", "regression", "clustering", "causal"
    
    # 조건부 필수 필드들 (clustering 제외하고 필수)
    target_column: Optional[str] = None  # 🔄 수정: target_col → target_column
    
    # Causal 전용 필드들 (기존 호환성 유지)
    treatment_column: Optional[str] = None  # 🔄 수정: treatment_col → treatment_column
    treatment_value: Optional[Any] = None
    
    # Classification 전용 필드들
    class_weight: Optional[str] = None  # "balanced" 등
    pos_label: Optional[Any] = None  # 이진 분류용
    average: Optional[str] = "weighted"  # f1 계산 방식
    
    # Regression 전용 필드들
    sample_weight_column: Optional[str] = None  # 🔄 수정: sample_weight_col → sample_weight_column
    
    # Clustering 전용 필드들
    n_clusters: Optional[int] = None
    true_labels_column: Optional[str] = None  # 🔄 수정: true_labels_col → true_labels_column
    
    # 기존 필드 유지 (Optional로 변경)
    features: Optional[Dict[str, str]] = None
    
    def validate_required_fields(self):
        """task_type에 따른 동적 필수 필드 검증 (27개 Recipe 대응)"""
        if self.task_type in ["classification", "regression", "causal"]:
            if not self.target_column:  # 🔄 수정: target_col → target_column
                raise ValueError(f"{self.task_type} 모델에는 target_column이 필요합니다.")
        
        if self.task_type == "causal":
            if not self.treatment_column or self.treatment_value is None:  # 🔄 수정: treatment_col → treatment_column
                raise ValueError("causal 모델에는 treatment_column과 treatment_value가 필요합니다.")
                
        # 지원하는 task_type 검증
        supported_types = ["classification", "regression", "clustering", "causal"]
        if self.task_type not in supported_types:
            raise ValueError(f"지원하지 않는 task_type: '{self.task_type}'. 지원 가능한 타입: {supported_types}")

class ModelHyperparametersSettings(RootModel[Dict[str, Any]]):
    root: Dict[str, Any]

class ModelSettings(BaseModel):
    class_path: str  # 새로 추가: 동적 모델 로딩용
    loader: LoaderSettings
    augmenter: Optional[AugmenterSettings] = None
    preprocessor: Optional[PreprocessorSettings] = None
    data_interface: MLTaskSettings
    hyperparameters: ModelHyperparametersSettings
    
    # 🆕 새로 추가 (Optional로 하위 호환성 보장)
    hyperparameter_tuning: Optional[HyperparameterTuningSettings] = None
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
    
    # 🆕 새로 추가 (Optional로 하위 호환성 보장)
    hyperparameter_tuning: Optional[HyperparameterTuningSettings] = None
    feature_store: Optional[FeatureStoreSettings] = None

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