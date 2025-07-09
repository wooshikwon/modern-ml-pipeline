import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# --- 기본 경로 및 환경 변수 로더 ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

_env_var_pattern = re.compile(r"\$\{([^}:\s]+)(?::([^}]*))?\}")

def _env_var_replacer(m: re.Match) -> str:
    env_var = m.group(1)
    default_value = m.group(2)
    return os.getenv(env_var, default_value or "")

def _load_yaml_with_env(file_path: Path) -> Dict[str, Any]:
    text = file_path.read_text(encoding="utf-8")
    return yaml.safe_load(re.sub(_env_var_pattern, _env_var_replacer, text))

# --- Pydantic 모델 정의 ---

# 환경 설정
class EnvironmentSettings(BaseModel):
    run_mode: str
    gcp_credential_path: Optional[str] = None
    gcp_project_id: str

# Loader 설정
class LoaderOutputSettings(BaseModel):
    type: str
    project_id: str
    dataset_id: str
    table_id: str
    unique_col: str

class LoaderSettings(BaseModel):
    sql_file_path: str
    output: LoaderOutputSettings

# Preprocessor 설정 (기존 TransformerSettings에서 이름 변경)
class PreprocessorOutputSettings(BaseModel):
    type: str
    bucket_name: str

class PreprocessorParamsSettings(BaseModel):
    criterion_col: str
    exclude_cols: List[str]

class PreprocessorSettings(BaseModel):
    params: PreprocessorParamsSettings
    output: PreprocessorOutputSettings

# Augmenter 설정 (신규 추가)
class AugmenterBatchSettings(BaseModel):
    feature_store_sql_path: str

class AugmenterRealtimeSettings(BaseModel):
    type: str
    host: str
    port: int

class AugmenterSettings(BaseModel):
    batch: AugmenterBatchSettings
    realtime: AugmenterRealtimeSettings

# Model 설정 (신규 및 수정)
class DataInterfaceSettings(BaseModel):
    features: List[str]
    target_col: str
    treatment_col: str
    treatment_value: Any  # 실험군 값을 명시적으로 정의

class ModelHyperparametersSettings(BaseModel):
    # 특정 하이퍼파라미터에 국한되지 않도록 유연한 딕셔너리 형태로 정의
    __root__: Dict[str, Any]

class ModelSettings(BaseModel):
    name: str
    data_interface: DataInterfaceSettings
    hyperparameters: ModelHyperparametersSettings

# Mlflow 설정
class MlflowSettings(BaseModel):
    tracking_uri: str
    experiment_name: str

# --- 최종 통합 Settings 클래스 ---
class Settings(BaseModel):
    environment: EnvironmentSettings
    loader: Dict[str, LoaderSettings]  # 여러 로더 설정을 지원하도록 Dict로 변경
    augmenter: AugmenterSettings # augmenter 설정 추가
    preprocessor: PreprocessorSettings  # transformer -> preprocessor
    mlflow: MlflowSettings
    model: ModelSettings

# --- 설정 로드 함수 ---
def load_settings(model_name: str) -> Settings:
    """
    환경 설정(config.yaml)과 지정된 모델 레시피를 로드하여 통합된 Settings 객체를 반환합니다.
    """
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    config_data = _load_yaml_with_env(config_path)

    model_recipe_path = BASE_DIR / "recipe" / f"{model_name}.yaml"
    if not model_recipe_path.exists():
        raise FileNotFoundError(f"모델 레시피 파일을 찾을 수 없습니다: {model_recipe_path}")
    model_recipe_data = _load_yaml_with_env(model_recipe_path)

    if model_recipe_data.get("name") != model_name:
        raise ValueError(
            f"모델 레시피 파일의 모델 이름이 일치하지 않습니다: "
            f"파일 내 이름 '{model_recipe_data.get('name')}', 요청된 이름 '{model_name}'"
        )

    # config.yaml 데이터와 recipe/{model_name}.yaml 데이터를 병합
    combined_data = {**config_data, "model": model_recipe_data}

    # Pydantic 모델로 파싱하여 최종 Settings 객체 생성 및 반환
    return Settings(**combined_data)
