import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

"""
1. 환경 변수를 로드하기
    - .env 파일에 정의된 환경 변수를 인자로 가지고 있는 config.yaml 파일을 파싱하여 딕셔너리 형식으로 반환한다.
    - _load_yaml_with_env 함수의 최종 return 값 예시
        {
        "environment": {
            "run_mode": "local",
            "gcp_credential_path": "/path/to/cred.json",
            "gcp_project_id": "my-gcp-project"
        },
        "loader": {
            "abt_logs": {
                "sql_file_path": "src/sql/abt_logs.sql",
        (...)
"""

# Path(__file__) : 현재 파일의 경로를 가져옴
# resolve() : 경로를 절대 경로로 변환
# parent : 부모 디렉토리를 가져옴
BASE_DIR = Path(__file__).resolve().parent.parent

# 환경 변수 로드
load_dotenv(BASE_DIR / ".env")

# config.yaml 파일에서 ${...} 형식의 환경 변수를 참조하기 위한 함수
# 함수명 앞에 _는 내부에서만 활용되는 비공개 함수임을 나타내는 관례
    # re.compile() : 정규 표현식 패턴을 컴파일하여 정규 표현식 객체로 반환
_env_var_pattern = re.compile(r"\$\{([^}:\s]+)(?::([^}]*))?\}")

def _env_var_replacer(m: re.Match) -> str:
    # 정규 표현식 패턴에서 첫 번째 그룹(env_var)과 두 번째 그룹(default_value)을 추출
    env_var = m.group(1)
    default_value = m.group(2)
    # os.getenv() 함수는 환경 변수 'key'가 존재하면 해당 환경 변수의 값을 반환하고, 존재하지 않으면 default_value를 반환
    return os.getenv(env_var, default_value or "")

def _load_yaml_with_env(file_path: str) -> Dict[str, Any]:
    # 함수 인자로 받은 파일 경로의 text(yaml 파일)를 **문자열(str)**로 읽음. 한글이므로 utf-8로 인코딩
    text = file_path.read_text(encoding="utf-8")

    # yaml.safe_load() 함수는 yaml 형식의 **문자열**을 파싱하여 **딕셔너리**로 반환
    # re.sub(pattern, repl, string, count=0, flags=0) -- pattern은 정규 표현식 컴파일된 패턴, repl은 함수 혹은 문자열, string은 치환 적용할 원본 문자열
        # _env_var_pattern 패턴에 매칭되는 부분을 _repl 함수로 치환하여 새로운 문자열을 반환
    return yaml.safe_load(re.sub(_env_var_pattern, _env_var_replacer, text))


"""
2. Pydantic 설정
    - 위에서 로드한 딕셔너리 형식의 객체를 클래스 인스턴스로 변환하여 환경 변수의 **형식**을 관리한다.
    - 아래 코드를 덕분에 우리는 config.yaml 파일을 아래의 형식으로 꺼내 쓸 수 있게 된다
        - print(settings.loader.abt_logs.output.type)
            # 출력: bigquery
        - print(settings.transformer.params.criterion_col)
            # 출력: rsvn_30_count
"""

# BaseSettings는 클래스 생성 시, 자동으로 정의한 환경 변수를 로드하고 설정 값을 초기화하는 기능을 제공
# 그러나, 이 프로젝트에서 환경을 config.yaml 파일에서 관리하고 있으므로, BaseSettings를 상속하지 않고, BaseModel을 사용한다.
# BaseModel은 데이터 형식을 검증하고 직렬화/역직렬화를 자동으로 처리해준다.


# --- 일반 환경 변수 설정
# config.yaml 파일에 정의된 기초 환경 변수를 클래스 인스턴스로 변환
class EnvironmentSettings(BaseModel):
    run_mode: str
    gcp_credential_path: Optional[str] = None
    gcp_project_id: str


# -- Loader 설정
# config.yaml 파일의 loader 설정 중 output 설정을 클래스 인스턴스로 변환
class LoaderOutputSettings(BaseModel):
    type: str
    project_id: str
    dataset_id: str
    table_id: str
    unique_col: str

# config.yaml 파일의 loader 설정을 클래스 인스턴스로 변환
# loader - output 설정을 하위 클래스로 가지고 있다.
class LoaderSettings(BaseModel):
    sql_file_path: str
    output: LoaderOutputSettings


# --- Transformer 설정
# config.yaml 파일의 transformer 설정 중 output 설정을 클래스 인스턴스로 변환
class TransformerOutputSettings(BaseModel):
    type: str
    bucket_name: str

# config.yaml 파일의 transformer 설정 중 params 설정을 클래스 인스턴스로 변환
class TransformerParamsSettings(BaseModel):
    criterion_col: str
    exclude_cols: List[str]

# config.yaml 파일의 transformer 설정을 클래스 인스턴스로 변환
# transformer - output, params 설정을 하위 클래스로 가지고 있다.
class TransformerSettings(BaseModel):
    params: TransformerParamsSettings
    output: TransformerOutputSettings


# -- Model 설정
class DataInterfaceSettings(BaseModel):
    features: List[str]
    target_col: str
    treatment_col: str

class ModelHyperparametersSettings(BaseModel):
    objective: str
    eval_metric: str


# --- Mlflow 설정
class MlflowSettings(BaseModel):
    tracking_uri: str
    experiment_name: str


# --- 최종 Settings 클래스 정의
class Settings(BaseModel):
    environment: EnvironmentSettings
    loader: LoaderSettings
    transformer: TransformerSettings
    mlflow: MlflowSettings
    model: ModelSettings


"""
3. 설정 클래스 인스턴스 생성
    - 이제 위에 정의된 함수 및 클래스를 실제로 활용하여 경로에 있는 yaml 파일을 불러와 설정 값을 초기화, 구조화
"""

def load_settings(model_name: str) -> Settings:
    """
    환경 설정(config.yaml)과 지정된 모델 레시피를 로드하여 통합된 Settings 객체를 반환합니다.
    """
    # 프로젝트 루트의 'config/' 디렉터리 안에 config.yaml 파일
    config_data = _load_yaml_with_env(BASE_DIR / "config.yaml")

    # 프로젝트 루트의 'recipe/' 디렉터리 안에 모델 레시피 파일
    model_recipe_path = BASE_DIR / "recipe" / f"{model_name}.yaml"

    if not model_recipe_path.exists():
        raise FileNotFoundError(f"모델 레시피 파일을 찾을 수 없습니다: {model_recipe_path}")
    model_recipe_data = _load_yaml_with_env(model_recipe_path)

    # 모델 레시피 파일의 'name' 필드와 요청된 model_name이 일치하는지 검증합니다.
    if model_recipe_data.get("name") != model_name:
        raise ValueError(
            f"모델 레시피 파일의 모델 이름이 일치하지 않습니다: "
            f"- 모델 레시피 파일 내부에 정의된 모델명: '{model_recipe_data.get('name')}'"
            f"- 요청된 모델 레시피 파일명: '{model_name}'"
        )

    # config.yaml 데이터와 recipe/{model_name}.yaml 데이터를 합칩니다.
    combined_data = {**config_data, "model": model_recipe_data}

    # 합쳐진 데이터를 Pydantic 모델에 맞게 파싱하여 최종 Settings 객체를 생성하고 반환합니다.
    return Settings(**combined_data)