"""
Config Schema - Infrastructure Settings (v3.0)
CLI 템플릿과 완벽히 호환되는 새로운 구조
완전히 재작성됨 - CLI config.yaml.j2와 100% 호환
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, Literal


class Environment(BaseModel):
    """환경 설정 - CLI 템플릿과 동일하게 단순화"""
    name: str = Field(..., description="환경 이름 (local, dev, prod 등)")


class MLflow(BaseModel):
    """MLflow 실험 추적 설정"""
    tracking_uri: str = Field(..., description="MLflow tracking server URI 또는 로컬 경로")
    experiment_name: str = Field(..., description="MLflow 실험 이름")
    tracking_username: Optional[str] = Field("", description="MLflow 인증 사용자명")
    tracking_password: Optional[str] = Field("", description="MLflow 인증 비밀번호") 
    s3_endpoint_url: Optional[str] = Field("", description="S3 호환 스토리지 엔드포인트 (MinIO 등)")


class DataSource(BaseModel):
    """데이터 소스 설정 - CLI 템플릿 구조 그대로"""
    name: str = Field(..., description="데이터 소스 이름")
    adapter_type: Literal["sql", "storage"] = Field(..., description="어댑터 타입")  # bigquery 제거
    config: Dict[str, Any] = Field(default_factory=dict, description="어댑터별 설정")
    
    # TODO: Pydantic V2 validator


class FeastOnlineStore(BaseModel):
    """Feast Online Store 설정"""
    type: Literal["redis", "dynamodb", "sqlite"] = Field(..., description="Online store 타입")
    # Redis 설정
    connection_string: Optional[str] = Field(None, description="Redis 연결 문자열")
    password: Optional[str] = Field("", description="Redis 비밀번호")
    # DynamoDB 설정
    region: Optional[str] = Field(None, description="DynamoDB 리전")
    table_name: Optional[str] = Field(None, description="DynamoDB 테이블 이름")
    # SQLite 설정
    path: Optional[str] = Field(None, description="SQLite 파일 경로")
    
    # TODO: Add Pydantic V2 validators when needed


class FeastOfflineStore(BaseModel):
    """Feast Offline Store 설정"""
    type: Literal["bigquery", "file"] = Field(..., description="Offline store 타입")
    # BigQuery 설정
    project_id: Optional[str] = Field(None, description="GCP 프로젝트 ID")
    dataset_id: Optional[str] = Field(None, description="BigQuery 데이터셋 ID")
    # File 설정
    path: Optional[str] = Field(None, description="Parquet 파일 저장 경로")
    
    # TODO: Pydantic V2 validator
    
    # TODO: Pydantic V2 validator


class FeastConfig(BaseModel):
    """Feast Feature Store 설정"""
    project: str = Field(..., description="Feast 프로젝트 이름")
    registry: str = Field(..., description="Feature/Entity 정의를 저장하는 Registry 경로")
    online_store: FeastOnlineStore = Field(..., description="실시간 서빙용 Online Store")
    offline_store: FeastOfflineStore = Field(..., description="Historical features용 Offline Store")
    entity_key_serialization_version: int = Field(2, description="Entity key 직렬화 버전")


class FeatureStore(BaseModel):
    """Feature Store 설정"""
    provider: Literal["feast", "none"] = Field(..., description="Feature store provider")
    feast_config: Optional[FeastConfig] = Field(None, description="Feast 설정 (provider가 feast일 때)")
    
    # TODO: Pydantic V2 validator


class AuthConfig(BaseModel):
    """API 인증 설정"""
    enabled: bool = Field(False, description="인증 활성화 여부")
    type: Literal["jwt", "basic", "oauth"] = Field("jwt", description="인증 타입")
    secret_key: Optional[str] = Field("", description="인증 시크릿 키")


class Serving(BaseModel):
    """API 서빙 설정"""
    model_config = {"protected_namespaces": ()}
    
    enabled: bool = Field(False, description="서빙 활성화 여부")
    host: str = Field("0.0.0.0", description="API 서버 호스트")
    port: int = Field(8000, ge=1024, le=65535, description="API 서버 포트")
    workers: int = Field(1, ge=1, description="워커 프로세스 수")
    model_stage: Optional[str] = Field("None", description="MLflow 모델 스테이지")
    auth: Optional[AuthConfig] = Field(None, description="인증 설정")


class ArtifactStore(BaseModel):
    """MLflow 아티팩트 저장소 설정"""
    type: Literal["local", "s3", "gcs"] = Field(..., description="저장소 타입")
    config: Dict[str, Any] = Field(default_factory=dict, description="저장소별 설정")
    
    # TODO: Pydantic V2 validator


class OutputTarget(BaseModel):
    """출력 타겟 설정 - 데이터 소스 스타일과 동일한 adapter_type + config 구조"""
    name: str = Field(..., description="출력 타겟 이름")
    enabled: bool = Field(True, description="저장 활성화 여부")
    adapter_type: Optional[Literal["storage", "sql"]] = Field(None, description="저장 어댑터 타입")  # bigquery 제거
    config: Optional[Dict[str, Any]] = Field(default=None, description="어댑터별 설정 (base_path/table 등)")


class Output(BaseModel):
    """출력 설정 - inference 결과 및 전처리 결과"""
    inference: OutputTarget = Field(..., description="배치 추론 출력 설정")
    preprocessed: OutputTarget = Field(..., description="전처리 결과 출력 설정")


class Config(BaseModel):
    """
    루트 인프라 설정 (configs/*.yaml)
    CLI config.yaml.j2 템플릿과 100% 호환
    """
    environment: Environment = Field(..., description="환경 설정")
    mlflow: Optional[MLflow] = Field(None, description="MLflow 설정")
    data_source: DataSource = Field(..., description="데이터 소스 설정")
    feature_store: FeatureStore = Field(
        default_factory=lambda: FeatureStore(provider="none"),
        description="Feature Store 설정"
    )
    serving: Optional[Serving] = Field(None, description="API 서빙 설정")
    artifact_store: Optional[ArtifactStore] = Field(None, description="아티팩트 저장소 설정")
    output: Optional[Output] = Field(None, description="출력 저장 설정 (선택 사항)")
    
    def get_adapter_config(self) -> Dict[str, Any]:
        """데이터 소스에서 어댑터 설정 추출 (호환성용)"""
        return {
            "type": self.data_source.adapter_type,
            "config": self.data_source.config
        }
    
    def has_feast(self) -> bool:
        """Feast 사용 여부 확인"""
        return self.feature_store.provider == "feast"
    
    def get_feast_config(self) -> Optional[FeastConfig]:
        """Feast 설정 반환"""
        if self.has_feast():
            return self.feature_store.feast_config
        return None
    
    class Config:
        """Pydantic 설정"""
        json_schema_extra = {
            "example": {
                "environment": {
                    "name": "local"
                },
                "mlflow": {
                    "tracking_uri": "./mlruns",
                    "experiment_name": "mmp-local"
                },
                "data_source": {
                    "name": "PostgreSQL",
                    "adapter_type": "sql",
                    "config": {
                        "connection_uri": "postgresql://user:pass@localhost:5432/db",
                        "query_timeout": 30
                    }
                },
                "feature_store": {
                    "provider": "feast",
                    "feast_config": {
                        "project": "feast_local",
                        "registry": "./feast_repo/registry.db",
                        "online_store": {
                            "type": "sqlite",
                            "path": "./feast_repo/online_store.db"
                        },
                        "offline_store": {
                            "type": "file",
                            "path": "./feast_repo/data"
                        }
                    }
                },
                "serving": {
                    "enabled": True,
                    "port": 8000,
                    "workers": 1
                }
            }
        }