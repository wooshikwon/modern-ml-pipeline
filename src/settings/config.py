"""순수 Config Pydantic 스키마 - 검증 로직 완전 제거"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, Literal


class Environment(BaseModel):
    name: str = Field(..., description="환경 이름")


class MLflow(BaseModel):
    tracking_uri: str = Field(..., description="MLflow tracking URI")
    experiment_name: str = Field(..., description="실험 이름")
    tracking_username: Optional[str] = Field(None, description="MLflow 인증 사용자명")
    tracking_password: Optional[str] = Field(None, description="MLflow 인증 비밀번호")
    s3_endpoint_url: Optional[str] = Field(None, description="S3 호환 엔드포인트 URL")


# DataSource 어댑터별 Config 모델들
class PostgreSQLConfig(BaseModel):
    connection_uri: str = Field(..., description="PostgreSQL 연결 URI")
    query_timeout: int = Field(default=300, description="쿼리 타임아웃(초)")


class BigQueryConfig(BaseModel):
    connection_uri: str = Field(..., description="BigQuery 연결 URI")
    project_id: str = Field(..., description="GCP 프로젝트 ID")
    dataset_id: str = Field(..., description="BigQuery 데이터셋 ID")
    location: str = Field(default="US", description="BigQuery 위치")
    use_pandas_gbq: bool = Field(default=True, description="pandas_gbq 사용 여부")
    query_timeout: int = Field(default=300, description="쿼리 타임아웃(초)")


class LocalFilesConfig(BaseModel):
    base_path: str = Field(..., description="로컬 파일 기본 경로")
    storage_options: Dict[str, Any] = Field(default_factory=dict, description="저장소 옵션")


class S3StorageOptions(BaseModel):
    aws_access_key_id: Optional[str] = Field(None, description="AWS Access Key ID")
    aws_secret_access_key: Optional[str] = Field(None, description="AWS Secret Access Key")
    region_name: str = Field(default="us-east-1", description="AWS 리전")


class S3Config(BaseModel):
    base_path: str = Field(..., description="S3 기본 경로")
    storage_options: S3StorageOptions = Field(..., description="S3 저장소 옵션")


class GCSStorageOptions(BaseModel):
    project: Optional[str] = Field(None, description="GCP 프로젝트 ID")
    token: Optional[str] = Field(None, description="인증 토큰")


class GCSConfig(BaseModel):
    base_path: str = Field(..., description="GCS 기본 경로")
    storage_options: GCSStorageOptions = Field(..., description="GCS 저장소 옵션")


class DataSource(BaseModel):
    name: str = Field(..., description="데이터 소스 이름")
    adapter_type: str = Field(..., description="어댑터 타입")
    config: Union[PostgreSQLConfig, BigQueryConfig, LocalFilesConfig, S3Config, GCSConfig] = Field(..., description="어댑터별 설정")


# Feast Online Store 타입별 모델들
class RedisOnlineStore(BaseModel):
    type: Literal["redis"] = "redis"
    connection_string: str = Field(..., description="Redis 연결 문자열")
    password: Optional[str] = Field(None, description="Redis 비밀번호")


class DynamoDBOnlineStore(BaseModel):
    type: Literal["dynamodb"] = "dynamodb"
    region: str = Field(..., description="AWS 리전")
    table_name: str = Field(..., description="DynamoDB 테이블 이름")


class SQLiteOnlineStore(BaseModel):
    type: Literal["sqlite"] = "sqlite"
    path: str = Field(..., description="SQLite 파일 경로")


# Feast Offline Store 타입별 모델들
class BigQueryOfflineStore(BaseModel):
    type: Literal["bigquery"] = "bigquery"
    project_id: str = Field(..., description="GCP 프로젝트 ID")
    dataset_id: str = Field(..., description="BigQuery 데이터셋 ID")


class FileOfflineStore(BaseModel):
    type: Literal["file"] = "file"
    path: str = Field(..., description="파일 저장 경로")


# Union 타입 정의
FeastOnlineStore = Union[RedisOnlineStore, DynamoDBOnlineStore, SQLiteOnlineStore]
FeastOfflineStore = Union[BigQueryOfflineStore, FileOfflineStore]


class FeastConfig(BaseModel):
    project: str = Field(..., description="Feast 프로젝트 이름")
    registry: str = Field(..., description="Registry 경로")
    online_store: FeastOnlineStore
    offline_store: FeastOfflineStore
    entity_key_serialization_version: int = Field(default=2, description="엔티티 키 직렬화 버전")


class FeatureStore(BaseModel):
    provider: str = Field(default="none", description="Feature Store 제공자")
    feast_config: Optional[FeastConfig] = Field(None, description="Feast 설정")


class AuthConfig(BaseModel):
    enabled: bool = Field(default=False, description="인증 활성화")
    type: str = Field(default="jwt", description="인증 타입")
    secret_key: Optional[str] = Field(None, description="인증 시크릿 키")


class Serving(BaseModel):
    enabled: bool = Field(default=False, description="서빙 활성화")
    host: str = Field(default="0.0.0.0", description="서빙 호스트")
    port: int = Field(default=8000, description="서빙 포트")
    workers: int = Field(default=1, description="워커 수")
    model_stage: Optional[str] = Field(None, description="모델 스테이지")
    auth: Optional[AuthConfig] = Field(None, description="인증 설정")


class ArtifactStore(BaseModel):
    type: str = Field(..., description="아티팩트 저장소 타입")
    config: Dict[str, Any] = Field(..., description="저장소 설정")


# Output 어댑터별 Config 모델들
class StorageOutputConfig(BaseModel):
    base_path: str = Field(..., description="저장 기본 경로")


class SQLOutputConfig(BaseModel):
    connection_uri: str = Field(..., description="데이터베이스 연결 URI")
    table: str = Field(..., description="테이블 이름")


class BigQueryOutputConfig(BaseModel):
    connection_uri: str = Field(..., description="BigQuery 연결 URI")
    project_id: str = Field(..., description="GCP 프로젝트 ID")
    dataset_id: str = Field(..., description="BigQuery 데이터셋 ID")
    table: str = Field(..., description="테이블 이름")
    location: str = Field(default="US", description="BigQuery 위치")
    use_pandas_gbq: bool = Field(default=True, description="pandas_gbq 사용 여부")


class OutputTarget(BaseModel):
    name: str = Field(..., description="출력 대상 이름")
    enabled: bool = Field(default=True, description="출력 활성화")
    adapter_type: str = Field(..., description="어댑터 타입")
    config: Union[StorageOutputConfig, SQLOutputConfig, BigQueryOutputConfig] = Field(..., description="어댑터별 출력 설정")


class Output(BaseModel):
    inference: OutputTarget = Field(..., description="추론 결과 출력")


class Config(BaseModel):
    """Config 스키마 - 순수 데이터 구조만"""
    environment: Environment
    mlflow: Optional[MLflow] = None
    data_source: DataSource
    feature_store: FeatureStore
    serving: Optional[Serving] = None
    artifact_store: Optional[ArtifactStore] = None
    output: Output

    # 검증 로직은 validation/ 으로 완전 분리