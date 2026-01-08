"""순수 Config Pydantic 스키마 - 검증 로직 완전 제거"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Environment(BaseModel):
    name: str = Field(..., description="환경 이름")
    description: Optional[str] = Field(None, description="환경 설명")


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
    config: Union[PostgreSQLConfig, BigQueryConfig, LocalFilesConfig, S3Config, GCSConfig] = Field(
        ..., description="어댑터별 설정"
    )


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


class PostgresOfflineStore(BaseModel):
    type: Literal["postgres"] = "postgres"
    host: str = Field(..., description="PostgreSQL 호스트")
    port: int = Field(default=5432, description="PostgreSQL 포트")
    database: str = Field(..., description="데이터베이스 이름")
    db_schema: Optional[str] = Field(None, description="스키마 이름")
    user: str = Field(..., description="사용자명")
    password: Optional[str] = Field(None, description="비밀번호")


# Union 타입 정의
FeastOnlineStore = Union[RedisOnlineStore, DynamoDBOnlineStore, SQLiteOnlineStore]
FeastOfflineStore = Union[BigQueryOfflineStore, FileOfflineStore, PostgresOfflineStore]


class FeastConfig(BaseModel):
    project: str = Field(..., description="Feast 프로젝트 이름")
    registry: str = Field(..., description="Registry 경로")
    provider: str = Field(default="local", description="Feast provider (local, gcp, aws)")
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


class CORSConfig(BaseModel):
    """CORS 설정 모델. 브라우저 기반 클라이언트 접근 시 필요."""

    enabled: bool = Field(default=False, description="CORS 활성화")
    allow_origins: List[str] = Field(default=["*"], description="허용할 Origin 목록")
    allow_methods: List[str] = Field(default=["*"], description="허용할 HTTP 메서드")
    allow_headers: List[str] = Field(default=["*"], description="허용할 헤더")
    allow_credentials: bool = Field(default=False, description="자격 증명 허용")


class Serving(BaseModel):
    """서빙 설정 모델."""

    model_config = {"protected_namespaces": ()}

    enabled: bool = Field(default=False, description="서빙 활성화")
    host: str = Field(default="0.0.0.0", description="서빙 호스트")
    port: int = Field(default=8000, description="서빙 포트")
    workers: int = Field(default=1, description="워커 수")
    model_stage: Optional[str] = Field(None, description="모델 스테이지")
    auth: Optional[AuthConfig] = Field(None, description="인증 설정")
    cors: Optional[CORSConfig] = Field(None, description="CORS 설정")
    request_timeout_seconds: int = Field(default=30, description="요청 타임아웃(초)")
    metrics_enabled: bool = Field(default=True, description="Prometheus 메트릭 활성화")


class Logging(BaseModel):
    """로깅 설정 모델"""

    base_path: str = Field(default="./logs", description="로그 저장 경로")
    level: str = Field(default="INFO", description="로그 레벨 (DEBUG, INFO, WARNING, ERROR)")
    retention_days: int = Field(default=30, description="로그 파일 보관 일수")
    upload_to_mlflow: bool = Field(
        default=True, description="학습 완료 시 MLflow artifact로 업로드"
    )


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
    config: Union[StorageOutputConfig, SQLOutputConfig, BigQueryOutputConfig] = Field(
        ..., description="어댑터별 출력 설정"
    )


class Output(BaseModel):
    inference: OutputTarget = Field(..., description="추론 결과 출력")


class Config(BaseModel):
    """Config 스키마 - 순수 데이터 구조만"""

    environment: Environment
    mlflow: Optional[MLflow] = None
    data_source: DataSource
    feature_store: FeatureStore
    serving: Optional[Serving] = None
    output: Output
    logging: Optional[Logging] = Field(default_factory=Logging, description="로깅 설정")

    # 검증 로직은 validation/ 으로 완전 분리
