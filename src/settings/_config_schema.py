# src/settings/_config_schema.py
from pydantic import BaseModel
from typing import Dict, Any, Optional
import requests
import os
from src.utils.system.logger import logger

class EnvironmentSettings(BaseModel):
    """환경별 기본 설정"""
    app_env: str
    gcp_project_id: str
    gcp_credential_path: Optional[str] = None

class MlflowSettings(BaseModel):
    """MLflow 실험 추적 설정"""
    tracking_uri: str
    experiment_name: str

    @classmethod
    def with_fallback(
        cls,
        server_uri: str,
        experiment_name: str,
        fallback_uri: Optional[str] = None,
        timeout: int = 5
    ) -> 'MlflowSettings':
        """
        MLflow Graceful Degradation - 서버 연결 실패 시 로컬 파일 모드로 fallback
        
        Args:
            server_uri: MLflow 서버 URI
            experiment_name: 실험명
            fallback_uri: 폴백 URI (기본: "./mlruns")
            timeout: 서버 연결 타임아웃 (초)
            
        Returns:
            MlflowSettings: 서버 모드 또는 폴백 모드 설정
        """
        if fallback_uri is None:
            fallback_uri = "./mlruns"
        
        # 서버 연결 테스트
        try:
            health_url = f"{server_uri}/health"
            response = requests.get(health_url, timeout=timeout)
            
            if response.status_code == 200:
                logger.info(f"MLflow 서버 연결 성공: {server_uri}")
                return cls(tracking_uri=server_uri, experiment_name=experiment_name)
            else:
                logger.warning(f"MLflow 서버 응답 오류 ({response.status_code}), 폴백 모드로 전환: {fallback_uri}")
                return cls(tracking_uri=fallback_uri, experiment_name=experiment_name)
                
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f"MLflow 서버 연결 실패: {e}, 폴백 모드로 전환: {fallback_uri}")
            return cls(tracking_uri=fallback_uri, experiment_name=experiment_name)
        except Exception as e:
            logger.error(f"MLflow 서버 연결 중 예상치 못한 오류: {e}, 폴백 모드로 전환: {fallback_uri}")
            return cls(tracking_uri=fallback_uri, experiment_name=experiment_name)

    @classmethod
    def auto_detect(
        cls,
        experiment_name: str,
        fallback_uri: Optional[str] = None,
        timeout: int = 5
    ) -> 'MlflowSettings':
        """
        환경 변수 기반 MLflow 설정 자동 감지
        
        MLFLOW_TRACKING_URI 환경변수 존재 여부로 서버/파일 모드를 자동 결정
        
        Args:
            experiment_name: 실험명
            fallback_uri: 폴백 URI (기본: "./mlruns")
            timeout: 서버 연결 타임아웃 (초)
            
        Returns:
            MlflowSettings: 자동 감지된 설정
        """
        if fallback_uri is None:
            fallback_uri = "./mlruns"
        
        # 환경변수에서 MLflow 서버 URI 확인
        server_uri = os.getenv('MLFLOW_TRACKING_URI')
        
        if not server_uri or server_uri.strip() == '':
            # 환경변수 없거나 빈 문자열이면 바로 폴백 모드
            logger.info(f"MLFLOW_TRACKING_URI 환경변수 없음, 로컬 파일 모드 사용: {fallback_uri}")
            return cls(tracking_uri=fallback_uri, experiment_name=experiment_name)
        else:
            # 환경변수 있으면 서버 연결 시도 → 실패시 폴백
            logger.info(f"MLFLOW_TRACKING_URI 환경변수 감지: {server_uri}, 서버 연결 시도 중...")
            return cls.with_fallback(
                server_uri=server_uri,
                experiment_name=experiment_name, 
                fallback_uri=fallback_uri,
                timeout=timeout
            )

class AdapterConfigSettings(BaseModel):
    """개별 어댑터 설정"""
    class_name: str
    config: Dict[str, Any] = {}

class DataAdapterSettings(BaseModel):
    """데이터 어댑터 설정 - Config-driven Dynamic Factory"""
    default_loader: str = "filesystem"
    default_storage: str = "filesystem"
    default_feature_store: str = "filesystem"
    adapters: Dict[str, AdapterConfigSettings] = {}

    def get_adapter_config(self, adapter_name: str) -> AdapterConfigSettings:
        if adapter_name not in self.adapters:
            raise ValueError(f"어댑터 설정을 찾을 수 없습니다: {adapter_name}")
        return self.adapters[adapter_name]
    
    def get_default_adapter(self, purpose: str) -> str:
        purpose_mapping = {
            "loader": self.default_loader,
            "storage": self.default_storage,
            "feature_store": self.default_feature_store,
        }
        if purpose not in purpose_mapping:
            raise ValueError(f"지원하지 않는 어댑터 목적: {purpose}")
        return purpose_mapping[purpose]

class RealtimeFeatureStoreConnectionSettings(BaseModel):
    """실시간 Feature Store 연결 설정"""
    host: str
    port: int
    db: int = 0

class RealtimeFeatureStoreSettings(BaseModel):
    """실시간 Feature Store 설정"""
    store_type: str
    connection: RealtimeFeatureStoreConnectionSettings

class ServingSettings(BaseModel):
    """API 서빙 설정"""
    enabled: bool = False
    model_stage: str
    realtime_feature_store: RealtimeFeatureStoreSettings

class PostgresStorageSettings(BaseModel):
    """PostgreSQL 저장 설정"""
    enabled: bool = False
    table_name: str = "batch_predictions"
    connection_uri: str

class ArtifactStoreSettings(BaseModel):
    """아티팩트 저장소 설정"""
    enabled: bool
    base_uri: str
    postgres_storage: Optional[PostgresStorageSettings] = None

class FeatureStoreSettings(BaseModel):
    """Feature Store 설정"""
    provider: str = "dynamic"
    feast_config: Optional[Dict[str, Any]] = None
    connection_timeout: int = 5000
    retry_attempts: int = 3
    connection_info: Dict[str, Any] = {}

class HyperparameterTuningSettings(BaseModel):
    """하이퍼파라미터 튜닝 설정 (인프라 제약)"""
    enabled: bool = False
    n_trials: int = 10
    metric: str = "accuracy"
    direction: str = "maximize"
    timeout: Optional[int] = None 