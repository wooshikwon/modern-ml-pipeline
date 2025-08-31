# src/settings/_config_schema.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import requests
import os
import subprocess
from src.utils.system.logger import logger

class EnvironmentSettings(BaseModel):
    """환경별 기본 설정 (v2.0)"""
    # app_env field removed in v2.0 - use ENV_NAME environment variable
    gcp_project_id: str
    gcp_credential_path: Optional[str] = None
    
    @property
    def env_name(self) -> str:
        """현재 환경 이름 (ENV_NAME 환경변수에서 가져옴)"""
        return os.environ.get('ENV_NAME', 'local')

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

    @classmethod
    def with_ui_launch(
        cls,
        tracking_uri: str,
        experiment_name: str,
        auto_launch_ui: bool = False,
        ui_port: int = 5000
    ) -> 'MlflowSettings':
        """
        로컬 MLflow UI 자동 실행 지원
        
        로컬 파일 모드시 백그라운드에서 MLflow UI를 자동으로 실행
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: 실험명
            auto_launch_ui: UI 자동 실행 여부
            ui_port: MLflow UI 포트 (기본: 5000)
            
        Returns:
            MlflowSettings: 설정 객체 (UI 실행은 백그라운드에서)
        """
        # 설정 객체 생성
        settings = cls(tracking_uri=tracking_uri, experiment_name=experiment_name)
        
        # UI 자동 실행 조건: 활성화 + 로컬 파일 모드
        if auto_launch_ui and cls._is_local_file_uri(tracking_uri):
            try:
                # 백그라운드에서 MLflow UI 실행
                logger.info(f"로컬 파일 모드 감지, MLflow UI 자동 실행 중... (포트: {ui_port})")
                subprocess.Popen(
                    ['mlflow', 'ui', '--backend-store-uri', tracking_uri, '--port', str(ui_port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True  # 독립적인 프로세스 그룹으로 실행
                )
                logger.info(f"MLflow UI가 백그라운드에서 실행됨: http://localhost:{ui_port}")
            except (OSError, subprocess.SubprocessError) as e:
                logger.warning(f"MLflow UI 자동 실행 실패: {e}")
                logger.warning("mlflow 명령이 설치되어 있는지 확인하세요")
            except Exception as e:
                logger.error(f"MLflow UI 실행 중 예상치 못한 오류: {e}")
        
        return settings
    
    @staticmethod
    def _is_local_file_uri(tracking_uri: str) -> bool:
        """URI가 로컬 파일 경로인지 확인"""
        return not tracking_uri.startswith(('http://', 'https://', 'ftp://', 'sftp://'))

    @classmethod
    def create_with_friendly_messages(
        cls,
        tracking_uri: str,
        experiment_name: str,
        auto_launch_ui: bool = False,
        ui_port: int = 5000
    ) -> 'MlflowSettings':
        """
        사용자 친화적 메시지와 함께 MLflow 설정 생성
        
        모드별로 적절한 안내 메시지를 출력하여 사용자가 현재 상태를 쉽게 이해할 수 있도록 지원
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: 실험명
            auto_launch_ui: UI 자동 실행 여부 (로컬 모드만)
            ui_port: MLflow UI 포트 (기본: 5000)
            
        Returns:
            MlflowSettings: 친화적 메시지가 출력된 설정 객체
        """
        # UI 자동 실행 포함해서 설정 생성
        if auto_launch_ui and cls._is_local_file_uri(tracking_uri):
            settings = cls.with_ui_launch(
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
                auto_launch_ui=auto_launch_ui,
                ui_port=ui_port
            )
        else:
            settings = cls(tracking_uri=tracking_uri, experiment_name=experiment_name)
        
        # 모드별 친화적 메시지 출력
        cls._print_mode_info(tracking_uri, ui_port if auto_launch_ui else None)
        
        return settings

    @classmethod
    def create_with_friendly_messages_fallback(
        cls,
        server_uri: str,
        experiment_name: str,
        fallback_uri: Optional[str] = None,
        timeout: int = 5
    ) -> 'MlflowSettings':
        """
        폴백 시나리오에서 친화적 메시지와 함께 설정 생성
        
        서버 연결 실패 시 전환 과정을 사용자에게 명확히 안내
        
        Args:
            server_uri: MLflow 서버 URI
            experiment_name: 실험명
            fallback_uri: 폴백 URI (기본: "./mlruns")
            timeout: 서버 연결 타임아웃 (초)
            
        Returns:
            MlflowSettings: 폴백 과정이 안내된 설정 객체
        """
        # 기존 fallback 로직 실행
        settings = cls.with_fallback(
            server_uri=server_uri,
            experiment_name=experiment_name,
            fallback_uri=fallback_uri,
            timeout=timeout
        )
        
        # 폴백이 발생했는지 확인
        actual_fallback = fallback_uri or "./mlruns"
        if settings.tracking_uri == actual_fallback:
            # 폴백 발생시 추가 안내
            logger.warning("⚠️  MLflow 서버에 연결할 수 없습니다")
            logger.info("🔄 로컬 파일 모드로 자동 전환되었습니다")
            cls._print_mode_info(settings.tracking_uri)
        else:
            # 서버 연결 성공시 서버 모드 안내  
            cls._print_mode_info(settings.tracking_uri)
        
        return settings

    @classmethod
    def create_with_friendly_messages_auto_detect(
        cls,
        experiment_name: str,
        fallback_uri: Optional[str] = None,
        timeout: int = 5
    ) -> 'MlflowSettings':
        """
        환경변수 자동 감지와 함께 친화적 메시지 출력
        
        감지 과정을 사용자에게 단계별로 안내
        
        Args:
            experiment_name: 실험명
            fallback_uri: 폴백 URI (기본: "./mlruns")
            timeout: 서버 연결 타임아웃 (초)
            
        Returns:
            MlflowSettings: 자동 감지 과정이 안내된 설정 객체
        """
        logger.info("🔍 MLflow 환경 자동 감지 중...")
        
        # 환경변수 확인
        server_uri = os.getenv('MLFLOW_TRACKING_URI')
        if not server_uri or server_uri.strip() == '':
            logger.info("📝 MLFLOW_TRACKING_URI 환경변수가 설정되지 않음")
            actual_fallback = fallback_uri or "./mlruns"
            settings = cls(tracking_uri=actual_fallback, experiment_name=experiment_name)
        else:
            logger.info(f"📝 MLFLOW_TRACKING_URI 환경변수 감지: {server_uri}")
            settings = cls.with_fallback(
                server_uri=server_uri,
                experiment_name=experiment_name,
                fallback_uri=fallback_uri,
                timeout=timeout
            )
        
        # 최종 모드 안내
        cls._print_mode_info(settings.tracking_uri)
        
        return settings

    @staticmethod
    def _print_mode_info(tracking_uri: str, ui_port: Optional[int] = None) -> None:
        """
        현재 MLflow 모드 정보를 사용자 친화적으로 출력
        
        Args:
            tracking_uri: 현재 tracking URI
            ui_port: UI 포트 (자동 실행된 경우)
        """
        if MlflowSettings._is_local_file_uri(tracking_uri):
            # 로컬 파일 모드
            logger.info("🎯 MLflow 실행 모드: 로컬 파일 모드")
            logger.info(f"📁 데이터 저장소: {tracking_uri}")
            
            if ui_port:
                # UI 자동 실행됨
                logger.info(f"🚀 MLflow UI 자동 실행됨: http://localhost:{ui_port}")
                logger.info("💻 브라우저에서 위 주소로 접속하여 실험을 확인하세요")
            else:
                # 수동 UI 실행 안내
                logger.info("💡 mlflow ui 명령으로 웹 UI를 실행할 수 있습니다")
        else:
            # 서버 모드
            logger.info("🎯 MLflow 실행 모드: 서버 모드")
            logger.info(f"📊 MLflow 서버: {tracking_uri}")
            logger.info(f"🌐 웹 UI: {tracking_uri}")
            logger.info("💡 서버가 실행 중인지 확인하세요")

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