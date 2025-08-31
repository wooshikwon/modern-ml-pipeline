"""
Environment Config Generator for Dynamic File Generation

대화형 UI에서 수집한 사용자 선택을 바탕으로
환경별 Config YAML 파일과 환경변수 템플릿을 동적 생성.
기존 Jinja2 템플릿 방식을 완전 대체.

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 동적 생성으로 유연성 극대화
"""

from typing import Dict, Any
from src.cli.utils.config_management import EnvironmentConfig, DataAdapterConfig


class EnvironmentConfigGenerator:
    """
    사용자 선택을 기반으로 환경별 Config 파일 동적 생성
    
    Interactive Environment Builder에서 수집한 설정을
    실제 YAML 파일과 환경변수 템플릿으로 변환.
    
    생성 파일:
    - configs/{env_name}.yaml: 환경별 완전한 설정
    - .env.{env_name}.template: 환경변수 템플릿
    
    Examples:
        generator = EnvironmentConfigGenerator()
        yaml_content = generator.generate_config_yaml(env_config)
        env_template = generator.generate_env_template(env_config)
    """
    
    def generate_config_yaml(self, env_config: EnvironmentConfig) -> str:
        """
        환경 설정을 YAML 형식으로 변환
        
        Args:
            env_config: InteractiveEnvironmentBuilder로 생성된 환경 설정
            
        Returns:
            str: 완전한 환경별 YAML 설정 내용
            
        Note:
            기존 base.yaml + env.yaml 병합 방식을 대체하여
            단일 파일에 모든 설정 포함
        """
        
        yaml_lines = []
        
        # 파일 헤더
        yaml_lines.extend([
            f"# {env_config.name} 환경 설정",
            f"# 생성일: {env_config.created_at}",
            f"# Modern ML Pipeline Config System v2.0 (Recipe-Config 분리)",
            "",
            "# 환경 기본 정보",
            "environment:",
            f"  name: \"{env_config.name}\"",
            ""
        ])
        
        # ML Tracking 설정
        yaml_lines.extend(self._generate_ml_tracking_yaml(env_config.ml_tracking))
        
        # Data Adapters 설정 (Recipe-Config 분리의 핵심)
        yaml_lines.extend(self._generate_data_adapters_yaml(env_config.data_adapters))
        
        # Feature Store 설정
        yaml_lines.extend(self._generate_feature_store_yaml(env_config.feature_store))
        
        # Storage 설정
        yaml_lines.extend(self._generate_storage_yaml(env_config.storage))
        
        return "\n".join(yaml_lines)
    
    def generate_env_template(self, env_config: EnvironmentConfig) -> str:
        """
        환경변수 템플릿 생성
        
        Args:
            env_config: 환경 설정 객체
            
        Returns:
            str: .env.{env_name}.template 파일 내용
            
        Note:
            사용자가 실제 값으로 채워야 하는 환경변수 템플릿 생성.
            예시 값과 설명 주석 포함.
        """
        
        env_lines = []
        
        # 파일 헤더
        env_lines.extend([
            f"# {env_config.name} 환경 변수 템플릿",
            f"# 생성일: {env_config.created_at}",
            f"# Modern ML Pipeline v2.0 - Recipe-Config 분리 아키텍처",
            "",
            "# 이 파일을 복사하여 .env.{env_name} 파일을 생성하고",
            "# 실제 환경에 맞는 값들로 수정하세요:",
            f"# cp .env.{env_config.name}.template .env.{env_config.name}",
            "",
            "# 기본 환경 설정",
            f"PROJECT_NAME=my-ml-project",
            f"EXPERIMENT_NAME={env_config.name}-experiments",
            ""
        ])
        
        # ML Tracking 환경변수
        env_lines.extend(self._generate_ml_tracking_env(env_config.ml_tracking))
        
        # Database 환경변수
        env_lines.extend(self._generate_database_env(env_config.data_adapters))
        
        # Feature Store 환경변수
        env_lines.extend(self._generate_feature_store_env(env_config.feature_store))
        
        # Storage 환경변수
        env_lines.extend(self._generate_storage_env(env_config.storage))
        
        return "\n".join(env_lines)
    
    def _generate_ml_tracking_yaml(self, ml_tracking: Dict[str, Any]) -> list[str]:
        """ML Tracking 설정 YAML 생성"""
        
        lines = ["# ML 실험 추적 설정", "ml_tracking:"]
        
        provider = ml_tracking.get("provider", "none")
        
        if provider == "none":
            lines.extend([
                "  provider: \"none\"",
                "  # 실험 추적 비활성화",
                ""
            ])
        else:
            lines.append(f"  provider: \"{provider}\"")
            
            if provider in ["mlflow_local", "mlflow_server"]:
                tracking_uri = ml_tracking.get("tracking_uri", "./mlruns")
                experiment_name = ml_tracking.get("experiment_name", "default-experiment")
                
                lines.extend([
                    f"  tracking_uri: \"{tracking_uri}\"",
                    f"  experiment_name: \"{experiment_name}\"",
                ])
                
                if provider == "mlflow_local":
                    lines.append("  # 로컬 MLflow 서버 (./mlruns 디렉토리)")
                else:
                    lines.append("  # 원격 MLflow 추적 서버")
            
            lines.append("")
        
        return lines
    
    def _generate_data_adapters_yaml(self, data_adapters: Dict[str, DataAdapterConfig]) -> list[str]:
        """Data Adapters 설정 YAML 생성 (Recipe-Config 분리의 핵심)"""
        
        lines = [
            "# 데이터 어댑터 설정 (Recipe-Config 분리 핵심)",
            "# Recipe에서는 adapter_type만 지정하고, 실제 연결 정보는 여기서 관리",
            "data_adapters:"
        ]
        
        if not data_adapters:
            lines.extend([
                "  # 설정된 데이터 어댑터가 없습니다",
                ""
            ])
            return lines
        
        for adapter_name, adapter_config in data_adapters.items():
            lines.append(f"  {adapter_name}:")
            lines.append(f"    type: \"{adapter_config.type}\"")
            
            if adapter_config.connection_params:
                lines.append("    connection:")
                for key, value in adapter_config.connection_params.items():
                    if isinstance(value, str):
                        lines.append(f"      {key}: \"{value}\"")
                    else:
                        lines.append(f"      {key}: {value}")
            
            # 어댑터별 설명 추가
            if adapter_name == "sql" and adapter_config.type != "none":
                lines.append(f"    # SQL 데이터 소스 ({adapter_config.type})")
            elif adapter_name == "storage":
                lines.append(f"    # 스토리지 어댑터 ({adapter_config.type})")
        
        lines.append("")
        return lines
    
    def _generate_feature_store_yaml(self, feature_store: Dict[str, Any]) -> list[str]:
        """Feature Store 설정 YAML 생성"""
        
        lines = ["# Feature Store 설정 (Point-in-Time 조인)", "feature_store:"]
        
        provider = feature_store.get("provider", "none")
        
        if provider == "none":
            lines.extend([
                "  provider: \"none\"",
                "  # Feature Store 사용하지 않음",
                ""
            ])
        else:
            lines.append(f"  provider: \"{provider}\"")
            
            if provider == "redis":
                connection_url = feature_store.get("connection_url", "redis://localhost:6379/0")
                namespace = feature_store.get("namespace", "ml_features")
                
                lines.extend([
                    f"  connection_url: \"{connection_url}\"",
                    f"  namespace: \"{namespace}\"",
                    "  # Redis 기반 Feature Store"
                ])
            
            lines.append("")
        
        return lines
    
    def _generate_storage_yaml(self, storage: Dict[str, Any]) -> list[str]:
        """Storage 설정 YAML 생성"""
        
        lines = ["# 아티팩트 저장소 설정", "storage:"]
        
        provider = storage.get("provider", "local")
        lines.append(f"  provider: \"{provider}\"")
        
        if provider == "s3":
            lines.extend([
                f"  bucket: \"{storage.get('bucket', 'ml-pipeline-artifacts')}\"",
                f"  region: \"{storage.get('region', 'us-west-2')}\"",
                "  # AWS S3 클라우드 스토리지"
            ])
        elif provider == "gcs":
            lines.extend([
                f"  bucket: \"{storage.get('bucket', 'ml-pipeline-artifacts')}\"",
                f"  credentials: \"{storage.get('credentials', './credentials/gcs.json')}\"",
                "  # Google Cloud Storage"
            ])
        else:  # local
            lines.extend([
                f"  base_path: \"{storage.get('base_path', './artifacts')}\"",
                "  # 로컬 파일 시스템 저장소"
            ])
        
        lines.append("")
        return lines
    
    def _generate_ml_tracking_env(self, ml_tracking: Dict[str, Any]) -> list[str]:
        """ML Tracking 환경변수 생성"""
        
        provider = ml_tracking.get("provider", "none")
        
        if provider == "none":
            return []
        
        lines = ["# ML Tracking (MLflow) 설정"]
        
        if provider == "mlflow_server":
            lines.extend([
                "# MLflow 서버 URL (예: http://localhost:5000)",
                "MLFLOW_TRACKING_URI=http://localhost:5000",
                ""
            ])
        elif provider == "mlflow_local":
            lines.extend([
                "# MLflow 로컬 설정 (기본값 사용시 주석 처리 가능)",
                "# MLFLOW_TRACKING_URI=./mlruns",
                ""
            ])
        
        lines.extend([
            f"# 실험 이름 (기본: {ml_tracking.get('experiment_name', 'default-experiment')})",
            f"EXPERIMENT_NAME={ml_tracking.get('experiment_name', 'default-experiment')}",
            ""
        ])
        
        return lines
    
    def _generate_database_env(self, data_adapters: Dict[str, DataAdapterConfig]) -> list[str]:
        """Database 환경변수 생성"""
        
        sql_adapter = data_adapters.get("sql")
        if not sql_adapter or sql_adapter.type == "none":
            return []
        
        lines = [f"# Database ({sql_adapter.type.upper()}) 설정"]
        
        if sql_adapter.type in ["postgresql", "mysql"]:
            default_port = 5432 if sql_adapter.type == "postgresql" else 3306
            lines.extend([
                f"# {sql_adapter.type.upper()} 연결 정보",
                "DATABASE_HOST=localhost",
                f"DATABASE_PORT={default_port}",
                "DATABASE_NAME=mlpipeline",
                "DATABASE_USER=user",
                "DATABASE_PASSWORD=password",
                ""
            ])
        elif sql_adapter.type == "bigquery":
            lines.extend([
                "# BigQuery 설정",
                "BIGQUERY_PROJECT_ID=your-project-id",
                "BIGQUERY_DATASET=ml_pipeline",
                "GOOGLE_APPLICATION_CREDENTIALS=./credentials/bigquery-service-account.json",
                ""
            ])
        
        return lines
    
    def _generate_feature_store_env(self, feature_store: Dict[str, Any]) -> list[str]:
        """Feature Store 환경변수 생성"""
        
        provider = feature_store.get("provider", "none")
        
        if provider == "none":
            return []
        
        lines = [f"# Feature Store ({provider.upper()}) 설정"]
        
        if provider == "redis":
            lines.extend([
                "# Redis Feature Store 연결 정보",
                "REDIS_URL=redis://localhost:6379/0",
                "FEATURE_STORE_NAMESPACE=ml_features",
                ""
            ])
        
        return lines
    
    def _generate_storage_env(self, storage: Dict[str, Any]) -> list[str]:
        """Storage 환경변수 생성"""
        
        provider = storage.get("provider", "local")
        
        lines = [f"# Storage ({provider.upper()}) 설정"]
        
        if provider == "s3":
            lines.extend([
                "# AWS S3 스토리지 설정",
                "S3_BUCKET=ml-pipeline-artifacts",
                "AWS_REGION=us-west-2",
                "AWS_ACCESS_KEY_ID=your-access-key",
                "AWS_SECRET_ACCESS_KEY=your-secret-key",
                ""
            ])
        elif provider == "gcs":
            lines.extend([
                "# Google Cloud Storage 설정",
                "GCS_BUCKET=ml-pipeline-artifacts",
                "GOOGLE_APPLICATION_CREDENTIALS=./credentials/gcs-service-account.json",
                ""
            ])
        else:  # local
            lines.extend([
                "# 로컬 스토리지 설정",
                "STORAGE_BASE_PATH=./artifacts",
                ""
            ])
        
        return lines