"""
Config Builder for Modern ML Pipeline CLI
Phase 3: Interactive configuration file generation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 대화형 인터페이스
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.cli.utils.interactive_ui import InteractiveUI
from src.cli.utils.template_engine import TemplateEngine


class InteractiveConfigBuilder:
    """대화형 환경 설정 빌더.
    
    사용자와의 대화형 인터페이스를 통해 환경별 설정 파일을 생성합니다.
    """
    
    def __init__(self):
        """InteractiveConfigBuilder 초기화."""
        self.ui = InteractiveUI()
        templates_dir = Path(__file__).parent.parent / "templates"
        self.template_engine = TemplateEngine(templates_dir)
    
    def run_interactive_flow(self, env_name: Optional[str] = None) -> Dict[str, Any]:
        """
        대화형 설정 플로우 실행.
        
        Args:
            env_name: 환경 이름 (선택사항)
            
        Returns:
            사용자 선택 사항을 담은 딕셔너리
        """
        selections = {}
        
        # 1. 환경 이름 입력
        if not env_name:
            env_name = self.ui.text_input(
                "환경 이름을 입력하세요 (예: local, dev, prod)",
                default="local",
                validator=lambda x: len(x) > 0 and x.replace("-", "").replace("_", "").isalnum()
            )
        selections["env_name"] = env_name
        
        self.ui.print_divider()
        
        # 2. MLflow 사용 여부
        self.ui.show_info("MLflow 설정")
        use_mlflow = self.ui.confirm("MLflow를 사용하시겠습니까?", default=True)
        selections["use_mlflow"] = use_mlflow
        
        if use_mlflow:
            # MLflow 추가 설정
            mlflow_tracking = self.ui.text_input(
                "MLflow Tracking URI",
                default="./mlruns" if env_name == "local" else "http://mlflow-server:5000"
            )
            selections["mlflow_tracking_uri"] = mlflow_tracking
        
        self.ui.print_divider()
        
        # 3. 데이터 소스 선택
        self.ui.show_info("데이터 소스 설정")
        data_sources = [
            "PostgreSQL",
            "BigQuery", 
            "Local Files",
            "S3",
            "GCS"
        ]
        
        data_source = self.ui.select_from_list(
            "데이터를 로드할 소스를 선택하세요",
            data_sources
        )
        selections["data_source"] = data_source
        
        # 데이터 소스별 추가 설정
        if data_source == "PostgreSQL":
            selections["db_host"] = self.ui.text_input("Database Host", default="localhost")
            selections["db_port"] = self.ui.number_input("Database Port", default=5432)
            selections["db_name"] = self.ui.text_input("Database Name", default="mlpipeline")
        elif data_source == "BigQuery":
            selections["gcp_project"] = self.ui.text_input("GCP Project ID")
            selections["bq_dataset"] = self.ui.text_input("BigQuery Dataset", default="ml_dataset")
        elif data_source == "S3":
            selections["s3_bucket"] = self.ui.text_input("S3 Bucket Name")
            selections["aws_region"] = self.ui.text_input("AWS Region", default="us-east-1")
        elif data_source == "GCS":
            selections["gcp_project"] = self.ui.text_input("GCP Project ID")
            selections["gcs_bucket"] = self.ui.text_input("GCS Bucket Name")
        
        self.ui.print_divider()
        
        # 4. Feature Store 선택
        self.ui.show_info("Feature Store 설정")
        feature_stores = [
            "없음",
            "Feast"
        ]
        
        feature_store = self.ui.select_from_list(
            "Feature Store를 선택하세요",
            feature_stores
        )
        selections["feature_store"] = feature_store
        
        # Feature Store별 추가 설정
        if feature_store == "Feast":
            selections["feast_project"] = self.ui.text_input(
                "Feast Project Name",
                default=f"feast_{env_name}"
            )
            selections["feast_registry"] = self.ui.text_input(
                "Feast Registry Path",
                default="./feast_repo/registry.db"
            )
        
        self.ui.print_divider()
        
        # 5. Artifact Storage 선택
        self.ui.show_info("Artifact Storage 설정")
        
        if use_mlflow:
            storages = [
                "Local",
                "S3",
                "GCS"
            ]
            
            artifact_storage = self.ui.select_from_list(
                "MLflow Artifacts를 저장할 스토리지를 선택하세요",
                storages
            )
            selections["artifact_storage"] = artifact_storage
            
            # Storage별 추가 설정
            if artifact_storage == "S3":
                if "s3_bucket" not in selections:
                    selections["artifact_s3_bucket"] = self.ui.text_input(
                        "Artifact S3 Bucket",
                        default="mlflow-artifacts"
                    )
                    selections["artifact_aws_region"] = self.ui.text_input(
                        "AWS Region",
                        default="us-east-1"
                    )
            elif artifact_storage == "GCS":
                if "gcs_bucket" not in selections:
                    selections["artifact_gcs_bucket"] = self.ui.text_input(
                        "Artifact GCS Bucket",
                        default="mlflow-artifacts"
                    )
        
        self.ui.print_divider()
        
        # 6. 추가 설정
        self.ui.show_info("추가 설정")
        
        # Serving 설정
        enable_serving = self.ui.confirm("API Serving을 활성화하시겠습니까?", default=False)
        selections["enable_serving"] = enable_serving
        
        if enable_serving:
            selections["serving_port"] = self.ui.number_input(
                "API Serving Port",
                default=8000,
                min_value=1024,
                max_value=65535
            )
        
        # Hyperparameter Tuning 설정
        enable_tuning = self.ui.confirm(
            "Hyperparameter Tuning (Optuna)을 활성화하시겠습니까?",
            default=False
        )
        selections["enable_hyperparameter_tuning"] = enable_tuning
        
        if enable_tuning:
            selections["tuning_timeout"] = self.ui.number_input(
                "Tuning Timeout (seconds)",
                default=300,
                min_value=60,
                max_value=3600
            )
        
        # Monitoring 설정
        enable_monitoring = self.ui.confirm("Monitoring을 활성화하시겠습니까?", default=False)
        selections["enable_monitoring"] = enable_monitoring
        
        self.ui.print_divider()
        
        # 선택 사항 확인
        self._show_selections_summary(selections)
        
        if not self.ui.confirm("\n이 설정으로 진행하시겠습니까?", default=True):
            self.ui.show_warning("설정이 취소되었습니다. 다시 시작해주세요.")
            return self.run_interactive_flow(env_name)
        
        return selections
    
    def _show_selections_summary(self, selections: Dict[str, Any]) -> None:
        """
        선택 사항 요약 표시.
        
        Args:
            selections: 사용자 선택 사항
        """
        summary = f"""
환경 이름: {selections['env_name']}
MLflow 사용: {'예' if selections.get('use_mlflow') else '아니오'}
데이터 소스: {selections.get('data_source', 'N/A')}
Feature Store: {selections.get('feature_store', '없음')}
Artifact Storage: {selections.get('artifact_storage', 'Local')}
API Serving: {'활성화' if selections.get('enable_serving') else '비활성화'}
Hyperparameter Tuning: {'활성화' if selections.get('enable_hyperparameter_tuning') else '비활성화'}
Monitoring: {'활성화' if selections.get('enable_monitoring') else '비활성화'}
"""
        self.ui.show_panel(summary, title="📋 설정 요약", style="cyan")
    
    def generate_config_file(self, env_name: str, selections: Dict[str, Any]) -> Path:
        """
        설정 파일 생성.
        
        Args:
            env_name: 환경 이름
            selections: 사용자 선택 사항
            
        Returns:
            생성된 설정 파일 경로
        """
        # 템플릿 컨텍스트 준비
        context = self._prepare_template_context(selections)
        
        # 설정 파일 경로
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / f"{env_name}.yaml"
        
        # 템플릿 렌더링 및 파일 생성
        self.template_engine.write_rendered_file(
            "configs/config.yaml.j2",
            config_path,
            context
        )
        
        return config_path
    
    def generate_env_template(self, env_name: str, selections: Dict[str, Any]) -> Path:
        """
        환경 변수 템플릿 파일 생성.
        
        Args:
            env_name: 환경 이름
            selections: 사용자 선택 사항
            
        Returns:
            생성된 환경 변수 템플릿 파일 경로
        """
        env_template_path = Path(f".env.{env_name}.template")
        
        # 환경 변수 템플릿 내용 생성
        env_content = self._generate_env_template_content(selections)
        
        # 파일 쓰기
        env_template_path.write_text(env_content, encoding='utf-8')
        
        return env_template_path
    
    def _prepare_template_context(self, selections: Dict[str, Any]) -> Dict[str, Any]:
        """
        템플릿 렌더링을 위한 컨텍스트 준비.
        
        Args:
            selections: 사용자 선택 사항
            
        Returns:
            템플릿 컨텍스트
        """
        context = selections.copy()
        context["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 데이터 소스별 플래그 설정
        data_source = selections.get("data_source", "")
        context["use_sql"] = data_source == "PostgreSQL"
        context["use_bq"] = data_source == "BigQuery"
        context["use_s3"] = data_source == "S3"
        context["use_gcs"] = data_source == "GCS"
        context["use_storage"] = data_source == "Local Files"
        context["use_azure"] = "Azure" in data_source
        
        # Feature Store 플래그
        feature_store = selections.get("feature_store", "없음")
        context["use_feast"] = feature_store == "Feast"
        
        # Storage 플래그
        artifact_storage = selections.get("artifact_storage", "Local")
        context["use_local_storage"] = artifact_storage == "Local"
        context["use_s3"] = artifact_storage == "S3"
        context["use_gcs"] = artifact_storage == "GCS"
        
        # 기타 플래그
        context["use_redis"] = context.get("use_feast", False)  # Feast uses Redis for online store
        context["enable_auth"] = False  # 기본값
        context["enable_grafana"] = context.get("enable_monitoring", False)
        
        # 기본값 설정
        context.setdefault("serving_workers", 1)
        context.setdefault("model_stage", "None")
        context.setdefault("tuning_direction", "maximize")
        context.setdefault("tuning_jobs", 2)
        context.setdefault("prometheus_port", 9090)
        
        return context
    
    def _generate_env_template_content(self, selections: Dict[str, Any]) -> str:
        """
        환경 변수 템플릿 내용 생성.
        
        Args:
            selections: 사용자 선택 사항
            
        Returns:
            환경 변수 템플릿 내용
        """
        lines = [
            f"# Environment variables for {selections['env_name']}",
            f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # MLflow 설정
        if selections.get("use_mlflow"):
            lines.extend([
                "# MLflow Configuration",
                f"MLFLOW_TRACKING_URI={selections.get('mlflow_tracking_uri', './mlruns')}",
                f"MLFLOW_EXPERIMENT_NAME=mmp-{selections['env_name']}",
                "",
            ])
        
        # 데이터 소스 설정
        data_source = selections.get("data_source", "")
        
        if data_source == "PostgreSQL":
            lines.extend([
                "# PostgreSQL Configuration",
                f"DB_HOST={selections.get('db_host', 'localhost')}",
                f"DB_PORT={selections.get('db_port', 5432)}",
                f"DB_NAME={selections.get('db_name', 'mlpipeline')}",
                "DB_USER=your_username",
                "DB_PASSWORD=your_password",
                "",
            ])
        elif data_source == "BigQuery":
            lines.extend([
                "# BigQuery Configuration",
                f"GCP_PROJECT_ID={selections.get('gcp_project', '')}",
                f"BQ_DATASET_ID={selections.get('bq_dataset', 'ml_dataset')}",
                "BQ_LOCATION=US",
                "",
            ])
        elif data_source == "S3":
            lines.extend([
                "# S3 Configuration",
                f"S3_BUCKET={selections.get('s3_bucket', '')}",
                f"AWS_REGION={selections.get('aws_region', 'us-east-1')}",
                "AWS_ACCESS_KEY_ID=your_access_key",
                "AWS_SECRET_ACCESS_KEY=your_secret_key",
                "",
            ])
        elif data_source == "GCS":
            lines.extend([
                "# GCS Configuration",
                f"GCP_PROJECT_ID={selections.get('gcp_project', '')}",
                f"GCS_BUCKET={selections.get('gcs_bucket', '')}",
                "GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json",
                "",
            ])
        
        # Feature Store 설정
        feature_store = selections.get("feature_store", "없음")
        
        if feature_store == "Feast":
            feast_project = selections.get('feast_project', f"feast_{selections['env_name']}")
            lines.extend([
                "# Feast Configuration",
                f"FEAST_PROJECT={feast_project}",
                f"FEAST_REGISTRY_PATH={selections.get('feast_registry', './feast_repo/registry.db')}",
                "REDIS_HOST=localhost",
                "REDIS_PORT=6379",
                "",
            ])
        
        # Artifact Storage 설정
        artifact_storage = selections.get("artifact_storage", "Local")
        
        if artifact_storage == "S3" and "artifact_s3_bucket" in selections:
            lines.extend([
                "# Artifact Storage (S3)",
                f"ARTIFACT_S3_BUCKET={selections.get('artifact_s3_bucket', 'mlflow-artifacts')}",
                f"ARTIFACT_AWS_REGION={selections.get('artifact_aws_region', 'us-east-1')}",
                "",
            ])
        elif artifact_storage == "GCS" and "artifact_gcs_bucket" in selections:
            lines.extend([
                "# Artifact Storage (GCS)",
                f"ARTIFACT_GCS_BUCKET={selections.get('artifact_gcs_bucket', 'mlflow-artifacts')}",
                "",
            ])

        # API Serving 설정
        if selections.get("enable_serving"):
            lines.extend([
                "# API Serving Configuration",
                "API_HOST=0.0.0.0",
                f"API_PORT={selections.get('serving_port', 8000)}",
                "API_WORKERS=1",
                "",
            ])
        
        # Hyperparameter Tuning 설정
        if selections.get("enable_hyperparameter_tuning"):
            lines.extend([
                "# Hyperparameter Tuning Configuration (Optuna)",
                f"HYPERPARAM_TIMEOUT={selections.get('tuning_timeout', 300)}",
                "HYPERPARAM_JOBS=2",
                f"OPTUNA_STUDY_NAME=mmp_{selections['env_name']}_study",
                "OPTUNA_STORAGE=sqlite:///optuna.db",
            ])
            
            lines.append("")
        
        # Monitoring 설정
        if selections.get("enable_monitoring"):
            lines.extend([
                "# Monitoring Configuration",
                f"METRICS_PORT={selections.get('prometheus_port', 9090)}",
                "GRAFANA_HOST=localhost",
                "GRAFANA_PORT=3000",
                "COLLECT_SYSTEM_METRICS=true",
                "COLLECT_MODEL_METRICS=true",
                "COLLECT_DATA_METRICS=true",
                "",
            ])
        
        return "\n".join(lines)