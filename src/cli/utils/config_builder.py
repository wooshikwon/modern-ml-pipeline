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
        
        # 데이터 소스 타입만 저장 (구체적 설정은 .env 파일에서)
        
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
            # Registry 위치 선택
            registry_location = self.ui.select_from_list(
                "Feast Registry 저장 위치",
                ["로컬", "S3", "GCS"]
            )
            selections["feast_registry_location"] = registry_location
            
            # Offline Store는 data_source에 따라 자동 결정
            self.ui.show_info(f"Offline Store는 {data_source}에 따라 자동 설정됩니다.")
            
            # Offline Store가 File인 경우
            if data_source in ["PostgreSQL", "Local Files", "S3", "GCS"]:
                self.ui.show_info("Offline Store는 Parquet 파일 형식을 사용합니다.")
                selections["feast_needs_offline_path"] = True
            
            # Online Store 설정
            use_online_store = self.ui.confirm(
                "Online Store를 사용하시겠습니까? (실시간 서빙용)",
                default=False
            )
            
            if use_online_store:
                online_store_type = self.ui.select_from_list(
                    "Online Store 타입",
                    ["Redis", "SQLite", "DynamoDB"]
                )
                selections["feast_online_store"] = online_store_type
            else:
                selections["feast_online_store"] = "SQLite"
        
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
            # Storage별 구체적 설정은 .env 파일에서
        
        self.ui.print_divider()
        
        # 6. Output targets 설정
        self.ui.show_info("Output 저장 설정")
        # Inference output
        infer_enabled = self.ui.confirm("배치 추론 결과를 저장하시겠습니까?", default=True)
        selections["inference_output_enabled"] = infer_enabled
        if infer_enabled:
            infer_source = self.ui.select_from_list(
                "추론 결과 저장 데이터 소스를 선택하세요",
                ["PostgreSQL", "BigQuery", "Local Files", "S3", "GCS"]
            )
            selections["inference_output_source"] = infer_source
        
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
        Inference Output: {selections.get('inference_output_source', 'Disabled' if not selections.get('inference_output_enabled', True) else 'Local Files')}
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
        
        # Feature Store 플래그
        feature_store = selections.get("feature_store", "없음")
        context["use_feast"] = feature_store == "Feast"
        
        # 기타 플래그
        context["enable_auth"] = False  # 기본값
        
        # 기본값 설정
        context.setdefault("serving_workers", 1)
        context.setdefault("model_stage", "None")
        
        # Output sources (템플릿 분기용)
        if selections.get("inference_output_enabled", True):
            context["inference_output_source"] = selections.get("inference_output_source", "Local Files")
        
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
                "MLFLOW_TRACKING_URI=./mlruns  # or http://mlflow-server:5000",
                f"MLFLOW_EXPERIMENT_NAME=mmp-{selections['env_name']}",
                "# Optional MLflow authentication",
                "MLFLOW_TRACKING_USERNAME=",
                "MLFLOW_TRACKING_PASSWORD=",
                "# Optional S3-compatible storage",
                "MLFLOW_S3_ENDPOINT_URL=",
                "",
            ])
        
        # 데이터 소스 설정
        data_source = selections.get("data_source", "")
        
        if data_source == "PostgreSQL":
            lines.extend([
                "# PostgreSQL Configuration",
                "DB_HOST=localhost",
                "DB_PORT=5432",
                "DB_NAME=mlpipeline",
                "DB_USER=your_username",
                "DB_PASSWORD=your_password",
                "DB_TIMEOUT=30",
                "",
            ])
        elif data_source == "BigQuery":
            lines.extend([
                "# BigQuery Configuration",
                "GCP_PROJECT_ID=your-project-id",
                "BQ_DATASET_ID=ml_dataset",
                "BQ_LOCATION=US",
                "BQ_TIMEOUT=30",
                "",
            ])
        elif data_source == "S3":
            lines.extend([
                "# S3 Configuration",
                "S3_BUCKET=your-data-bucket",
                f"S3_PREFIX={selections['env_name']}",
                "AWS_REGION=us-east-1",
                "AWS_ACCESS_KEY_ID=your_access_key",
                "AWS_SECRET_ACCESS_KEY=your_secret_key",
                "",
            ])
        elif data_source == "GCS":
            lines.extend([
                "# GCS Configuration",
                "GCP_PROJECT_ID=your-project-id",
                "GCS_BUCKET=your-data-bucket",
                f"GCS_PREFIX={selections['env_name']}",
                "GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json",
                "",
            ])
        elif data_source == "Local Files":
            lines.extend([
                "# Local Files Configuration",
                "DATA_PATH=./data",
                "",
            ])
        
        # Feature Store 설정
        feature_store = selections.get("feature_store", "없음")
        
        if feature_store == "Feast":
            lines.extend([
                "# Feast Configuration",
                f"FEAST_PROJECT=feast_{selections['env_name']}",
                "",
            ])
            
            # Registry 설정
            registry_location = selections.get("feast_registry_location", "로컬")
            if registry_location == "로컬":
                lines.extend([
                    "# Feast Registry (Local)",
                    "FEAST_REGISTRY_PATH=./feast_repo/registry.db",
                    "",
                ])
            elif registry_location == "S3":
                lines.extend([
                    "# Feast Registry (S3)",
                    f"FEAST_REGISTRY_PATH=s3://your-bucket/feast-registry/{selections['env_name']}/registry.db",
                    "",
                ])
            else:  # GCS
                lines.extend([
                    "# Feast Registry (GCS)",
                    f"FEAST_REGISTRY_PATH=gs://your-bucket/feast-registry/{selections['env_name']}/registry.db",
                    "",
                ])
            
            # Offline Store 설정 (File 타입인 경우)
            if selections.get("feast_needs_offline_path"):
                lines.extend([
                    "# Feast Offline Store (Parquet files)",
                    "FEAST_OFFLINE_PATH=./feast_repo/data",
                    "",
                ])
            
            # Online Store 설정
            online_store = selections.get("feast_online_store", "SQLite")
            if online_store == "Redis":
                lines.extend([
                    "# Feast Online Store (Redis)",
                    "REDIS_HOST=localhost",
                    "REDIS_PORT=6379",
                    "REDIS_PASSWORD=",  # Optional
                    "",
                ])
            elif online_store == "DynamoDB":
                lines.extend([
                    "# Feast Online Store (DynamoDB)",
                    "DYNAMODB_REGION=us-east-1",
                    "DYNAMODB_TABLE_NAME=feast-online-store",
                    "",
                ])
            else:  # SQLite
                lines.extend([
                    "# Feast Online Store (SQLite)",
                    "FEAST_ONLINE_STORE_PATH=./feast_repo/online_store.db",
                    "",
                ])
        
        # Artifact Storage 설정
        artifact_storage = selections.get("artifact_storage", "Local")
        
        if artifact_storage == "S3":
            lines.extend([
                "# MLflow Artifact Storage (S3)",
                "ARTIFACT_S3_BUCKET=mlflow-artifacts",
                f"ARTIFACT_S3_PREFIX={selections['env_name']}",
                "# Reuse AWS credentials from data source if same",
                "",
            ])
        elif artifact_storage == "GCS":
            lines.extend([
                "# MLflow Artifact Storage (GCS)",
                "ARTIFACT_GCS_BUCKET=mlflow-artifacts",
                f"ARTIFACT_GCS_PREFIX={selections['env_name']}",
                "# Reuse GCP credentials from data source if same",
                "",
            ])
        elif artifact_storage == "Local":
            lines.extend([
                "# MLflow Artifact Storage (Local)",
                "MLFLOW_ARTIFACT_PATH=./mlruns/artifacts",
                "",
            ])
        
        # API Serving 설정
        if selections.get("enable_serving"):
            lines.extend([
                "# API Serving Configuration",
                "API_HOST=0.0.0.0",
                "API_PORT=8000",
                "API_WORKERS=1",
                "",
            ])
        
        # Output: Inference
        infer_enabled = selections.get("inference_output_enabled", True)
        lines.extend([
            "# Inference Output",
            f"INFER_OUTPUT_ENABLED={'true' if infer_enabled else 'false'}",
        ])
        if infer_enabled:
            infer_src = selections.get("inference_output_source", "Local Files")
            if infer_src == "Local Files":
                lines.extend([
                    "INFER_OUTPUT_BASE_PATH=./artifacts/predictions",
                    "",
                ])
            elif infer_src == "S3":
                lines.extend([
                    "INFER_OUTPUT_S3_BUCKET=mmp-out",
                    f"INFER_OUTPUT_S3_PREFIX={selections['env_name']}/preds",
                    "# AWS credentials (if not already set above)",
                    "AWS_ACCESS_KEY_ID=",
                    "AWS_SECRET_ACCESS_KEY=",
                    "AWS_REGION=us-east-1",
                    "",
                ])
            elif infer_src == "GCS":
                lines.extend([
                    "INFER_OUTPUT_GCS_BUCKET=mmp-out",
                    f"INFER_OUTPUT_GCS_PREFIX={selections['env_name']}/preds",
                    "# GCP credentials (if not already set above)",
                    "GCP_PROJECT_ID=",
                    "GOOGLE_APPLICATION_CREDENTIALS=",
                    "",
                ])
            elif infer_src == "PostgreSQL":
                lines.extend([
                    f"INFER_OUTPUT_PG_TABLE=predictions_{selections['env_name']}",
                    "# Reuse DB_* settings above",
                    "",
                ])
            else:  # BigQuery
                lines.extend([
                    "INFER_OUTPUT_BQ_DATASET=analytics",
                    f"INFER_OUTPUT_BQ_TABLE=predictions_{selections['env_name']}",
                    "# Reuse GCP credentials above",
                    "BQ_LOCATION=US",
                    "",
                ])
        
        return "\n".join(lines)