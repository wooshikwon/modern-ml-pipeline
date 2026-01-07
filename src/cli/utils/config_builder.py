"""
Config Builder for Modern ML Pipeline CLI

대화형 인터페이스를 통해 환경별 설정 파일(configs/*.yaml)을 생성합니다.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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
        total_steps = 5

        # Step 1: 환경 이름 입력
        self.ui.show_step(1, total_steps, "환경 이름")
        if not env_name:
            env_name = self.ui.text_input(
                "환경 이름을 입력하세요 (예: local, dev, prod)",
                default="local",
                validator=lambda x: len(x) > 0 and x.replace("-", "").replace("_", "").isalnum(),
            )
        selections["env_name"] = env_name

        # Step 2: MLflow 설정
        self.ui.show_step(2, total_steps, "MLflow 설정")
        use_mlflow = self.ui.confirm("MLflow를 사용하시겠습니까?", default=True)
        selections["use_mlflow"] = use_mlflow

        if use_mlflow:
            # MLflow 추가 설정
            mlflow_tracking = self.ui.text_input(
                "MLflow Tracking URI",
                default="./mlruns" if env_name == "local" else "http://mlflow-server:5000",
            )
            selections["mlflow_tracking_uri"] = mlflow_tracking

        # Step 3: 데이터 소스 선택
        self.ui.show_step(3, total_steps, "데이터 소스")
        data_sources = ["PostgreSQL", "BigQuery", "Local Files", "S3", "GCS"]

        data_source = self.ui.select_from_list(
            "데이터를 로드할 소스를 선택하세요", data_sources, allow_cancel=False
        )
        selections["data_source"] = data_source

        # Step 4: Feature Store 선택
        self.ui.show_step(4, total_steps, "Feature Store")
        feature_stores = ["없음", "Feast"]

        feature_store = self.ui.select_from_list(
            "Feature Store를 선택하세요", feature_stores, allow_cancel=False
        )
        selections["feature_store"] = feature_store

        # Feature Store별 추가 설정
        if feature_store == "Feast":
            # Registry 위치 선택
            registry_location = self.ui.select_from_list(
                "Feast Registry 저장 위치", ["로컬", "S3", "GCS"], allow_cancel=False
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
                "Online Store를 사용하시겠습니까? (실시간 서빙용)", default=False
            )

            if use_online_store:
                online_store_type = self.ui.select_from_list(
                    "Online Store 타입", ["Redis", "SQLite", "DynamoDB"], allow_cancel=False
                )
                selections["feast_online_store"] = online_store_type
            else:
                selections["feast_online_store"] = "SQLite"

        # Step 5: 배치 추론 결과 저장 설정
        self.ui.show_step(5, total_steps, "배치 추론 출력")
        infer_enabled = self.ui.confirm("배치 추론 결과를 저장하시겠습니까?", default=True)
        selections["inference_output_enabled"] = infer_enabled
        if infer_enabled:
            infer_source = self.ui.select_from_list(
                "추론 결과 저장 데이터 소스를 선택하세요",
                ["PostgreSQL", "BigQuery", "Local Files", "S3", "GCS"],
                allow_cancel=False,
            )
            selections["inference_output_source"] = infer_source

            # 저장소별 추가 설정
            if infer_source == "S3":
                selections["infer_s3_bucket"] = self.ui.text_input(
                    "S3 Bucket 이름", default="mmp-predictions"
                )
                selections["infer_s3_prefix"] = self.ui.text_input(
                    "S3 Prefix (경로)", default=f"{env_name}/predictions"
                )
            elif infer_source == "GCS":
                selections["infer_gcs_bucket"] = self.ui.text_input(
                    "GCS Bucket 이름", default="mmp-predictions"
                )
                selections["infer_gcs_prefix"] = self.ui.text_input(
                    "GCS Prefix (경로)", default=f"{env_name}/predictions"
                )
            elif infer_source == "BigQuery":
                selections["infer_bq_dataset"] = self.ui.text_input(
                    "BigQuery Dataset", default="ml_predictions"
                )
                selections["infer_bq_table"] = self.ui.text_input(
                    "BigQuery Table", default=f"predictions_{env_name}"
                )
                selections["infer_bq_location"] = self.ui.text_input(
                    "BigQuery Location", default="US"
                )
            elif infer_source == "PostgreSQL":
                selections["infer_pg_table"] = self.ui.text_input(
                    "PostgreSQL Table 이름", default=f"predictions_{env_name}"
                )
            elif infer_source == "Local Files":
                selections["infer_local_path"] = self.ui.text_input(
                    "저장 경로", default="./artifacts/predictions"
                )

        # 최종 확인
        infer_output_display = self._format_inference_output_display(selections)
        summary_data = {
            "환경 이름": selections["env_name"],
            "MLflow 사용": "예" if selections.get("use_mlflow") else "아니오",
            "데이터 소스": selections.get("data_source", "N/A"),
            "Feature Store": selections.get("feature_store", "없음"),
            "Inference Output": infer_output_display,
        }

        if not self.ui.show_summary_and_confirm(summary_data, title="설정 요약"):
            self.ui.show_warning("설정이 취소되었습니다. 다시 시작해주세요.")
            return self.run_interactive_flow(env_name)

        return selections

    def _get_inference_output_display(self, selections: Dict[str, Any]) -> str:
        """Inference output 표시 문자열 반환."""
        if not selections.get("inference_output_enabled", True):
            return "Disabled"
        return selections.get("inference_output_source", "Local Files")

    def _format_inference_output_display(self, selections: Dict[str, Any]) -> str:
        """Inference output 상세 표시 문자열 반환."""
        if not selections.get("inference_output_enabled", True):
            return "Disabled"

        source = selections.get("inference_output_source", "Local Files")

        if source == "S3":
            bucket = selections.get("infer_s3_bucket", "")
            prefix = selections.get("infer_s3_prefix", "")
            return f"S3 (s3://{bucket}/{prefix})"
        elif source == "GCS":
            bucket = selections.get("infer_gcs_bucket", "")
            prefix = selections.get("infer_gcs_prefix", "")
            return f"GCS (gs://{bucket}/{prefix})"
        elif source == "BigQuery":
            dataset = selections.get("infer_bq_dataset", "")
            table = selections.get("infer_bq_table", "")
            location = selections.get("infer_bq_location", "US")
            return f"BigQuery ({dataset}.{table} @ {location})"
        elif source == "PostgreSQL":
            table = selections.get("infer_pg_table", "")
            return f"PostgreSQL ({table})"
        elif source == "Local Files":
            path = selections.get("infer_local_path", "./artifacts/predictions")
            return f"Local ({path})"

        return source

    def _show_selections_summary(self, selections: Dict[str, Any]) -> None:
        """
        선택 사항 요약 표시.

        Args:
            selections: 사용자 선택 사항
        """
        summary = f"""환경 이름: {selections['env_name']}
MLflow 사용: {'예' if selections.get('use_mlflow') else '아니오'}
데이터 소스: {selections.get('data_source', 'N/A')}
Feature Store: {selections.get('feature_store', '없음')}
Inference Output: {self._format_inference_output_display(selections)}"""

        self.ui.show_panel(summary, title="설정 요약")

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
        self.template_engine.write_rendered_file("configs/config.yaml.j2", config_path, context)

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
        env_template_path.write_text(env_content, encoding="utf-8")

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
            context["inference_output_source"] = selections.get(
                "inference_output_source", "Local Files"
            )

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

        # 필요한 인증 타입 수집
        needs_aws = False
        needs_gcp = False
        needs_postgresql = False

        data_source = selections.get("data_source", "")
        feature_store = selections.get("feature_store", "없음")
        feast_registry = selections.get("feast_registry_location", "로컬")
        feast_online = selections.get("feast_online_store", "SQLite")
        infer_src = selections.get("inference_output_source", "Local Files")

        # AWS 필요 여부 확인
        if data_source == "S3":
            needs_aws = True
        if feature_store == "Feast" and feast_registry == "S3":
            needs_aws = True
        if feature_store == "Feast" and feast_online == "DynamoDB":
            needs_aws = True
        if infer_src == "S3":
            needs_aws = True

        # GCP 필요 여부 확인
        if data_source in ["BigQuery", "GCS"]:
            needs_gcp = True
        if feature_store == "Feast" and feast_registry == "GCS":
            needs_gcp = True
        if infer_src in ["GCS", "BigQuery"]:
            needs_gcp = True

        # PostgreSQL 필요 여부 확인
        if data_source == "PostgreSQL":
            needs_postgresql = True
        if infer_src == "PostgreSQL":
            needs_postgresql = True

        # MLflow 설정
        if selections.get("use_mlflow"):
            mlflow_uri = selections.get("mlflow_tracking_uri", "./mlruns")
            is_remote_mlflow = mlflow_uri.startswith("http://") or mlflow_uri.startswith("https://")

            lines.extend(
                [
                    "# MLflow Configuration",
                    f"MLFLOW_TRACKING_URI={mlflow_uri}",
                    f"MLFLOW_EXPERIMENT_NAME=mmp-{selections['env_name']}",
                ]
            )

            if is_remote_mlflow:
                lines.extend(
                    [
                        "# MLflow Authentication (required for remote server)",
                        "MLFLOW_TRACKING_USERNAME=",
                        "MLFLOW_TRACKING_PASSWORD=",
                    ]
                )

            lines.append("")

        # 클라우드 인증 정보 (통합 생성)
        if needs_aws:
            lines.extend(
                [
                    "# AWS Configuration",
                    "AWS_ACCESS_KEY_ID=your_access_key",
                    "AWS_SECRET_ACCESS_KEY=your_secret_key",
                    "AWS_REGION=us-east-1",
                    "",
                ]
            )

        if needs_gcp:
            lines.extend(
                [
                    "# GCP Configuration",
                    "GCP_PROJECT_ID=your-project-id",
                    "GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json",
                    "",
                ]
            )

        if needs_postgresql:
            lines.extend(
                [
                    "# PostgreSQL Configuration",
                    "DB_HOST=localhost",
                    "DB_PORT=5432",
                    "DB_NAME=mlpipeline",
                    "DB_USER=your_username",
                    "DB_PASSWORD=your_password",
                    "DB_TIMEOUT=30",
                    "",
                ]
            )

        # 데이터 소스별 추가 설정
        if data_source == "S3":
            lines.extend(
                [
                    "# S3 Data Source",
                    "S3_BUCKET=your-data-bucket",
                    f"S3_PREFIX={selections['env_name']}",
                    "",
                ]
            )
        elif data_source == "GCS":
            lines.extend(
                [
                    "# GCS Data Source",
                    "GCS_BUCKET=your-data-bucket",
                    f"GCS_PREFIX={selections['env_name']}",
                    "",
                ]
            )
        elif data_source == "Local Files":
            lines.extend(
                [
                    "# Local Files Configuration",
                    "DATA_PATH=./data",
                    "",
                ]
            )

        # Feature Store 설정
        if feature_store == "Feast":
            lines.extend(
                [
                    "# Feast Configuration",
                    f"FEAST_PROJECT=feast_{selections['env_name']}",
                ]
            )

            if feast_registry == "로컬":
                lines.append("FEAST_REGISTRY_PATH=./feast_repo/registry.db")
            elif feast_registry == "S3":
                lines.append(
                    f"FEAST_REGISTRY_PATH=s3://your-bucket/feast-registry/{selections['env_name']}/registry.db"
                )
            else:  # GCS
                lines.append(
                    f"FEAST_REGISTRY_PATH=gs://your-bucket/feast-registry/{selections['env_name']}/registry.db"
                )

            lines.append("")

            if selections.get("feast_needs_offline_path"):
                lines.extend(
                    [
                        "# Feast Offline Store",
                        "FEAST_OFFLINE_PATH=./feast_repo/data",
                        "",
                    ]
                )

            if feast_online == "Redis":
                lines.extend(
                    [
                        "# Feast Online Store (Redis)",
                        "REDIS_HOST=localhost",
                        "REDIS_PORT=6379",
                        "REDIS_PASSWORD=",
                        "",
                    ]
                )
            elif feast_online == "DynamoDB":
                lines.extend(
                    [
                        "# Feast Online Store (DynamoDB)",
                        "DYNAMODB_TABLE_NAME=feast-online-store",
                        "",
                    ]
                )
            else:  # SQLite
                lines.extend(
                    [
                        "# Feast Online Store (SQLite)",
                        "FEAST_ONLINE_STORE_PATH=./feast_repo/online_store.db",
                        "",
                    ]
                )

        # API Serving 설정
        if selections.get("enable_serving"):
            lines.extend(
                [
                    "# API Serving Configuration",
                    "API_HOST=0.0.0.0",
                    "API_PORT=8000",
                    "API_WORKERS=1",
                    "",
                ]
            )

        # Output: Inference
        infer_enabled = selections.get("inference_output_enabled", True)
        lines.extend(
            [
                "# Inference Output",
                f"INFER_OUTPUT_ENABLED={'true' if infer_enabled else 'false'}",
            ]
        )
        if infer_enabled:
            if infer_src == "Local Files":
                local_path = selections.get("infer_local_path", "./artifacts/predictions")
                lines.append(f"INFER_OUTPUT_BASE_PATH={local_path}")
            elif infer_src == "S3":
                s3_bucket = selections.get("infer_s3_bucket", "mmp-predictions")
                s3_prefix = selections.get("infer_s3_prefix", f"{selections['env_name']}/predictions")
                lines.extend(
                    [
                        f"INFER_OUTPUT_S3_BUCKET={s3_bucket}",
                        f"INFER_OUTPUT_S3_PREFIX={s3_prefix}",
                    ]
                )
            elif infer_src == "GCS":
                gcs_bucket = selections.get("infer_gcs_bucket", "mmp-predictions")
                gcs_prefix = selections.get("infer_gcs_prefix", f"{selections['env_name']}/predictions")
                lines.extend(
                    [
                        f"INFER_OUTPUT_GCS_BUCKET={gcs_bucket}",
                        f"INFER_OUTPUT_GCS_PREFIX={gcs_prefix}",
                    ]
                )
            elif infer_src == "PostgreSQL":
                pg_table = selections.get("infer_pg_table", f"predictions_{selections['env_name']}")
                lines.append(f"INFER_OUTPUT_PG_TABLE={pg_table}")
            elif infer_src == "BigQuery":
                bq_dataset = selections.get("infer_bq_dataset", "ml_predictions")
                bq_table = selections.get("infer_bq_table", f"predictions_{selections['env_name']}")
                bq_location = selections.get("infer_bq_location", "US")
                lines.extend(
                    [
                        f"INFER_OUTPUT_BQ_DATASET={bq_dataset}",
                        f"INFER_OUTPUT_BQ_TABLE={bq_table}",
                        f"BQ_LOCATION={bq_location}",
                    ]
                )
            lines.append("")

        return "\n".join(lines)
