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
        selections: Dict[str, Any] = {}
        total_steps = 6

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
            mlflow_tracking = self.ui.text_input(
                "MLflow Tracking URI",
                default="./mlruns" if env_name == "local" else "http://mlflow-server:5000",
            )
            selections["mlflow_tracking_uri"] = mlflow_tracking

            mlflow_experiment = self.ui.text_input(
                "MLflow Experiment 이름",
                default=f"mmp-{env_name}",
            )
            selections["mlflow_experiment_name"] = mlflow_experiment

        # Step 3: 데이터 소스 선택
        self.ui.show_step(3, total_steps, "데이터 소스")
        data_sources = ["PostgreSQL", "BigQuery", "Local Files", "S3", "GCS"]

        data_source = self.ui.select_from_list(
            "데이터를 로드할 소스를 선택하세요", data_sources, allow_cancel=False
        )
        selections["data_source"] = data_source

        # 데이터 소스별 추가 설정
        if data_source == "BigQuery":
            selections["gcp_project_id"] = self.ui.text_input(
                "GCP Project ID", default="my-project"
            )
            selections["bq_location"] = self.ui.text_input(
                "BigQuery Location", default="US"
            )
        elif data_source == "GCS":
            selections["gcp_project_id"] = self.ui.text_input(
                "GCP Project ID", default="my-project"
            )
            selections["gcs_data_bucket"] = self.ui.text_input(
                "GCS Bucket 이름", default="mmp-data"
            )
            selections["gcs_data_prefix"] = self.ui.text_input(
                "GCS Prefix (경로)", default=env_name
            )
        elif data_source == "S3":
            selections["s3_data_bucket"] = self.ui.text_input(
                "S3 Bucket 이름", default="mmp-data"
            )
            selections["s3_data_prefix"] = self.ui.text_input(
                "S3 Prefix (경로)", default=env_name
            )
            selections["aws_region"] = self.ui.text_input(
                "AWS Region", default="us-east-1"
            )
        elif data_source == "PostgreSQL":
            selections["db_host"] = self.ui.text_input("DB Host", default="localhost")
            selections["db_port"] = self.ui.text_input("DB Port", default="5432")
            selections["db_name"] = self.ui.text_input("DB Name", default="mlpipeline")
        elif data_source == "Local Files":
            selections["local_data_path"] = self.ui.text_input(
                "데이터 경로", default="./data"
            )

        # Step 4: Feature Store 선택
        self.ui.show_step(4, total_steps, "Feature Store")
        feature_stores = ["없음", "Feast"]

        feature_store = self.ui.select_from_list(
            "Feature Store를 선택하세요", feature_stores, allow_cancel=False
        )
        selections["feature_store"] = feature_store

        if feature_store == "Feast":
            selections["feast_project"] = self.ui.text_input(
                "Feast Project 이름", default=f"feast_{env_name}"
            )
            registry_location = self.ui.select_from_list(
                "Feast Registry 저장 위치", ["로컬", "S3", "GCS"], allow_cancel=False
            )
            selections["feast_registry_location"] = registry_location

            if registry_location == "로컬":
                selections["feast_registry_path"] = self.ui.text_input(
                    "Registry 경로", default="./feast_repo/registry.db"
                )
            elif registry_location == "S3":
                selections["feast_registry_path"] = self.ui.text_input(
                    "Registry S3 경로",
                    default=f"s3://mmp-feast/registry/{env_name}/registry.db",
                )
            else:  # GCS
                selections["feast_registry_path"] = self.ui.text_input(
                    "Registry GCS 경로",
                    default=f"gs://mmp-feast/registry/{env_name}/registry.db",
                )

            self.ui.show_info(f"Offline Store는 {data_source}에 따라 자동 설정됩니다.")

            if data_source in ["PostgreSQL", "Local Files", "S3", "GCS"]:
                self.ui.show_info("Offline Store는 Parquet 파일 형식을 사용합니다.")
                selections["feast_offline_path"] = self.ui.text_input(
                    "Offline Store 경로", default="./feast_repo/data"
                )

            use_online_store = self.ui.confirm(
                "Online Store를 사용하시겠습니까? (실시간 서빙용)", default=False
            )

            if use_online_store:
                online_store_type = self.ui.select_from_list(
                    "Online Store 타입", ["Redis", "SQLite", "DynamoDB"], allow_cancel=False
                )
                selections["feast_online_store"] = online_store_type

                if online_store_type == "Redis":
                    selections["redis_host"] = self.ui.text_input(
                        "Redis Host", default="localhost"
                    )
                    selections["redis_port"] = self.ui.text_input(
                        "Redis Port", default="6379"
                    )
                elif online_store_type == "SQLite":
                    selections["feast_online_store_path"] = self.ui.text_input(
                        "SQLite 경로", default="./feast_repo/online_store.db"
                    )
                elif online_store_type == "DynamoDB":
                    selections["dynamodb_table"] = self.ui.text_input(
                        "DynamoDB Table", default="feast-online-store"
                    )
                    selections["dynamodb_region"] = self.ui.text_input(
                        "DynamoDB Region", default="us-east-1"
                    )
            else:
                selections["feast_online_store"] = "SQLite"
                selections["feast_online_store_path"] = "./feast_repo/online_store.db"

        # Step 5: 로깅 설정
        self.ui.show_step(5, total_steps, "로깅 설정")
        selections["log_path"] = self.ui.text_input("로그 저장 경로", default="./logs")
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        selections["log_level"] = self.ui.select_from_list(
            "로그 레벨", log_levels, allow_cancel=False
        )
        selections["log_retention_days"] = self.ui.text_input(
            "로그 보관 기간 (일)", default="30"
        )

        # Step 6: 배치 추론 결과 저장 설정
        self.ui.show_step(6, total_steps, "배치 추론 출력")
        infer_enabled = self.ui.confirm("배치 추론 결과를 저장하시겠습니까?", default=True)
        selections["inference_output_enabled"] = infer_enabled

        if infer_enabled:
            infer_source = self.ui.select_from_list(
                "추론 결과 저장 위치를 선택하세요",
                ["PostgreSQL", "BigQuery", "Local Files", "S3", "GCS"],
                allow_cancel=False,
            )
            selections["inference_output_source"] = infer_source

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
                if "gcp_project_id" not in selections:
                    selections["gcp_project_id"] = self.ui.text_input(
                        "GCP Project ID", default="my-project"
                    )
                selections["infer_bq_dataset"] = self.ui.text_input(
                    "BigQuery Dataset", default="ml_predictions"
                )
                selections["infer_bq_table"] = self.ui.text_input(
                    "BigQuery Table", default=f"predictions_{env_name}"
                )
                if "bq_location" not in selections:
                    selections["bq_location"] = self.ui.text_input(
                        "BigQuery Location", default="US"
                    )
            elif infer_source == "PostgreSQL":
                if "db_host" not in selections:
                    selections["db_host"] = self.ui.text_input("DB Host", default="localhost")
                    selections["db_port"] = self.ui.text_input("DB Port", default="5432")
                    selections["db_name"] = self.ui.text_input("DB Name", default="mlpipeline")
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
            "MLflow": self._format_mlflow_display(selections),
            "데이터 소스": self._format_data_source_display(selections),
            "Feature Store": selections.get("feature_store", "없음"),
            "로깅": f"{selections.get('log_path', './logs')} ({selections.get('log_level', 'INFO')})",
            "추론 출력": infer_output_display,
        }

        if not self.ui.show_summary_and_confirm(summary_data, title="설정 요약"):
            self.ui.show_warning("설정이 취소되었습니다. 다시 시작해주세요.")
            return self.run_interactive_flow(env_name)

        return selections

    def _format_mlflow_display(self, selections: Dict[str, Any]) -> str:
        """MLflow 설정 표시 문자열 반환."""
        if not selections.get("use_mlflow"):
            return "사용 안 함"
        uri = selections.get("mlflow_tracking_uri", "./mlruns")
        exp = selections.get("mlflow_experiment_name", "")
        return f"{uri} ({exp})"

    def _format_data_source_display(self, selections: Dict[str, Any]) -> str:
        """데이터 소스 표시 문자열 반환."""
        source = selections.get("data_source", "")
        if source == "BigQuery":
            project = selections.get("gcp_project_id", "")
            return f"BigQuery ({project})"
        elif source == "GCS":
            bucket = selections.get("gcs_data_bucket", "")
            return f"GCS (gs://{bucket}/...)"
        elif source == "S3":
            bucket = selections.get("s3_data_bucket", "")
            return f"S3 (s3://{bucket}/...)"
        elif source == "PostgreSQL":
            host = selections.get("db_host", "localhost")
            db = selections.get("db_name", "")
            return f"PostgreSQL ({host}/{db})"
        elif source == "Local Files":
            path = selections.get("local_data_path", "./data")
            return f"Local ({path})"
        return source

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
            return f"BigQuery ({dataset}.{table})"
        elif source == "PostgreSQL":
            table = selections.get("infer_pg_table", "")
            return f"PostgreSQL ({table})"
        elif source == "Local Files":
            path = selections.get("infer_local_path", "./artifacts/predictions")
            return f"Local ({path})"

        return source

    def generate_config_file(self, env_name: str, selections: Dict[str, Any]) -> Path:
        """
        설정 파일 생성.

        Args:
            env_name: 환경 이름
            selections: 사용자 선택 사항

        Returns:
            생성된 설정 파일 경로
        """
        context = self._prepare_template_context(selections)

        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / f"{env_name}.yaml"

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
        env_content = self._generate_env_template_content(selections)
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

        feature_store = selections.get("feature_store", "없음")
        context["use_feast"] = feature_store == "Feast"

        context["enable_auth"] = False
        context.setdefault("serving_workers", 1)
        context.setdefault("model_stage", "None")

        if selections.get("inference_output_enabled", True):
            context["inference_output_source"] = selections.get(
                "inference_output_source", "Local Files"
            )

        return context

    def _generate_env_template_content(self, selections: Dict[str, Any]) -> str:
        """
        환경 변수 템플릿 내용 생성 (민감 정보만 포함).

        Args:
            selections: 사용자 선택 사항

        Returns:
            환경 변수 템플릿 내용
        """
        lines = [
            f"# Environment variables for {selections['env_name']}",
            f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "# 민감한 인증 정보만 환경 변수로 관리합니다.",
            "",
        ]

        needs_aws = False
        needs_gcp = False
        needs_postgresql = False
        needs_redis = False

        data_source = selections.get("data_source", "")
        feature_store = selections.get("feature_store", "없음")
        feast_registry = selections.get("feast_registry_location", "로컬")
        feast_online = selections.get("feast_online_store", "SQLite")
        infer_src = selections.get("inference_output_source", "Local Files")

        # AWS 필요 여부
        if data_source == "S3" or infer_src == "S3":
            needs_aws = True
        if feature_store == "Feast" and feast_registry == "S3":
            needs_aws = True
        if feature_store == "Feast" and feast_online == "DynamoDB":
            needs_aws = True

        # GCP 필요 여부
        if data_source in ["BigQuery", "GCS"] or infer_src in ["GCS", "BigQuery"]:
            needs_gcp = True
        if feature_store == "Feast" and feast_registry == "GCS":
            needs_gcp = True

        # PostgreSQL 필요 여부
        if data_source == "PostgreSQL" or infer_src == "PostgreSQL":
            needs_postgresql = True

        # Redis 필요 여부
        if feature_store == "Feast" and feast_online == "Redis":
            needs_redis = True

        # MLflow 원격 서버 인증
        if selections.get("use_mlflow"):
            mlflow_uri = selections.get("mlflow_tracking_uri", "./mlruns")
            is_remote = mlflow_uri.startswith("http://") or mlflow_uri.startswith("https://")
            if is_remote:
                lines.extend(
                    [
                        "# MLflow Authentication (원격 서버용)",
                        "MLFLOW_TRACKING_USERNAME=",
                        "MLFLOW_TRACKING_PASSWORD=",
                        "",
                    ]
                )

        # 클라우드 인증 정보
        if needs_gcp:
            lines.extend(
                [
                    "# GCP Authentication",
                    "GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json",
                    "",
                ]
            )

        if needs_aws:
            lines.extend(
                [
                    "# AWS Authentication",
                    "AWS_ACCESS_KEY_ID=",
                    "AWS_SECRET_ACCESS_KEY=",
                    "",
                ]
            )

        if needs_postgresql:
            lines.extend(
                [
                    "# PostgreSQL Authentication",
                    "DB_USER=postgres",
                    "DB_PASSWORD=",
                    "",
                ]
            )

        if needs_redis:
            lines.extend(
                [
                    "# Redis Authentication (선택사항)",
                    "REDIS_PASSWORD=",
                    "",
                ]
            )

        if len(lines) == 4:  # 헤더만 있는 경우
            lines.append("# 이 환경에서는 추가 인증 정보가 필요하지 않습니다.")
            lines.append("")

        return "\n".join(lines)
