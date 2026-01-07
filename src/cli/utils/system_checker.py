"""
System Checker for Modern ML Pipeline CLI
Phase 5: Config-based connection validation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 모듈화된 설계
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def expand_env_vars(value: str) -> str:
    """환경 변수를 확장하고 기본값을 처리.

    ${VAR:default} 형식과 ${VAR} 형식 모두 지원.

    Args:
        value: 환경 변수가 포함된 문자열

    Returns:
        환경 변수가 확장된 문자열
    """
    if not isinstance(value, str):
        return value

    # ${VAR:default} 패턴 처리
    pattern = r"\$\{([^}:]+):([^}]*)\}"

    def replace_with_default(match):
        var_name = match.group(1)
        default_value = match.group(2)
        return os.environ.get(var_name, default_value)

    result = re.sub(pattern, replace_with_default, value)

    # ${VAR} 패턴 처리 (기본값 없음)
    result = os.path.expandvars(result)

    return result


class CheckStatus(Enum):
    """체크 상태 열거형."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class CheckResult:
    """체크 결과 데이터 클래스."""

    service: str
    status: CheckStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    solution: Optional[str] = None


class SystemChecker:
    """시스템 연결 상태 체커.

    Config 파일 기반으로 설정된 서비스들의 실제 연결 상태를 검증합니다.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        env_name: str,
        config_path: str = None,
        recipe: Dict[str, Any] = None,
    ):
        """
        SystemChecker 초기화.

        Args:
            config: 로드된 config 딕셔너리
            env_name: 환경 이름 (호환성을 위해 유지)
            config_path: config 파일 경로 (display용)
            recipe: 로드된 recipe 딕셔너리 (선택, 모델 기반 의존성 체크용)
        """
        self.config = config
        self.env_name = env_name
        self.config_path = config_path or f"configs/{env_name}.yaml"
        self.recipe = recipe
        self.results: List[CheckResult] = []

    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 체크 실행 (system_check_command.py 호환성을 위해 Dict 반환).

        Returns:
            체크 결과 딕셔너리 (service_name -> result)
        """
        self.results = []

        # 0. 패키지 의존성 체크 (config 기반)
        self.results.append(self.check_package_dependencies())

        # 1. MLflow 체크
        if "mlflow" in self.config:
            self.results.append(self.check_mlflow())

        # 2. 데이터 소스 체크 (새로운 단일 data_source 구조)
        if "data_source" in self.config:
            data_source = self.config["data_source"]
            self.results.append(self.check_data_source(data_source))
            
            # 2.1 데이터 소스 호환성 체크 (Config 어댑터 vs Recipe URI)
            if self.recipe:
                self.results.append(self.check_data_source_compatibility())

        # 3. Feature Store 체크 (provider가 none이면 건너뜀)
        if "feature_store" in self.config:
            provider = self.config["feature_store"].get("provider", "").lower()
            if provider and provider != "none":
                self.results.append(self.check_feature_store())

        # 4. Output targets 체크 (inference output만 체크)
        if "output" in self.config:
            output_config = self.config["output"]
            if "inference" in output_config:
                self.results.append(
                    self.check_output_target("inference", output_config["inference"])
                )

        # 5. Serving 체크
        if self.config.get("serving", {}).get("enabled", False):
            self.results.append(self.check_serving())

        # system_check_command.py와의 호환성을 위해 Dict 형태로 반환
        results_dict = {}
        for result in self.results:
            results_dict[result.service] = {
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "solution": result.solution,
            }

        return results_dict

    def check_mlflow(self) -> CheckResult:
        """
        MLflow 연결 체크.

        Returns:
            체크 결과
        """
        mlflow_config = self.config["mlflow"]
        tracking_uri = expand_env_vars(mlflow_config.get("tracking_uri", "./mlruns"))

        try:
            if tracking_uri.startswith("http"):
                # HTTP endpoint check
                try:
                    import requests
                except ImportError:
                    return CheckResult(
                        service="MLflow",
                        status=CheckStatus.WARNING,
                        message="requests library not installed",
                        solution="pip install requests",
                    )
                response = requests.get(f"{tracking_uri}/health", timeout=5)
                if response.status_code == 200:
                    return CheckResult(
                        service="MLflow",
                        status=CheckStatus.SUCCESS,
                        message=f"MLflow server is healthy at {tracking_uri}",
                        details={"tracking_uri": tracking_uri},
                    )
                else:
                    raise Exception(f"Health check returned {response.status_code}")
            else:
                # Local directory check
                tracking_path = Path(tracking_uri)
                if tracking_path.exists() or tracking_uri == "./mlruns":
                    return CheckResult(
                        service="MLflow",
                        status=CheckStatus.SUCCESS,
                        message="MLflow local tracking directory is accessible",
                        details={"tracking_uri": tracking_uri},
                    )
                else:
                    raise FileNotFoundError(f"Directory not found: {tracking_uri}")

        except Exception as e:
            return CheckResult(
                service="MLflow",
                status=CheckStatus.FAILED,
                message=f"MLflow connection failed: {str(e)}",
                details={"tracking_uri": tracking_uri, "error": str(e)},
                solution="Check if MLflow server is running or tracking directory exists",
            )

    def check_data_source(self, data_source: Dict[str, Any]) -> CheckResult:
        """
        데이터 소스 연결 체크 (새로운 단일 data_source 구조).

        Args:
            data_source: 데이터 소스 설정

        Returns:
            체크 결과
        """
        source_name = data_source.get("name", "Unknown")
        adapter_type = data_source.get("adapter_type", "")
        config = data_source.get("config", {})

        # SQL Adapter - data_source.name 또는 config 구조로 타입 판별
        if adapter_type == "sql":
            # BigQuery: project_id 필드 존재 또는 name이 BigQuery
            if source_name == "BigQuery" or "project_id" in config:
                return self._check_bigquery(source_name, config)
            # PostgreSQL: connection_uri에 postgresql:// 포함
            connection_uri = str(config.get("connection_uri", ""))
            if "postgresql://" in connection_uri:
                return self._check_postgresql(source_name, config)
            # bigquery:// URI 패턴
            if "bigquery://" in connection_uri:
                return self._check_bigquery(source_name, config)
            # 기타 SQL은 PostgreSQL로 시도
            return self._check_postgresql(source_name, config)

        # Storage Adapter (Local Files, S3, GCS)
        elif adapter_type == "storage":
            base_path = config.get("base_path", "")
            if base_path.startswith("s3://"):
                return self._check_s3(source_name, config)
            elif base_path.startswith("gs://"):
                return self._check_gcs(source_name, config)
            else:
                return self._check_storage(source_name, config)

        else:
            return CheckResult(
                service=f"DataSource:{source_name}",
                status=CheckStatus.SKIPPED,
                message=f"Unknown adapter type: {adapter_type}",
            )

    def check_adapter(self, adapter_name: str, adapter_config: Dict[str, Any]) -> CheckResult:
        """
        레거시 어댑터 체크 메서드 (하위 호환성).

        Args:
            adapter_name: 어댑터 이름
            adapter_config: 어댑터 설정

        Returns:
            체크 결과
        """
        class_name = adapter_config.get("class_name", "")
        config = adapter_config.get("config", {})

        # SQL Adapter (PostgreSQL, BigQuery, etc.)
        if "SqlAdapter" in class_name:
            connection_uri = str(config.get("connection_uri", ""))
            if "postgresql://" in connection_uri:
                return self._check_postgresql(adapter_name, config)
            elif "bigquery://" in connection_uri:
                return self._check_bigquery(adapter_name, config)
            else:
                # Other SQL databases
                return self._check_postgresql(adapter_name, config)

        # Storage Adapter (Local Files)
        elif "StorageAdapter" in class_name:
            return self._check_storage(adapter_name, config)

        # S3 Adapter
        elif "S3Adapter" in class_name or "s3://" in str(config.get("base_path", "")):
            return self._check_s3(adapter_name, config)

        # GCS Adapter
        elif "GCSAdapter" in class_name or "gs://" in str(config.get("base_path", "")):
            return self._check_gcs(adapter_name, config)

        else:
            return CheckResult(
                service=f"Adapter:{adapter_name}",
                status=CheckStatus.SKIPPED,
                message=f"Unknown adapter type: {class_name}",
            )

    def _check_postgresql(self, source_name: str, config: Dict[str, Any]) -> CheckResult:
        """PostgreSQL 연결 체크."""
        connection_uri = expand_env_vars(config.get("connection_uri", ""))

        try:
            # Parse connection URI
            import urllib.parse

            parsed = urllib.parse.urlparse(connection_uri)

            try:
                import psycopg2
            except ImportError:
                return CheckResult(
                    service=f"PostgreSQL ({source_name})",
                    status=CheckStatus.WARNING,
                    message="psycopg2 library not installed",
                    solution="pip install psycopg2-binary",
                )
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password,
            )
            conn.close()

            return CheckResult(
                service=f"PostgreSQL:{source_name}",
                status=CheckStatus.SUCCESS,
                message="PostgreSQL connection successful",
                details={"host": parsed.hostname, "database": parsed.path[1:]},
            )

        except Exception as e:
            return CheckResult(
                service=f"PostgreSQL:{source_name}",
                status=CheckStatus.FAILED,
                message=f"PostgreSQL connection failed: {str(e)}",
                solution="Check database credentials and network connectivity",
            )

    def _check_bigquery(self, source_name: str, config: Dict[str, Any]) -> CheckResult:
        """BigQuery 연결 체크 (project_id만 필요)."""
        project_id = expand_env_vars(config.get("project_id", ""))

        if not project_id:
            return CheckResult(
                service=f"BigQuery:{source_name}",
                status=CheckStatus.WARNING,
                message="BigQuery project_id not configured",
                solution="Set GCP_PROJECT_ID environment variable",
            )

        try:
            from google.cloud import bigquery

            client = bigquery.Client(project=project_id)
            # project 접근 가능 여부만 확인 (datasets 목록 조회)
            list(client.list_datasets(max_results=1))

            return CheckResult(
                service=f"BigQuery:{source_name}",
                status=CheckStatus.SUCCESS,
                message=f"BigQuery project accessible: {project_id}",
                details={"project": project_id},
            )

        except ImportError:
            return CheckResult(
                service=f"BigQuery:{source_name}",
                status=CheckStatus.WARNING,
                message="google-cloud-bigquery not installed",
                solution="Run: pip install google-cloud-bigquery",
            )
        except Exception as e:
            return CheckResult(
                service=f"BigQuery:{source_name}",
                status=CheckStatus.FAILED,
                message=f"BigQuery connection failed: {str(e)}",
                solution="Check GCP credentials and permissions",
            )

    def _check_storage(self, source_name: str, config: Dict[str, Any]) -> CheckResult:
        """Local storage 체크."""
        base_path = expand_env_vars(config.get("base_path", "./data"))
        storage_path = Path(base_path)

        if storage_path.exists():
            return CheckResult(
                service=f"Storage:{source_name}",
                status=CheckStatus.SUCCESS,
                message="Storage path exists",
                details={"path": str(storage_path.absolute())},
            )
        else:
            return CheckResult(
                service=f"Storage:{source_name}",
                status=CheckStatus.WARNING,
                message=f"Storage path does not exist: {base_path}",
                solution=f"Create directory: mkdir -p {base_path}",
            )

    def _check_s3(self, source_name: str, config: Dict[str, Any]) -> CheckResult:
        """S3 연결 체크."""
        try:
            import boto3

            # Create S3 client
            s3 = boto3.client("s3")

            # Try to list buckets
            buckets = s3.list_buckets()

            return CheckResult(
                service=f"S3:{source_name}",
                status=CheckStatus.SUCCESS,
                message="S3 connection successful",
                details={"bucket_count": len(buckets.get("Buckets", []))},
            )

        except ImportError:
            return CheckResult(
                service=f"S3:{source_name}",
                status=CheckStatus.WARNING,
                message="boto3 not installed",
                solution="Run: pip install boto3",
            )
        except Exception as e:
            return CheckResult(
                service=f"S3:{source_name}",
                status=CheckStatus.FAILED,
                message=f"S3 connection failed: {str(e)}",
                solution="Check AWS credentials and permissions",
            )

    def _check_gcs(self, source_name: str, config: Dict[str, Any]) -> CheckResult:
        """GCS 연결 체크."""
        try:
            from google.cloud import storage

            # Create GCS client
            client = storage.Client()

            # Try to list buckets
            buckets = list(client.list_buckets())

            return CheckResult(
                service=f"GCS:{source_name}",
                status=CheckStatus.SUCCESS,
                message="GCS connection successful",
                details={"bucket_count": len(buckets)},
            )

        except ImportError:
            return CheckResult(
                service=f"GCS:{source_name}",
                status=CheckStatus.WARNING,
                message="google-cloud-storage not installed",
                solution="Run: pip install google-cloud-storage",
            )
        except Exception as e:
            return CheckResult(
                service=f"GCS:{source_name}",
                status=CheckStatus.FAILED,
                message=f"GCS connection failed: {str(e)}",
                solution="Check GCP credentials and permissions",
            )

    def check_output_target(self, output_type: str, output_config: Dict[str, Any]) -> CheckResult:
        """
        Output target 연결 체크 (inference output).

        Args:
            output_type: 출력 타입 (예: "inference")
            output_config: 출력 설정

        Returns:
            체크 결과
        """
        if not output_config.get("enabled", True):
            return CheckResult(
                service=f"Output:{output_type}",
                status=CheckStatus.SKIPPED,
                message=f"{output_type} output is disabled",
            )

        adapter_type = output_config.get("adapter_type", "")
        config = output_config.get("config", {})

        # Storage adapter (Local Files, S3, GCS)
        if adapter_type == "storage":
            base_path = config.get("base_path", "")
            if base_path.startswith("s3://"):
                return self._check_s3(f"{output_type}_output", config)
            elif base_path.startswith("gs://"):
                return self._check_gcs(f"{output_type}_output", config)
            else:
                # Local storage
                return self._check_storage(f"{output_type}_output", config)

        # SQL adapter (PostgreSQL, BigQuery)
        elif adapter_type == "sql":
            connection_uri = config.get("connection_uri", "")
            if "postgresql://" in connection_uri:
                return self._check_postgresql(f"{output_type}_output", config)
            elif "bigquery://" in connection_uri:
                return self._check_bigquery(f"{output_type}_output", config)
            else:
                return CheckResult(
                    service=f"Output:{output_type}",
                    status=CheckStatus.WARNING,
                    message=f"Unknown SQL type in connection URI: {connection_uri}",
                )

        else:
            return CheckResult(
                service=f"Output:{output_type}",
                status=CheckStatus.SKIPPED,
                message=f"Unknown output adapter type: {adapter_type}",
            )

    def check_feature_store(self) -> CheckResult:
        """Feature Store 체크."""
        fs_config = self.config["feature_store"]
        provider = fs_config.get("provider", "").lower()

        # "none" provider는 Feature Store 미사용
        if provider == "none" or not provider:
            return CheckResult(
                service="FeatureStore",
                status=CheckStatus.SKIPPED,
                message="Feature store not configured (provider: none)",
            )

        if provider == "feast":
            return self._check_feast(fs_config.get("feast_config", {}))
        elif provider == "tecton":
            return self._check_tecton(fs_config)
        else:
            return CheckResult(
                service="FeatureStore",
                status=CheckStatus.SKIPPED,
                message=f"Unknown feature store provider: {provider}",
            )

    def _check_feast(self, feast_config: Dict[str, Any]) -> CheckResult:
        """Feast Feature Store 체크."""
        registry_path = expand_env_vars(feast_config.get("registry", "./feast_repo/registry.db"))

        try:
            # Check if registry exists
            if Path(registry_path).exists():
                # Check online store
                online_store = feast_config.get("online_store", {})
                if online_store.get("type") == "redis":
                    # Check Redis connection
                    try:
                        import redis
                    except ImportError:
                        return CheckResult(
                            service="Feature Store (Feast)",
                            status=CheckStatus.WARNING,
                            message="redis library not installed",
                            solution="pip install redis",
                        )
                    redis_conn = expand_env_vars(
                        online_store.get("connection_string", "localhost:6379")
                    )
                    host, port = redis_conn.split(":")
                    r = redis.Redis(host=host, port=int(port), socket_connect_timeout=5)
                    r.ping()

                return CheckResult(
                    service="Feast",
                    status=CheckStatus.SUCCESS,
                    message="Feast registry and online store accessible",
                    details={"registry": registry_path},
                )
            else:
                return CheckResult(
                    service="Feast",
                    status=CheckStatus.WARNING,
                    message=f"Feast registry not found: {registry_path}",
                    solution="Initialize Feast: feast init && feast apply",
                )

        except Exception as e:
            return CheckResult(
                service="Feast",
                status=CheckStatus.FAILED,
                message=f"Feast check failed: {str(e)}",
                solution="Check Feast configuration and Redis connection",
            )

    def _check_tecton(self, tecton_config: Dict[str, Any]) -> CheckResult:
        """Tecton Feature Store 체크."""
        tecton_url = expand_env_vars(tecton_config.get("tecton_url", ""))

        if not tecton_url:
            return CheckResult(
                service="Tecton",
                status=CheckStatus.WARNING,
                message="Tecton URL not configured",
                solution="Set TECTON_URL environment variable",
            )

        try:
            try:
                import requests
            except ImportError:
                return CheckResult(
                    service="Feature Store (Tecton)",
                    status=CheckStatus.WARNING,
                    message="requests library not installed",
                    solution="pip install requests",
                )
            response = requests.get(f"{tecton_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                return CheckResult(
                    service="Tecton",
                    status=CheckStatus.SUCCESS,
                    message="Tecton service is healthy",
                    details={"url": tecton_url},
                )
            else:
                raise Exception(f"Health check returned {response.status_code}")

        except Exception as e:
            return CheckResult(
                service="Tecton",
                status=CheckStatus.FAILED,
                message=f"Tecton connection failed: {str(e)}",
                solution="Check Tecton URL and API key",
            )

    def check_serving(self) -> CheckResult:
        """API Serving 체크."""
        serving_config = self.config.get("serving", {})
        host = expand_env_vars(serving_config.get("host", "0.0.0.0"))
        port = int(expand_env_vars(str(serving_config.get("port", 8000))))

        # Just check if port is valid
        if 1 <= port <= 65535:
            return CheckResult(
                service="Serving",
                status=CheckStatus.SUCCESS,
                message="Serving configuration valid",
                details={"host": host, "port": port},
            )
        else:
            return CheckResult(
                service="Serving",
                status=CheckStatus.WARNING,
                message=f"Invalid port number: {port}",
                solution="Port must be between 1 and 65535",
            )

    def check_data_source_compatibility(self) -> CheckResult:
        """데이터 소스 어댑터와 데이터 경로의 호환성 검사."""
        if not self.recipe:
            return CheckResult(
                service="DataSourceCompatibility",
                status=CheckStatus.SKIPPED,
                message="Recipe not provided, skipping compatibility check"
            )

        data_source = self.config.get("data_source", {})
        adapter_type = data_source.get("adapter_type", "").lower()
        
        # Recipe에서 데이터 경로 추출
        data_config = self.recipe.get("data", {})
        loader_config = data_config.get("loader", {})
        source_uri = loader_config.get("source_uri", "")
        
        if not source_uri:
            return CheckResult(
                service="DataSourceCompatibility",
                status=CheckStatus.SKIPPED,
                message="No source_uri defined in recipe"
            )

        is_sql_file = source_uri.endswith((".sql", ".sql.j2"))
        
        # 1. Storage 어댑터 + SQL 파일 조합 체크
        if adapter_type == "storage" and is_sql_file:
            return CheckResult(
                service="DataSourceCompatibility",
                status=CheckStatus.FAILED,
                message=f"호환되지 않는 조합: 'storage' 어댑터는 SQL 파일({Path(source_uri).suffix})을 처리할 수 없습니다.",
                details={"adapter": adapter_type, "source_uri": source_uri},
                solution=(
                    f"1. {self.config_path}의 'adapter_type'을 'sql'로 변경하세요.\n"
                    f"2. 또는 {self.recipe.get('name', 'recipe')}.yaml의 'source_uri'를 '.csv'나 '.parquet' 파일로 변경하세요."
                )
            )
            
        # 2. SQL 어댑터 + 비 SQL 파일 조합 체크 (경고 수준)
        if adapter_type == "sql" and not is_sql_file and "." in source_uri:
            # 확장자가 있는데 SQL 관련이 아닌 경우
            ext = Path(source_uri).suffix.lower()
            if ext in (".csv", ".parquet", ".json"):
                return CheckResult(
                    service="DataSourceCompatibility",
                    status=CheckStatus.WARNING,
                    message=f"주의: 'sql' 어댑터가 데이터 파일({ext})을 읽으려 합니다. 쿼리 실행이 실패할 수 있습니다.",
                    details={"adapter": adapter_type, "source_uri": source_uri},
                    solution=f"데이터 파일을 읽으려면 {self.config_path}의 'adapter_type'을 'storage'로 변경하세요."
                )

        return CheckResult(
            service="DataSourceCompatibility",
            status=CheckStatus.SUCCESS,
            message="데이터 소스 어댑터와 경로가 호환됩니다.",
            details={"adapter": adapter_type, "source_uri": source_uri}
        )

    def check_package_dependencies(self) -> CheckResult:
        """
        Config 및 Recipe에 따른 패키지 의존성 체크.
        설정된 서비스 및 모델에 필요한 extras 패키지가 설치되어 있는지 확인.

        Returns:
            체크 결과
        """
        missing_packages = []
        extras_needed = []
        checked_packages = []

        # BigQuery 사용 여부 확인
        data_source = self.config.get("data_source", {})
        source_name = data_source.get("name", "")
        connection_uri = str(data_source.get("config", {}).get("connection_uri", ""))

        if (
            source_name == "BigQuery"
            or "bigquery://" in connection_uri
            or "project_id" in data_source.get("config", {})
        ):
            checked_packages.append("sqlalchemy-bigquery")
            try:
                import sqlalchemy_bigquery  # noqa: F401
            except ImportError:
                missing_packages.append("sqlalchemy-bigquery")
                extras_needed.append("cloud-extras")

        # Output에서도 BigQuery 사용 여부 확인
        output_config = self.config.get("output", {})
        for output_name, output_cfg in output_config.items():
            if isinstance(output_cfg, dict):
                out_uri = str(output_cfg.get("config", {}).get("connection_uri", ""))
                if "bigquery://" in out_uri:
                    if "sqlalchemy-bigquery" not in checked_packages:
                        checked_packages.append("sqlalchemy-bigquery")
                    try:
                        import sqlalchemy_bigquery  # noqa: F401
                    except ImportError:
                        if "sqlalchemy-bigquery" not in missing_packages:
                            missing_packages.append("sqlalchemy-bigquery")
                            if "cloud-extras" not in extras_needed:
                                extras_needed.append("cloud-extras")

        # Feature Store (Feast) 사용 여부 확인
        fs_config = self.config.get("feature_store", {})
        if fs_config.get("provider", "").lower() == "feast":
            checked_packages.append("feast")
            try:
                import feast  # noqa: F401
            except ImportError:
                missing_packages.append("feast")
                extras_needed.append("feature-store")

        # Recipe 기반 모델 의존성 확인
        if self.recipe:
            model_config = self.recipe.get("model", {})
            library = model_config.get("library", "").lower()

            # ml-extras: LightGBM, CatBoost (XGBoost는 core에 포함)
            ml_extras_libs = ["lightgbm", "catboost"]
            for lib in ml_extras_libs:
                if library == lib:
                    checked_packages.append(lib)
                    try:
                        __import__(lib)
                    except ImportError:
                        missing_packages.append(lib)
                        if "ml-extras" not in extras_needed:
                            extras_needed.append("ml-extras")

            # torch-extras: PyTorch 기반 모델 (FT-Transformer, LSTM 등)
            if library in ["torch", "pytorch"]:
                checked_packages.append("torch")
                try:
                    import torch  # noqa: F401
                except ImportError:
                    missing_packages.append("torch")
                    extras_needed.append("torch-extras")

        # 결과 반환
        if missing_packages:
            extras_str = ",".join(sorted(set(extras_needed)))
            return CheckResult(
                service="PackageDependencies",
                status=CheckStatus.FAILED,
                message=f"필요한 패키지 미설치: {', '.join(missing_packages)}",
                details={"missing": missing_packages, "extras": list(set(extras_needed))},
                solution=f'pip install "modern-ml-pipeline[{extras_str}]"',
            )
        else:
            return CheckResult(
                service="PackageDependencies",
                status=CheckStatus.SUCCESS,
                message="필요한 모든 패키지가 설치되어 있습니다",
                details={"checked": checked_packages if checked_packages else ["기본 패키지"]},
            )
