"""
System Checker for Modern ML Pipeline CLI
Phase 5: Config-based connection validation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 모듈화된 설계
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from src.cli.utils.interactive_ui import InteractiveUI


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
    
    def __init__(self, config: Dict[str, Any], env_name: str, config_path: str = None):
        """
        SystemChecker 초기화.
        
        Args:
            config: 로드된 config 딕셔너리
            env_name: 환경 이름 (호환성을 위해 유지)
            config_path: config 파일 경로 (display용)
        """
        self.config = config
        self.env_name = env_name
        self.config_path = config_path or f"configs/{env_name}.yaml"
        self.ui = InteractiveUI()
        self.results: List[CheckResult] = []
    
    def run_all_checks(self) -> List[CheckResult]:
        """
        모든 체크 실행.
        
        Returns:
            체크 결과 리스트
        """
        self.results = []
        
        # 1. MLflow 체크
        if "mlflow" in self.config:
            self.results.append(self.check_mlflow())
        
        # 2. 데이터 소스 체크 (새로운 단일 data_source 구조)
        if "data_source" in self.config:
            data_source = self.config["data_source"]
            self.results.append(self.check_data_source(data_source))
        
        # 3. Feature Store 체크
        if "feature_store" in self.config:
            self.results.append(self.check_feature_store())
        
        # 4. Artifact Storage 체크 (새로운 단일 artifact_store 구조)
        if "artifact_store" in self.config:
            artifact_store = self.config["artifact_store"]
            self.results.append(self.check_artifact_store(artifact_store))
        
        # 5. Output targets 체크 (inference output만 체크)
        if "output" in self.config:
            output_config = self.config["output"]
            if "inference" in output_config:
                self.results.append(self.check_output_target("inference", output_config["inference"]))
        
        # 6. Serving 체크
        if self.config.get("serving", {}).get("enabled", False):
            self.results.append(self.check_serving())
        
        return self.results
    
    def check_mlflow(self) -> CheckResult:
        """
        MLflow 연결 체크.
        
        Returns:
            체크 결과
        """
        mlflow_config = self.config["mlflow"]
        tracking_uri = os.path.expandvars(mlflow_config.get("tracking_uri", "./mlruns"))
        
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
                        solution="pip install requests"
                    )
                response = requests.get(f"{tracking_uri}/health", timeout=5)
                if response.status_code == 200:
                    return CheckResult(
                        service="MLflow",
                        status=CheckStatus.SUCCESS,
                        message=f"MLflow server is healthy at {tracking_uri}",
                        details={"tracking_uri": tracking_uri}
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
                        details={"tracking_uri": tracking_uri}
                    )
                else:
                    raise FileNotFoundError(f"Directory not found: {tracking_uri}")
                    
        except Exception as e:
            return CheckResult(
                service="MLflow",
                status=CheckStatus.FAILED,
                message=f"MLflow connection failed: {str(e)}",
                details={"tracking_uri": tracking_uri, "error": str(e)},
                solution="Check if MLflow server is running or tracking directory exists"
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
        
        # SQL Adapter - check connection URI to determine specific database type
        if adapter_type == "sql":
            connection_uri = str(config.get("connection_uri", ""))
            if "bigquery://" in connection_uri:
                return self._check_bigquery(source_name, config)
            elif "postgresql://" in connection_uri:
                return self._check_postgresql(source_name, config)
            else:
                # Default to PostgreSQL for other SQL databases
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
                message=f"Unknown adapter type: {adapter_type}"
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
                message=f"Unknown adapter type: {class_name}"
            )
    
    def _check_postgresql(self, source_name: str, config: Dict[str, Any]) -> CheckResult:
        """PostgreSQL 연결 체크."""
        connection_uri = os.path.expandvars(config.get("connection_uri", ""))
        
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
                    solution="pip install psycopg2-binary"
                )
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password
            )
            conn.close()
            
            return CheckResult(
                service=f"PostgreSQL:{source_name}",
                status=CheckStatus.SUCCESS,
                message="PostgreSQL connection successful",
                details={"host": parsed.hostname, "database": parsed.path[1:]}
            )
            
        except Exception as e:
            return CheckResult(
                service=f"PostgreSQL:{source_name}",
                status=CheckStatus.FAILED,
                message=f"PostgreSQL connection failed: {str(e)}",
                solution="Check database credentials and network connectivity"
            )
    
    def _check_bigquery(self, source_name: str, config: Dict[str, Any]) -> CheckResult:
        """BigQuery 연결 체크."""
        project_id = os.path.expandvars(config.get("project_id", ""))
        dataset_id = os.path.expandvars(config.get("dataset_id", ""))
        
        if not project_id:
            return CheckResult(
                service=f"BigQuery:{source_name}",
                status=CheckStatus.WARNING,
                message="BigQuery project_id not configured",
                solution="Set GCP_PROJECT_ID environment variable"
            )
        
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=project_id)
            # Try to get dataset
            dataset = client.get_dataset(dataset_id)
            
            return CheckResult(
                service=f"BigQuery:{source_name}",
                status=CheckStatus.SUCCESS,
                message="BigQuery dataset accessible",
                details={"project": project_id, "dataset": dataset_id}
            )
            
        except ImportError:
            return CheckResult(
                service=f"BigQuery:{source_name}",
                status=CheckStatus.WARNING,
                message="google-cloud-bigquery not installed",
                solution="Run: pip install google-cloud-bigquery"
            )
        except Exception as e:
            return CheckResult(
                service=f"BigQuery:{source_name}",
                status=CheckStatus.FAILED,
                message=f"BigQuery connection failed: {str(e)}",
                solution="Check GCP credentials and permissions"
            )
    
    def _check_storage(self, source_name: str, config: Dict[str, Any]) -> CheckResult:
        """Local storage 체크."""
        base_path = config.get("base_path", "./data")
        storage_path = Path(base_path)
        
        if storage_path.exists():
            return CheckResult(
                service=f"Storage:{source_name}",
                status=CheckStatus.SUCCESS,
                message="Storage path exists",
                details={"path": str(storage_path.absolute())}
            )
        else:
            return CheckResult(
                service=f"Storage:{source_name}",
                status=CheckStatus.WARNING,
                message=f"Storage path does not exist: {base_path}",
                solution=f"Create directory: mkdir -p {base_path}"
            )
    
    def _check_s3(self, source_name: str, config: Dict[str, Any]) -> CheckResult:
        """S3 연결 체크."""
        try:
            import boto3
            
            # Create S3 client
            s3 = boto3.client('s3')
            
            # Try to list buckets
            buckets = s3.list_buckets()
            
            return CheckResult(
                service=f"S3:{source_name}",
                status=CheckStatus.SUCCESS,
                message="S3 connection successful",
                details={"bucket_count": len(buckets.get('Buckets', []))}
            )
            
        except ImportError:
            return CheckResult(
                service=f"S3:{source_name}",
                status=CheckStatus.WARNING,
                message="boto3 not installed",
                solution="Run: pip install boto3"
            )
        except Exception as e:
            return CheckResult(
                service=f"S3:{source_name}",
                status=CheckStatus.FAILED,
                message=f"S3 connection failed: {str(e)}",
                solution="Check AWS credentials and permissions"
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
                details={"bucket_count": len(buckets)}
            )
            
        except ImportError:
            return CheckResult(
                service=f"GCS:{source_name}",
                status=CheckStatus.WARNING,
                message="google-cloud-storage not installed",
                solution="Run: pip install google-cloud-storage"
            )
        except Exception as e:
            return CheckResult(
                service=f"GCS:{source_name}",
                status=CheckStatus.FAILED,
                message=f"GCS connection failed: {str(e)}",
                solution="Check GCP credentials and permissions"
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
                message=f"{output_type} output is disabled"
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
                    message=f"Unknown SQL type in connection URI: {connection_uri}"
                )
        
        else:
            return CheckResult(
                service=f"Output:{output_type}",
                status=CheckStatus.SKIPPED,
                message=f"Unknown output adapter type: {adapter_type}"
            )
    
    def check_feature_store(self) -> CheckResult:
        """Feature Store 체크."""
        fs_config = self.config["feature_store"]
        provider = fs_config.get("provider", "").lower()
        
        if provider == "feast":
            return self._check_feast(fs_config.get("feast_config", {}))
        elif provider == "tecton":
            return self._check_tecton(fs_config)
        else:
            return CheckResult(
                service="FeatureStore",
                status=CheckStatus.SKIPPED,
                message=f"Unknown feature store provider: {provider}"
            )
    
    def _check_feast(self, feast_config: Dict[str, Any]) -> CheckResult:
        """Feast Feature Store 체크."""
        registry_path = os.path.expandvars(feast_config.get("registry", "./feast_repo/registry.db"))
        
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
                            solution="pip install redis"
                        )
                    redis_conn = os.path.expandvars(online_store.get("connection_string", "localhost:6379"))
                    host, port = redis_conn.split(":")
                    r = redis.Redis(host=host, port=int(port), socket_connect_timeout=5)
                    r.ping()
                    
                return CheckResult(
                    service="Feast",
                    status=CheckStatus.SUCCESS,
                    message="Feast registry and online store accessible",
                    details={"registry": registry_path}
                )
            else:
                return CheckResult(
                    service="Feast",
                    status=CheckStatus.WARNING,
                    message=f"Feast registry not found: {registry_path}",
                    solution="Initialize Feast: feast init && feast apply"
                )
                
        except Exception as e:
            return CheckResult(
                service="Feast",
                status=CheckStatus.FAILED,
                message=f"Feast check failed: {str(e)}",
                solution="Check Feast configuration and Redis connection"
            )
    
    def _check_tecton(self, tecton_config: Dict[str, Any]) -> CheckResult:
        """Tecton Feature Store 체크."""
        tecton_url = os.path.expandvars(tecton_config.get("tecton_url", ""))
        
        if not tecton_url:
            return CheckResult(
                service="Tecton",
                status=CheckStatus.WARNING,
                message="Tecton URL not configured",
                solution="Set TECTON_URL environment variable"
            )
        
        try:
            try:
                import requests
            except ImportError:
                return CheckResult(
                    service="Feature Store (Tecton)",
                    status=CheckStatus.WARNING,
                    message="requests library not installed",
                    solution="pip install requests"
                )
            response = requests.get(f"{tecton_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                return CheckResult(
                    service="Tecton",
                    status=CheckStatus.SUCCESS,
                    message="Tecton service is healthy",
                    details={"url": tecton_url}
                )
            else:
                raise Exception(f"Health check returned {response.status_code}")
                
        except Exception as e:
            return CheckResult(
                service="Tecton",
                status=CheckStatus.FAILED,
                message=f"Tecton connection failed: {str(e)}",
                solution="Check Tecton URL and API key"
            )
    
    def check_artifact_store(self, artifact_store: Dict[str, Any]) -> CheckResult:
        """Artifact Storage 체크 (새로운 단일 artifact_store 구조)."""
        store_type = artifact_store.get("type", "")
        store_config = artifact_store.get("config", {})
        
        if store_type == "local":
            base_path = os.path.expandvars(store_config.get("base_path", "./mlruns/artifacts"))
            artifact_path = Path(base_path)
            
            if artifact_path.exists() or base_path in ["./mlruns/artifacts", "./artifacts"]:
                return CheckResult(
                    service="ArtifactStore:Local",
                    status=CheckStatus.SUCCESS,
                    message="Local artifact store accessible",
                    details={"path": str(artifact_path.absolute())}
                )
            else:
                return CheckResult(
                    service="ArtifactStore:Local",
                    status=CheckStatus.WARNING,
                    message=f"Artifact directory does not exist: {base_path}",
                    solution=f"Create directory: mkdir -p {base_path}"
                )
        
        elif store_type == "s3":
            bucket = os.path.expandvars(store_config.get("bucket", ""))
            return self._check_s3_bucket(bucket)
        
        elif store_type == "gcs":
            bucket = os.path.expandvars(store_config.get("bucket", ""))
            return self._check_gcs_bucket(bucket)
        
        else:
            return CheckResult(
                service="ArtifactStore",
                status=CheckStatus.SKIPPED,
                message=f"Unknown artifact store type: {store_type}"
            )
    
    def _check_s3_bucket(self, bucket: str) -> CheckResult:
        """S3 bucket 체크."""
        try:
            import boto3
            s3 = boto3.client('s3')
            s3.head_bucket(Bucket=bucket)
            
            return CheckResult(
                service="ArtifactStore:S3",
                status=CheckStatus.SUCCESS,
                message=f"S3 bucket accessible: {bucket}",
                details={"bucket": bucket}
            )
            
        except Exception as e:
            return CheckResult(
                service="ArtifactStore:S3",
                status=CheckStatus.FAILED,
                message=f"S3 bucket access failed: {str(e)}",
                solution=f"Check if bucket '{bucket}' exists and you have permissions"
            )
    
    def _check_gcs_bucket(self, bucket: str) -> CheckResult:
        """GCS bucket 체크."""
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket_obj = client.get_bucket(bucket)
            
            return CheckResult(
                service="ArtifactStore:GCS",
                status=CheckStatus.SUCCESS,
                message=f"GCS bucket accessible: {bucket}",
                details={"bucket": bucket}
            )
            
        except Exception as e:
            return CheckResult(
                service="ArtifactStore:GCS",
                status=CheckStatus.FAILED,
                message=f"GCS bucket access failed: {str(e)}",
                solution=f"Check if bucket '{bucket}' exists and you have permissions"
            )
    
    def check_serving(self) -> CheckResult:
        """API Serving 체크."""
        serving_config = self.config.get("serving", {})
        host = os.path.expandvars(serving_config.get("host", "0.0.0.0"))
        port = int(os.path.expandvars(str(serving_config.get("port", 8000))))
        
        # Just check if port is valid
        if 1 <= port <= 65535:
            return CheckResult(
                service="Serving",
                status=CheckStatus.SUCCESS,
                message="Serving configuration valid",
                details={"host": host, "port": port}
            )
        else:
            return CheckResult(
                service="Serving",
                status=CheckStatus.WARNING,
                message=f"Invalid port number: {port}",
                solution="Port must be between 1 and 65535"
            )
    
    def display_results(self, show_actionable: bool = False) -> None:
        """
        체크 결과를 표시.
        
        Args:
            show_actionable: 해결책 표시 여부
        """
        # Summary
        success_count = len([r for r in self.results if r.status == CheckStatus.SUCCESS])
        failed_count = len([r for r in self.results if r.status == CheckStatus.FAILED])
        warning_count = len([r for r in self.results if r.status == CheckStatus.WARNING])
        skipped_count = len([r for r in self.results if r.status == CheckStatus.SKIPPED])
        
        # Title
        self.ui.show_panel(
            f"Environment: {self.env_name}\n"
            f"Config: {self.config_path}",
            title="🔍 System Check Results",
            style="cyan"
        )
        
        # Results table
        headers = ["Service", "Status", "Message"]
        rows = []
        
        for result in self.results:
            status_icon = {
                CheckStatus.SUCCESS: "✅",
                CheckStatus.FAILED: "❌",
                CheckStatus.WARNING: "⚠️",
                CheckStatus.SKIPPED: "⏭️"
            }.get(result.status, "❓")
            
            rows.append([
                result.service,
                f"{status_icon} {result.status.value}",
                result.message[:50] + "..." if len(result.message) > 50 else result.message
            ])
        
        self.ui.show_table("Check Results", headers, rows)
        
        # Summary
        self.ui.print_divider()
        self.ui.console.print(f"""
Summary:
  ✅ Success: {success_count}
  ❌ Failed: {failed_count}
  ⚠️ Warning: {warning_count}
  ⏭️ Skipped: {skipped_count}
""")
        
        # Show actionable solutions if requested
        if show_actionable and (failed_count > 0 or warning_count > 0):
            self.ui.print_divider()
            self.ui.show_warning("Actionable Solutions:")
            
            for result in self.results:
                if result.status in [CheckStatus.FAILED, CheckStatus.WARNING] and result.solution:
                    self.ui.console.print(f"\n[bold]{result.service}:[/bold]")
                    self.ui.console.print(f"  💡 {result.solution}")
        
        # Overall status
        self.ui.print_divider()
        if failed_count == 0:
            if warning_count == 0:
                self.ui.show_success("All checks passed! System is ready.")
            else:
                self.ui.show_warning(f"System is operational with {warning_count} warning(s).")
        else:
            self.ui.show_error(f"System check failed with {failed_count} error(s).")