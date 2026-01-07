"""
Unit tests for SystemChecker
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.cli.utils.system_checker import CheckStatus, SystemChecker


class TestSystemChecker:
    """SystemChecker 테스트."""

    @pytest.fixture
    def basic_config(self):
        """기본 설정 fixture."""
        return {
            "environment": {"name": "test"},
            "mlflow": {"tracking_uri": "./mlruns", "experiment_name": "test-exp"},
            "data_source": {
                "name": "Local Files",
                "adapter_type": "storage",
                "config": {"base_path": "./data"},
            },
            "feature_store": {"provider": "none"},
            "output": {
                "inference": {
                    "enabled": True,
                    "adapter_type": "storage",
                    "config": {"base_path": "./artifacts/predictions"},
                }
            },
            "serving": {"enabled": False},
        }

    def test_init(self, basic_config):
        """초기화 테스트."""
        checker = SystemChecker(basic_config, "test")
        assert checker.env_name == "test"
        assert checker.config == basic_config
        assert checker.results == []

    def test_check_mlflow_local_success(self, basic_config, tmp_path):
        """MLflow 로컬 체크 성공 테스트."""
        checker = SystemChecker(basic_config, "test")

        # Mock Path.exists to return True
        with patch.object(Path, "exists", return_value=True):
            result = checker.check_mlflow()

        assert result.status == CheckStatus.SUCCESS
        assert "MLflow local tracking directory is accessible" in result.message

    def test_check_mlflow_http_success(self, basic_config):
        """MLflow HTTP 체크 성공 테스트."""
        basic_config["mlflow"]["tracking_uri"] = "http://localhost:5000"
        checker = SystemChecker(basic_config, "test")

        # Mock requests
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = checker.check_mlflow()

        assert result.status == CheckStatus.SUCCESS
        assert "MLflow server is healthy" in result.message

    def test_check_mlflow_http_failed(self, basic_config):
        """MLflow HTTP 체크 실패 테스트."""
        basic_config["mlflow"]["tracking_uri"] = "http://localhost:5000"
        checker = SystemChecker(basic_config, "test")

        # Mock requests to fail
        with patch("requests.get", side_effect=Exception("Connection refused")):
            result = checker.check_mlflow()

        assert result.status == CheckStatus.FAILED
        assert "MLflow connection failed" in result.message

    def test_check_data_source_storage(self, basic_config, tmp_path):
        """Storage 데이터 소스 체크 테스트."""
        checker = SystemChecker(basic_config, "test")

        with patch.object(Path, "exists", return_value=True):
            result = checker.check_data_source(basic_config["data_source"])

        assert result.status == CheckStatus.SUCCESS
        assert "Storage path exists" in result.message

    def test_check_data_source_postgresql(self, basic_config):
        """PostgreSQL 데이터 소스 체크 테스트."""
        basic_config["data_source"] = {
            "name": "PostgreSQL",
            "adapter_type": "sql",
            "config": {"connection_uri": "postgresql://user:pass@localhost:5432/db"},
        }
        checker = SystemChecker(basic_config, "test")

        # Mock psycopg2
        with patch("psycopg2.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            result = checker.check_data_source(basic_config["data_source"])

        assert result.status == CheckStatus.SUCCESS
        assert "PostgreSQL connection successful" in result.message

    def test_check_data_source_bigquery(self, basic_config):
        """BigQuery 데이터 소스 체크 테스트."""
        basic_config["data_source"] = {
            "name": "BigQuery",
            "adapter_type": "sql",  # 새로운 구조에서는 sql
            "config": {
                "connection_uri": "bigquery://project/dataset",
                "project_id": "test-project",
                "dataset_id": "test-dataset",
            },
        }
        checker = SystemChecker(basic_config, "test")

        # Mock google.cloud.bigquery
        with patch("google.cloud.bigquery.Client") as mock_client_class:
            mock_client = Mock()
            # list_datasets가 iterable을 반환하도록 Mock
            mock_client.list_datasets.return_value = iter([])
            mock_client_class.return_value = mock_client

            result = checker.check_data_source(basic_config["data_source"])

        assert result.status == CheckStatus.SUCCESS
        assert "BigQuery project accessible" in result.message

    def test_check_data_source_s3(self, basic_config):
        """S3 데이터 소스 체크 테스트."""
        basic_config["data_source"] = {
            "name": "S3",
            "adapter_type": "storage",
            "config": {"base_path": "s3://bucket/prefix"},
        }
        checker = SystemChecker(basic_config, "test")

        # Mock boto3
        with patch("boto3.client") as mock_client:
            mock_s3 = Mock()
            mock_s3.list_buckets.return_value = {"Buckets": [{"Name": "bucket1"}]}
            mock_client.return_value = mock_s3

            result = checker.check_data_source(basic_config["data_source"])

        assert result.status == CheckStatus.SUCCESS
        assert "S3 connection successful" in result.message

    def test_check_output_target_enabled(self, basic_config):
        """Output target 체크 - 활성화 상태."""
        checker = SystemChecker(basic_config, "test")

        with patch.object(Path, "exists", return_value=True):
            result = checker.check_output_target("inference", basic_config["output"]["inference"])

        assert result.status == CheckStatus.SUCCESS

    def test_check_output_target_disabled(self, basic_config):
        """Output target 체크 - 비활성화 상태."""
        basic_config["output"]["inference"]["enabled"] = False
        checker = SystemChecker(basic_config, "test")

        result = checker.check_output_target("inference", basic_config["output"]["inference"])

        assert result.status == CheckStatus.SKIPPED
        assert "inference output is disabled" in result.message

    def test_check_output_target_sql(self, basic_config):
        """SQL Output target 체크."""
        basic_config["output"]["inference"] = {
            "enabled": True,
            "adapter_type": "sql",
            "config": {
                "connection_uri": "postgresql://user:pass@localhost:5432/db",
                "table": "predictions",
            },
        }
        checker = SystemChecker(basic_config, "test")

        with patch("psycopg2.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            result = checker.check_output_target("inference", basic_config["output"]["inference"])

        assert result.status == CheckStatus.SUCCESS

    def test_check_feature_store_none(self, basic_config):
        """Feature Store none 체크."""
        checker = SystemChecker(basic_config, "test")
        result = checker.check_feature_store()

        assert result.status == CheckStatus.SKIPPED
        assert "Feature store not configured" in result.message

    def test_check_feature_store_feast(self, basic_config):
        """Feast Feature Store 체크."""
        basic_config["feature_store"] = {
            "provider": "feast",
            "feast_config": {
                "registry": "./feast_repo/registry.db",
                "online_store": {"type": "sqlite"},
            },
        }
        checker = SystemChecker(basic_config, "test")

        with patch.object(Path, "exists", return_value=True):
            result = checker.check_feature_store()

        assert result.status == CheckStatus.SUCCESS
        assert "Feast registry and online store accessible" in result.message

    def test_check_serving_valid(self, basic_config):
        """Serving 체크 - 유효한 설정."""
        basic_config["serving"] = {"enabled": True, "host": "0.0.0.0", "port": 8000}
        checker = SystemChecker(basic_config, "test")

        result = checker.check_serving()

        assert result.status == CheckStatus.SUCCESS
        assert "Serving configuration valid" in result.message

    def test_check_serving_invalid_port(self, basic_config):
        """Serving 체크 - 유효하지 않은 포트."""
        basic_config["serving"] = {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 70000,  # Invalid port
        }
        checker = SystemChecker(basic_config, "test")

        result = checker.check_serving()

        assert result.status == CheckStatus.WARNING
        assert "Invalid port number" in result.message

    def test_run_all_checks(self, basic_config):
        """전체 체크 실행 테스트."""
        checker = SystemChecker(basic_config, "test")

        with patch.object(Path, "exists", return_value=True):
            results = checker.run_all_checks()

        # New format: Dict[str, Dict[str, Any]]
        assert isinstance(results, dict)
        assert len(results) > 0

        # Check that key services were checked
        service_names = list(results.keys())
        assert any("MLflow" in s for s in service_names)
        assert any("Storage" in s for s in service_names)
        # Output checks use the underlying adapter service names (e.g., "Storage:inference_output")
        assert any("inference_output" in s for s in service_names)

        # Verify structure of each result
        for service_name, result in results.items():
            assert "status" in result
            assert "message" in result
            assert "details" in result
            assert "solution" in result
            assert result["status"] in ["success", "failed", "warning", "skipped"]

    # test_display_results removed - display functionality moved to console_manager
    # SystemChecker now focuses only on checking, not displaying

    def test_no_preprocessed_output_check(self, basic_config):
        """Preprocessed output이 체크되지 않는지 확인."""
        # Add preprocessed config (should be ignored)
        basic_config["output"]["preprocessed"] = {
            "enabled": True,
            "adapter_type": "storage",
            "config": {"base_path": "./artifacts/preprocessed"},
        }

        checker = SystemChecker(basic_config, "test")

        with patch.object(Path, "exists", return_value=True):
            results = checker.run_all_checks()

        # Check that preprocessed output was NOT checked
        service_names = list(results.keys())
        assert not any("preprocessed" in s.lower() for s in service_names)
        # But inference output should be checked (using underlying adapter naming)
        assert any("inference_output" in s for s in service_names)
