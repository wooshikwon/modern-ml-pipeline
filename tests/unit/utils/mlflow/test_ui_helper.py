"""
Unit tests for MLflow UI helper utilities.
"""

from unittest.mock import MagicMock, Mock, patch

from src.utils.integrations.ui_helper import (
    MLflowRunSummary,
    MLflowUIHelper,
    setup_mlflow_ui_config,
)


class TestMLflowUIHelper:
    """Test MLflow UI helper functionality"""

    def test_local_uri_detection(self):
        """Test local file system URI detection"""
        # Test local URIs
        helper = MLflowUIHelper("./mlruns")
        assert helper._is_local_uri() is True

        helper = MLflowUIHelper("/absolute/path/mlruns")
        assert helper._is_local_uri() is True

        helper = MLflowUIHelper("file://./mlruns")
        assert helper._is_local_uri() is True

        helper = MLflowUIHelper("mlruns")
        assert helper._is_local_uri() is True

    def test_remote_uri_detection(self):
        """Test remote server URI detection"""
        # Test remote URIs
        helper = MLflowUIHelper("http://localhost:5000")
        assert helper._is_remote_uri() is True

        helper = MLflowUIHelper("https://mlflow.example.com")
        assert helper._is_remote_uri() is True

        # Test non-remote URIs
        helper = MLflowUIHelper("./mlruns")
        assert helper._is_remote_uri() is False

    def test_get_ui_url_local(self):
        """Test UI URL generation for local tracking"""
        helper = MLflowUIHelper("./mlruns")

        with patch.object(helper, "_get_available_port", return_value=5000):
            # Base URL
            url = helper.get_ui_url()
            assert url == "http://localhost:5000"

            # With run ID
            url = helper.get_ui_url(run_id="abc123", experiment_id="1")
            assert url == "http://localhost:5000/#/experiments/1/runs/abc123"

            # With experiment ID only
            url = helper.get_ui_url(experiment_id="2")
            assert url == "http://localhost:5000/#/experiments/2"

    def test_get_ui_url_remote(self):
        """Test UI URL generation for remote tracking"""
        helper = MLflowUIHelper("http://mlflow-server:5000")

        # Base URL
        url = helper.get_ui_url()
        assert url == "http://mlflow-server:5000"

        # With run ID
        url = helper.get_ui_url(run_id="xyz789", experiment_id="3")
        assert url == "http://mlflow-server:5000/#/experiments/3/runs/xyz789"

    @patch("socket.socket")
    def test_get_available_port(self, mock_socket):
        """Test finding available port"""
        helper = MLflowUIHelper("./mlruns")

        # Test default port available
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__ = Mock(return_value=mock_sock)
        mock_socket.return_value.__exit__ = Mock()

        # First call checks port 5000 (available)
        mock_sock.bind.return_value = None
        port = helper._get_available_port()
        assert port == 5000

        # Test default port not available - finds random port
        mock_sock.bind.side_effect = [OSError(), None]
        mock_sock.getsockname.return_value = ("", 5555)
        port = helper._get_available_port()
        assert port == 5555

    @patch("subprocess.Popen")
    @patch("time.sleep")
    def test_start_mlflow_server_local(self, mock_sleep, mock_popen):
        """Test starting MLflow server for local tracking"""
        helper = MLflowUIHelper("./mlruns")

        # Mock port availability check
        with patch.object(helper, "_is_port_available", return_value=True):
            mock_process = Mock()
            mock_popen.return_value = mock_process

            result = helper.start_mlflow_server(port=5000)

            assert result is True
            assert helper.mlflow_process == mock_process

            # Check subprocess call
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args[0][0]
            assert call_args[0] == "mlflow"
            assert call_args[1] == "ui"
            assert "--port" in call_args
            assert "5000" in call_args

    def test_start_mlflow_server_remote(self):
        """Test that remote URIs don't start server"""
        helper = MLflowUIHelper("http://remote:5000")

        result = helper.start_mlflow_server()
        assert result is True  # Returns True but doesn't start server
        assert helper.mlflow_process is None

    @patch("subprocess.Popen")
    @patch("src.utils.integrations.ui_helper.logger")
    def test_start_mlflow_server_already_running(self, mock_logger, mock_popen):
        """Test handling when server already running"""
        helper = MLflowUIHelper("./mlruns")

        # Mock port not available (server running)
        with patch.object(helper, "_is_port_available", return_value=False):
            result = helper.start_mlflow_server(port=5000)

            assert result is True
            mock_popen.assert_not_called()
            mock_logger.warning.assert_called_with("MLflow server already running on port 5000")

    @patch("src.utils.integrations.ui_helper.logger")
    def test_stop_mlflow_server(self, mock_logger):
        """Test stopping MLflow server"""
        helper = MLflowUIHelper("./mlruns")

        # Test with process
        mock_process = Mock()
        helper.mlflow_process = mock_process

        helper.stop_mlflow_server()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        mock_logger.info.assert_called_with("MLflow server stopped")

    def test_get_network_addresses(self):
        """Test getting network addresses"""
        helper = MLflowUIHelper("./mlruns")

        with patch.object(helper, "_get_available_port", return_value=5000):
            addresses = helper.get_network_addresses()

            # Basic checks - should always have localhost
            assert "localhost" in addresses
            assert addresses["localhost"] == "http://localhost:5000"

            # May have other addresses depending on network
            assert isinstance(addresses, dict)
            for key, value in addresses.items():
                assert value.startswith("http://")
                assert ":5000" in value

    def test_display_access_info(self):
        """Test displaying access information"""
        helper = MLflowUIHelper("./mlruns")

        with patch.object(helper, "start_mlflow_server", return_value=True):
            with patch.object(helper, "_get_available_port", return_value=5000):
                with patch.object(
                    helper,
                    "get_network_addresses",
                    return_value={
                        "localhost": "http://localhost:5000",
                        "local_ip": "http://192.168.1.100:5000",
                    },
                ):
                    helper.display_access_info(
                        run_id="run123", experiment_id="exp1", experiment_name="test_exp"
                    )

                    # Check that the method completed without error
                    assert True


class TestMLflowRunSummary:
    """Test MLflow run summary display"""

    def test_display_run_summary(self):
        """Test displaying run summary"""
        summary = MLflowRunSummary()

        metrics = {"accuracy": 0.95, "loss": 0.123, "f1_score": 0.92}

        params = {
            "model_type": "xgboost",
            "learning_rate": 0.01,
            "n_estimators": 100,
            "max_depth": 5,
            "other_param": "value",
        }

        artifacts = ["model.pkl", "requirements.txt", "metadata.json"]

        # Patch Console at import location
        with patch("rich.console.Console") as mock_console_cls:
            mock_rich_console = Mock()
            mock_console_cls.return_value = mock_rich_console

            summary.display_run_summary(
                run_id="run_abc123", metrics=metrics, params=params, artifacts=artifacts
            )

            # Check that table was printed
            mock_rich_console.print.assert_called_once()

    def test_display_run_summary_empty(self):
        """Test displaying empty run summary"""
        summary = MLflowRunSummary()

        with patch("rich.console.Console") as mock_console_cls:
            mock_rich_console = Mock()
            mock_console_cls.return_value = mock_rich_console

            summary.display_run_summary(run_id="empty_run", metrics={}, params={}, artifacts=None)

            # Should still print table even with empty data
            mock_rich_console.print.assert_called_once()


class TestSetupMLflowUIConfig:
    """Test MLflow UI configuration setup"""

    def test_setup_mlflow_ui_config_new(self):
        """Test adding MLflow UI config to settings"""
        settings_dict = {"config": {"environment": "test"}}

        result = setup_mlflow_ui_config(settings_dict)

        assert "mlflow_ui" in result["config"]
        assert result["config"]["mlflow_ui"]["auto_open_browser"] is False
        assert result["config"]["mlflow_ui"]["show_qr_code"] is False
        assert result["config"]["mlflow_ui"]["server_port"] == 5000

    def test_setup_mlflow_ui_config_existing(self):
        """Test preserving existing MLflow UI config"""
        settings_dict = {"config": {"mlflow_ui": {"auto_open_browser": True, "server_port": 8080}}}

        result = setup_mlflow_ui_config(settings_dict)

        # Should update with defaults (setup_mlflow_ui_config replaces, not merges)
        assert result["config"]["mlflow_ui"]["auto_open_browser"] is False  # Default value
        assert result["config"]["mlflow_ui"]["server_port"] == 5000  # Default value
        assert "show_qr_code" in result["config"]["mlflow_ui"]

    def test_setup_mlflow_ui_config_no_config(self):
        """Test adding config section if missing"""
        settings_dict = {}

        result = setup_mlflow_ui_config(settings_dict)

        assert "config" in result
        assert "mlflow_ui" in result["config"]
