"""
MLflow UI Helper Utilities
Provides easy access to MLflow web UI after training
"""

import socket
import subprocess
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.utils.core.logger import logger


class MLflowUIHelper:
    """Helper class for MLflow UI access and visualization"""

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self.mlflow_process = None

    def get_ui_url(self, run_id: Optional[str] = None, experiment_id: Optional[str] = None) -> str:
        """
        Get the appropriate MLflow UI URL based on tracking URI

        Args:
            run_id: Optional specific run ID to link to
            experiment_id: Optional experiment ID to link to

        Returns:
            Complete URL to access MLflow UI
        """
        # Parse tracking URI to determine type
        if self._is_local_uri():
            # Local file system - need to start MLflow server
            host = "localhost"
            port = self._get_available_port()
            base_url = f"http://{host}:{port}"
        elif self._is_remote_uri():
            # Remote MLflow server
            parsed = urlparse(self.tracking_uri)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
        else:
            # Default to localhost
            base_url = "http://localhost:5000"

        # Build specific URL
        if run_id and experiment_id:
            return f"{base_url}/#/experiments/{experiment_id}/runs/{run_id}"
        elif experiment_id:
            return f"{base_url}/#/experiments/{experiment_id}"
        else:
            return base_url

    def _is_local_uri(self) -> bool:
        """Check if tracking URI is local file system"""
        return (
            self.tracking_uri.startswith("./")
            or self.tracking_uri.startswith("/")
            or self.tracking_uri.startswith("file://")
            or self.tracking_uri == "mlruns"
        )

    def _is_remote_uri(self) -> bool:
        """Check if tracking URI is remote server"""
        return self.tracking_uri.startswith("http://") or self.tracking_uri.startswith("https://")

    def _get_available_port(self) -> int:
        """Find an available port for MLflow server"""
        # Try default port first
        if self._is_port_available(5000):
            return 5000

        # Find random available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except:
                return False

    def start_mlflow_server(self, port: int = 5000) -> bool:
        """
        Start MLflow tracking server for local URI

        Args:
            port: Port to run server on

        Returns:
            True if server started successfully
        """
        if not self._is_local_uri():
            return True  # No need to start for remote URI

        # Check if server already running
        if not self._is_port_available(port):
            logger.warning(f"MLflow server already running on port {port}")
            return True

        try:
            # Determine the backend store URI
            if self.tracking_uri in ["./mlruns", "mlruns"]:
                backend_store = "./mlruns"
            else:
                backend_store = self.tracking_uri

            # Start MLflow server as subprocess
            cmd = [
                "mlflow",
                "ui",
                "--backend-store-uri",
                backend_store,
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ]

            self.mlflow_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            # Wait for server to start
            time.sleep(2)

            logger.info(f"MLflow server started on port {port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start MLflow server: {e}")
            return False

    def stop_mlflow_server(self):
        """Stop the MLflow server if it was started"""
        if self.mlflow_process:
            self.mlflow_process.terminate()
            self.mlflow_process.wait()
            logger.info("MLflow server stopped")

    def get_network_addresses(self) -> Dict[str, str]:
        """
        Get all network addresses for accessing MLflow UI

        Returns:
            Dictionary of network interfaces and their URLs
        """
        addresses = {}
        port = self._get_available_port()

        # Localhost
        addresses["localhost"] = f"http://localhost:{port}"

        # Get machine hostname
        try:
            hostname = socket.gethostname()
            addresses["hostname"] = f"http://{hostname}:{port}"
        except:
            pass

        # Get IP addresses
        try:
            # Get primary IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            addresses["local_ip"] = f"http://{local_ip}:{port}"

            # Get all IPs
            hostname = socket.gethostname()
            all_ips = socket.gethostbyname_ex(hostname)[2]
            for i, ip in enumerate(all_ips):
                if ip != local_ip:
                    addresses[f"ip_{i}"] = f"http://{ip}:{port}"
        except:
            pass

        return addresses

    def display_access_info(self, run_id: str, experiment_id: str, experiment_name: str):
        """
        Display MLflow UI access information

        Args:
            run_id: The MLflow run ID
            experiment_id: The experiment ID
            experiment_name: The experiment name
        """
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Get URLs
        base_url = self.get_ui_url()
        run_url = self.get_ui_url(run_id=run_id, experiment_id=experiment_id)

        # Start server if local
        if self._is_local_uri():
            port = self._get_available_port()
            self.start_mlflow_server(port)
            base_url = f"http://localhost:{port}"
            run_url = f"{base_url}/#/experiments/{experiment_id}/runs/{run_id}"

        # Create access info table
        table = Table(title="🔗 MLflow UI Access", box=box.ROUNDED)
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("URL", style="bright_blue")

        table.add_row("Dashboard", base_url)
        table.add_row("Experiment", f"{base_url}/#/experiments/{experiment_id}")
        table.add_row("This Run", run_url)

        # Add network addresses for remote access
        if self._is_local_uri():
            addresses = self.get_network_addresses()
            if "local_ip" in addresses:
                table.add_row("Remote Access", addresses["local_ip"])

        console.print(table)

        # Show instructions
        instructions = []
        if self._is_local_uri():
            instructions.append("• MLflow server is running locally")
            instructions.append(f"• Access the UI at: [bright_blue]{base_url}[/bright_blue]")
            instructions.append("• Press Ctrl+C to stop the server")
        else:
            instructions.append(f"• MLflow server: [bright_blue]{self.tracking_uri}[/bright_blue]")
            instructions.append(f"• Direct link to this run: [bright_blue]{run_url}[/bright_blue]")

        console.print(Panel("\n".join(instructions), title="ℹ️ Instructions", border_style="blue"))


class MLflowRunSummary:
    """Generate and display MLflow run summary"""

    def __init__(self):
        pass

    def display_run_summary(
        self,
        run_id: str,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        artifacts: Optional[list] = None,
    ):
        """
        Display a summary of the MLflow run

        Args:
            run_id: The MLflow run ID
            metrics: Dictionary of metrics
            params: Dictionary of parameters
            artifacts: List of artifact paths
        """
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Create summary table
        table = Table(title=f"📊 Run Summary: {run_id[:8]}...", box=box.ROUNDED)
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Details", style="white")

        # Add metrics
        if metrics:
            metrics_str = "\n".join(
                [
                    f"• {k}: {v:.4f}" if isinstance(v, float) else f"• {k}: {v}"
                    for k, v in metrics.items()
                ]
            )
            table.add_row("Metrics", metrics_str)

        # Add key parameters
        if params:
            # Show only important params
            important_params = {
                k: v
                for k, v in params.items()
                if k in ["model_type", "learning_rate", "n_estimators", "max_depth"]
            }
            if important_params:
                params_str = "\n".join([f"• {k}: {v}" for k, v in important_params.items()])
                table.add_row("Key Parameters", params_str)

        # Add artifacts
        if artifacts:
            artifacts_str = "\n".join([f"• {a}" for a in artifacts[:5]])  # Show first 5
            if len(artifacts) > 5:
                artifacts_str += f"\n• ... and {len(artifacts) - 5} more"
            table.add_row("Artifacts", artifacts_str)

        console.print(table)


def setup_mlflow_ui_config(settings_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add MLflow UI configuration to settings

    Args:
        settings_dict: Settings dictionary

    Returns:
        Updated settings with MLflow UI config
    """
    mlflow_ui_config = {
        "mlflow_ui": {
            "auto_open_browser": False,  # Can be overridden by CLI flag
            "show_qr_code": False,  # For remote access from mobile
            "server_port": 5000,  # Default MLflow UI port
            "server_host": "0.0.0.0",  # Allow external connections
            "keep_server_running": False,  # Keep server running after script ends
        }
    }

    # Merge with existing config
    if "config" not in settings_dict:
        settings_dict["config"] = {}
    settings_dict["config"].update(mlflow_ui_config)

    return settings_dict
