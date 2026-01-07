from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

import pandas as pd
from mlflow.tracking import MlflowClient


class MLflowTestContext:
    """Minimal-contract MLflow context for tests.
    - Provides: settings, data_path, tracking_uri, experiment_name
    - Helpers (required): experiment_exists, get_experiment_run_count, get_run_metrics
    - Helpers (optional): verify_mlflow_artifacts
    """

    def __init__(
        self, isolated_temp_directory, settings_builder, test_data_generator, seed: int = 42
    ):
        self.temp_dir = isolated_temp_directory
        self.settings_builder = settings_builder
        self.data_generator = test_data_generator
        self.seed = seed

    def for_classification(
        self, experiment: str, model: str = "RandomForestClassifier"
    ) -> "_MLflowContextManager":
        return _MLflowContextManager(
            task="classification",
            experiment_suffix=experiment,
            model_class=f"sklearn.ensemble.{model}",
            context=self,
        )

    def for_regression(
        self, experiment: str, model: str = "RandomForestRegressor"
    ) -> "_MLflowContextManager":
        return _MLflowContextManager(
            task="regression",
            experiment_suffix=experiment,
            model_class=f"sklearn.ensemble.{model}",
            context=self,
        )


class _MLflowContextManager:
    def __init__(
        self, task: str, experiment_suffix: str, model_class: str, context: MLflowTestContext
    ):
        self.task = task
        self.experiment_suffix = experiment_suffix
        self.model_class = model_class
        self.context = context
        self.mlflow_client: Optional[MlflowClient] = None
        self.experiment_id: Optional[str] = None
        self.settings = None
        self.data_path = None
        self.experiment_name = None
        self.mlflow_uri = None

    def __enter__(self) -> "_MLflowContextManager":
        # 1) MLflow URI 표준화
        self.mlflow_uri = f"file://{self.context.temp_dir}/mlruns"
        # 2) 실험명은 uuid 기반
        self.experiment_name = f"{self.experiment_suffix}-{uuid4().hex[:8]}"

        # 3) 결정론적 데이터 생성
        X, y = self.context.data_generator.classification_data(
            n_samples=50, n_features=4, random_state=self.context.seed
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
        df["target"] = y

        self.data_path = self.context.temp_dir / f"data_{self.experiment_suffix}.csv"
        df.to_csv(self.data_path, index=False)

        # 4) Settings 자동 구성
        self.settings = (
            self.context.settings_builder.with_task(self.task)
            .with_model(self.model_class)
            .with_data_path(str(self.data_path))
            .with_mlflow(self.mlflow_uri, self.experiment_name)
            .build()
        )

        # 5) MLflow client 준비 및 experiment id 확보
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_uri)
        exp = self.mlflow_client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            self.experiment_id = self.mlflow_client.create_experiment(self.experiment_name)
        else:
            self.experiment_id = exp.experiment_id

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Clean up any MLflow server processes that might have been started
        self._cleanup_mlflow_processes()
        # temp_dir cleanup is handled by fixture
        return None

    def _cleanup_mlflow_processes(self) -> None:
        """Clean up any MLflow server processes to prevent memory leaks"""
        import os
        import signal
        import subprocess

        try:
            # Find MLflow server processes
            result = subprocess.run(
                ["pgrep", "-f", "mlflow.server"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    try:
                        pid_int = int(pid.strip())
                        os.kill(pid_int, signal.SIGTERM)
                    except (ValueError, ProcessLookupError, PermissionError):
                        # Process already dead or permission denied
                        pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # pgrep not available or timeout - skip cleanup
            pass

    # ===== Helpers (required) =====
    def experiment_exists(self) -> bool:
        assert self.mlflow_client is not None
        return self.mlflow_client.get_experiment_by_name(self.experiment_name) is not None

    def get_experiment_run_count(self) -> int:
        assert self.mlflow_client is not None and self.experiment_id is not None
        runs = self.mlflow_client.search_runs([self.experiment_id])
        return len(runs)

    def get_run_metrics(self) -> dict[str, Any]:
        assert self.mlflow_client is not None and self.experiment_id is not None
        runs = self.mlflow_client.search_runs(
            [self.experiment_id], max_results=1, order_by=["attributes.start_time DESC"]
        )
        if not runs:
            return {}
        return dict(runs[0].data.metrics)

    # ===== Helpers (optional) =====
    def verify_mlflow_artifacts(self) -> bool:
        # Placeholder: keep as optional helper; concrete checks can be added in pilot phase
        # e.g., verify model artifact directory exists and contains signature/schema
        return True
