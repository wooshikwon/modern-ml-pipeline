"""
E2E: Quantile Regression + Monitoring 전체 사이클

로컬 CSV + file-based MLflow로 외부 의존성 없이 실행.
train → baseline 저장 확인 → inference → drift 메트릭 확인.
"""

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from mmp.pipelines.train_pipeline import run_train_pipeline


@pytest.fixture
def quantile_monitoring_env(isolated_temp_directory, settings_builder):
    """Quantile + Monitoring E2E 테스트 환경 구성."""
    # 1. 테스트 데이터 생성
    np.random.seed(42)
    n = 300
    data = pd.DataFrame({
        "entity_id": range(n),
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "feat_3": np.random.randn(n),
    })
    data["target"] = 3 * data["feat_1"] + 2 * data["feat_2"] + np.random.randn(n) * 0.5

    data_path = isolated_temp_directory / "data.csv"
    data.to_csv(data_path, index=False)

    # 2. MLflow tracking URI
    mlruns_dir = isolated_temp_directory / "mlruns"
    mlruns_dir.mkdir()
    tracking_uri = f"file://{mlruns_dir}"

    # 3. Settings 구성
    settings = (
        settings_builder
        .with_task("regression")
        .with_model(
            "mmp.models.custom.quantile_ensemble.QuantileRegressorEnsemble",
            hyperparameters={
                "base_class_path": "lightgbm.LGBMRegressor",
                "quantiles": [0.5, 0.75, 0.95],
                "verbose": -1,
                "n_estimators": 20,
            },
        )
        .with_data_path(str(data_path))
        .with_target_column("target")
        .with_mlflow(tracking_uri, "test_quantile_monitoring")
        .build()
    )

    # 4. Monitoring 활성화
    from mmp.settings.recipe import MonitoringConfig
    settings.recipe.monitoring = MonitoringConfig(enabled=True)

    return settings, tracking_uri, data_path


class TestQuantileMonitoringE2E:
    """Quantile Regression + Monitoring 전체 사이클 E2E 테스트."""

    def test_train_saves_baseline_and_quantile_metrics(self, quantile_monitoring_env):
        """학습: quantile 메트릭 + monitoring baseline이 MLflow에 저장되는지."""
        settings, tracking_uri, _ = quantile_monitoring_env

        result = run_train_pipeline(settings, record_requirements=False)
        assert result.run_id is not None

        # MLflow에서 run 조회
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(result.run_id)
        metrics = run.data.metrics

        # Quantile 메트릭 확인
        assert "pinball_loss_p50" in metrics
        assert "pinball_loss_p75" in metrics
        assert "pinball_loss_p95" in metrics
        assert "mean_pinball_loss" in metrics
        assert "interval_coverage" in metrics
        assert metrics["mean_pinball_loss"] > 0

        # p50 기준 standard 메트릭
        assert "r2_score" in metrics
        assert "root_mean_squared_error" in metrics

        # Monitoring baseline artifact 확인
        client = mlflow.tracking.MlflowClient()
        baseline_path = client.download_artifacts(result.run_id, "monitoring/baseline.json")
        with open(baseline_path) as f:
            baseline = json.load(f)

        # 네임스페이스 구조 확인
        assert "DataDriftMonitor" in baseline
        assert "PredictionDriftMonitor" in baseline

        # DataDriftMonitor baseline 구조
        data_baseline = baseline["DataDriftMonitor"]
        assert "numerical" in data_baseline
        assert "meta" in data_baseline
        assert data_baseline["meta"]["row_count"] > 0

        # PredictionDriftMonitor baseline 구조 (quantile → multi-column)
        pred_baseline = baseline["PredictionDriftMonitor"]
        assert "prediction" in pred_baseline
        pred_keys = list(pred_baseline["prediction"].keys())
        assert any("pred_p" in k for k in pred_keys)

    def test_standard_regression_with_monitoring(self, isolated_temp_directory, settings_builder):
        """기존 회귀 모델 + monitoring이 기존 동작을 깨지 않는지."""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            "entity_id": range(n),
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
        })
        data["target"] = 2 * data["feat_1"] + np.random.randn(n) * 0.3

        data_path = isolated_temp_directory / "reg_data.csv"
        data.to_csv(data_path, index=False)

        mlruns_dir = isolated_temp_directory / "mlruns_reg"
        mlruns_dir.mkdir()
        tracking_uri = f"file://{mlruns_dir}"

        settings = (
            settings_builder
            .with_task("regression")
            .with_model(
                "sklearn.ensemble.RandomForestRegressor",
                hyperparameters={"n_estimators": 10, "random_state": 42},
            )
            .with_data_path(str(data_path))
            .with_target_column("target")
            .with_mlflow(tracking_uri, "test_reg_monitoring")
            .build()
        )

        from mmp.settings.recipe import MonitoringConfig
        settings.recipe.monitoring = MonitoringConfig(enabled=True)

        result = run_train_pipeline(settings, record_requirements=False)
        assert result.run_id is not None

        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(result.run_id)

        # Standard 메트릭 (quantile이 아님)
        assert "r2_score" in run.data.metrics
        assert "mean_squared_error" in run.data.metrics
        assert "mean_pinball_loss" not in run.data.metrics  # quantile 아님

        # Baseline은 저장됨
        client = mlflow.tracking.MlflowClient()
        baseline_path = client.download_artifacts(result.run_id, "monitoring/baseline.json")
        with open(baseline_path) as f:
            baseline = json.load(f)
        assert "DataDriftMonitor" in baseline

    def test_no_monitoring_leaves_no_artifacts(self, isolated_temp_directory, settings_builder):
        """monitoring 비활성 시 baseline artifact가 생성되지 않는지."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            "entity_id": range(n),
            "feat_1": np.random.randn(n),
        })
        data["target"] = data["feat_1"] + np.random.randn(n) * 0.1

        data_path = isolated_temp_directory / "no_mon_data.csv"
        data.to_csv(data_path, index=False)

        mlruns_dir = isolated_temp_directory / "mlruns_nomon"
        mlruns_dir.mkdir()
        tracking_uri = f"file://{mlruns_dir}"

        settings = (
            settings_builder
            .with_task("regression")
            .with_model(
                "sklearn.linear_model.LinearRegression",
                hyperparameters={"fit_intercept": True},
            )
            .with_data_path(str(data_path))
            .with_target_column("target")
            .with_mlflow(tracking_uri, "test_no_monitoring")
            .build()
        )
        # monitoring 설정 없음 (기본값 None)

        result = run_train_pipeline(settings, record_requirements=False)

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        try:
            client.download_artifacts(result.run_id, "monitoring/baseline.json")
            assert False, "baseline.json should not exist"
        except Exception:
            pass  # expected: artifact 없음
