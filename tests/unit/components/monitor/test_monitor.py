"""
Monitor Component Tests

DataDriftMonitor / PredictionDriftMonitor 유닛 테스트.
PSI 계산, baseline/evaluate 흐름, alert 판정을 검증한다.
"""

import numpy as np
import pandas as pd

from mmp.components.monitor.base import Alert, MonitorReport
from mmp.components.monitor.modules.data_drift import (
    DataDriftMonitor,
    _compute_js_divergence,
)
from mmp.components.monitor.modules.prediction_drift import PredictionDriftMonitor
from mmp.components.monitor.utils import compute_psi


class TestMonitorReport:
    """MonitorReport 동작 테스트"""

    def test_status_ok_when_no_alerts(self):
        report = MonitorReport()
        assert report.status == "ok"

    def test_status_warning(self):
        report = MonitorReport(alerts=[
            Alert("data_drift", "feat", "psi", 0.15, 0.1, "warning", "msg"),
        ])
        assert report.status == "warning"

    def test_status_alert_overrides_warning(self):
        report = MonitorReport(alerts=[
            Alert("data_drift", "feat1", "psi", 0.15, 0.1, "warning", "msg"),
            Alert("data_drift", "feat2", "psi", 0.25, 0.2, "alert", "msg"),
        ])
        assert report.status == "alert"

    def test_to_dict_serializable(self):
        import json
        report = MonitorReport(
            metrics={"psi_feat1": 0.05},
            alerts=[Alert("data_drift", "feat1", "psi", 0.05, 0.2, "warning", "msg")],
        )
        result = report.to_dict()
        json.dumps(result)  # should not raise
        assert result["status"] == "warning"
        assert result["alert_count"] == 1

    def test_merge_combines_reports(self):
        r1 = MonitorReport(metrics={"a": 1.0}, alerts=[])
        r2 = MonitorReport(
            metrics={"b": 2.0},
            alerts=[Alert("test", "f", "m", 0.5, 0.3, "alert", "msg")],
        )
        r1.merge(r2)
        assert "a" in r1.metrics and "b" in r1.metrics
        assert len(r1.alerts) == 1
        assert r1.status == "alert"


class TestPSI:
    """PSI 계산 정확성 테스트"""

    def test_identical_distributions(self):
        np.random.seed(42)
        data = np.random.randn(1000)
        counts, bin_edges = np.histogram(data, bins=10)
        proportions = (counts / counts.sum()).tolist()

        psi = compute_psi(data, bin_edges.tolist(), proportions)
        assert psi < 0.01  # nearly identical → PSI ≈ 0

    def test_shifted_distribution(self):
        np.random.seed(42)
        baseline = np.random.randn(1000)
        counts, bin_edges = np.histogram(baseline, bins=10)
        proportions = (counts / counts.sum()).tolist()

        shifted = baseline + 3.0  # large shift
        psi = compute_psi(shifted, bin_edges.tolist(), proportions)
        assert psi > 0.2  # significant drift

    def test_empty_values(self):
        psi = compute_psi(np.array([]), [0.0, 1.0, 2.0], [0.5, 0.5])
        assert psi == 0.0


class TestJSDivergence:
    """JS divergence 계산 테스트"""

    def test_identical_categories(self):
        series = pd.Series(["A", "A", "B", "B"])
        ref = {"A": 0.5, "B": 0.5}
        js = _compute_js_divergence(series, ref)
        assert js < 0.01

    def test_different_categories(self):
        series = pd.Series(["A", "A", "A", "A"])
        ref = {"A": 0.25, "B": 0.75}
        js = _compute_js_divergence(series, ref)
        assert js > 0.1


class TestDataDriftMonitor:
    """DataDriftMonitor baseline + evaluate 테스트"""

    def _make_settings(self, settings_builder, monitoring_enabled=True):
        settings = settings_builder.with_task("regression").build()
        # Inject monitoring config manually for testing
        if monitoring_enabled:
            from mmp.settings.recipe import MonitoringConfig
            settings.recipe.monitoring = MonitoringConfig(enabled=True)
        return settings

    def test_baseline_structure(self, settings_builder):
        settings = self._make_settings(settings_builder)
        monitor = DataDriftMonitor(settings)

        X_train = pd.DataFrame({
            "num_feat": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat_feat": ["a", "b", "a", "b", "a"],
        })
        baseline = monitor.compute_baseline(X_train, X_train, None, None, {})

        assert "numerical" in baseline
        assert "categorical" in baseline
        assert "meta" in baseline
        assert "num_feat" in baseline["numerical"]
        assert "cat_feat" in baseline["categorical"]
        assert baseline["meta"]["row_count"] == 5

    def test_no_drift_returns_ok(self, settings_builder):
        settings = self._make_settings(settings_builder)
        monitor = DataDriftMonitor(settings)

        np.random.seed(42)
        X = pd.DataFrame({"feat": np.random.randn(200)})
        baseline = monitor.compute_baseline(X, X, None, None, {})

        report = monitor.evaluate(X, None, baseline)
        assert report.status == "ok"
        assert report.metrics["mean_psi"] < 0.1

    def test_drift_triggers_alert(self, settings_builder):
        settings = self._make_settings(settings_builder)
        monitor = DataDriftMonitor(settings)

        np.random.seed(42)
        X_train = pd.DataFrame({"feat": np.random.randn(500)})
        baseline = monitor.compute_baseline(X_train, X_train, None, None, {})

        X_drift = pd.DataFrame({"feat": np.random.randn(500) + 5.0})
        report = monitor.evaluate(X_drift, None, baseline)

        assert report.status in ("warning", "alert")
        assert report.metrics["max_psi"] > 0.1

    def test_schema_drift_detection(self, settings_builder):
        settings = self._make_settings(settings_builder)
        monitor = DataDriftMonitor(settings)

        X_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        baseline = monitor.compute_baseline(X_train, X_train, None, None, {})

        X_new = pd.DataFrame({"a": [1.0, 2.0], "c": [5.0, 6.0]})  # b missing, c new
        report = monitor.evaluate(X_new, None, baseline)

        schema_alerts = [a for a in report.alerts if a.category == "schema"]
        assert len(schema_alerts) == 2  # missing + new


class TestPredictionDriftMonitor:
    """PredictionDriftMonitor baseline + evaluate 테스트"""

    def _make_settings(self, settings_builder):
        settings = settings_builder.with_task("regression").build()
        from mmp.settings.recipe import MonitoringConfig
        settings.recipe.monitoring = MonitoringConfig(enabled=True)
        return settings

    def test_baseline_single_prediction(self, settings_builder):
        settings = self._make_settings(settings_builder)
        monitor = PredictionDriftMonitor(settings)

        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        baseline = monitor.compute_baseline(None, None, y_pred, None, {"metrics": {"rmse": 1.0}})

        assert "_single" in baseline["prediction"]
        assert "training_metrics" in baseline
        assert baseline["training_metrics"]["rmse"] == 1.0

    def test_baseline_quantile_prediction(self, settings_builder):
        settings = self._make_settings(settings_builder)
        monitor = PredictionDriftMonitor(settings)

        y_pred = pd.DataFrame({
            "pred_p50": [1.0, 2.0, 3.0],
            "pred_p90": [2.0, 3.0, 4.0],
        })
        baseline = monitor.compute_baseline(None, None, y_pred, None, {"metrics": {}})

        assert "pred_p50" in baseline["prediction"]
        assert "pred_p90" in baseline["prediction"]

    def test_no_drift_returns_ok(self, settings_builder):
        settings = self._make_settings(settings_builder)
        monitor = PredictionDriftMonitor(settings)

        np.random.seed(42)
        y_pred = np.random.randn(200)
        baseline = monitor.compute_baseline(None, None, y_pred, None, {"metrics": {}})

        report = monitor.evaluate(None, y_pred, baseline)
        assert report.status == "ok"

    def test_extrapolation_detection(self, settings_builder):
        settings = self._make_settings(settings_builder)
        monitor = PredictionDriftMonitor(settings)

        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        baseline = monitor.compute_baseline(None, None, y_train, None, {"metrics": {}})

        y_infer = np.array([100.0, 200.0, 300.0])  # way outside range
        report = monitor.evaluate(None, y_infer, baseline)

        assert report.metrics["extrapolation_ratio"] == 1.0
        extrap_alerts = [a for a in report.alerts if a.metric_name == "extrapolation_ratio"]
        assert len(extrap_alerts) > 0
