"""PredictionDriftMonitor - 예측값 분포 변화 감지."""

import numpy as np
import pandas as pd

from mmp.components.monitor.base import Alert, BaseMonitor, MonitorReport
from mmp.components.monitor.utils import NUM_BINS, compute_psi
from mmp.settings import Settings
from mmp.utils.core.logger import logger


def _compute_pred_stats(values: np.ndarray) -> dict:
    """예측값의 히스토그램 및 기본 통계를 계산한다."""
    values = values.astype(float)
    counts, bin_edges = np.histogram(values, bins=NUM_BINS)
    total = counts.sum()
    proportions = (counts / total).tolist() if total > 0 else [0.0] * NUM_BINS
    return {
        "bin_edges": [float(e) for e in bin_edges],
        "bin_proportions": proportions,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


class PredictionDriftMonitor(BaseMonitor):
    """예측값 분포 드리프트 모니터."""

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def compute_baseline(
        self,
        X_train,
        X_test,
        y_test_pred,
        y_test_true,
        metadata: dict,
    ) -> dict:
        prediction: dict[str, dict] = {}

        if isinstance(y_test_pred, pd.DataFrame):
            for col in y_test_pred.columns:
                values = y_test_pred[col].dropna().values
                if len(values) > 0:
                    prediction[col] = _compute_pred_stats(values)
        else:
            values = np.asarray(y_test_pred).ravel()
            if len(values) > 0:
                prediction["_single"] = _compute_pred_stats(values)

        result = {"prediction": prediction}
        # 학습 메트릭 보존
        if "metrics" in metadata:
            result["training_metrics"] = {
                k: float(v) for k, v in metadata["metrics"].items()
            }

        logger.debug(
            "[MONITOR] prediction_drift baseline 계산 완료 - %d개 분포",
            len(prediction),
        )
        return result

    def evaluate(self, X_inference, y_pred, baseline: dict) -> MonitorReport:
        monitoring_config = getattr(self.settings.recipe, "monitoring", None)
        if monitoring_config:
            cfg = monitoring_config.prediction_drift
            psi_threshold = cfg.psi_threshold
            extrap_threshold = cfg.extrapolation_threshold
        else:
            psi_threshold = 0.2
            extrap_threshold = 0.05

        metrics: dict[str, float] = {}
        alerts: list[Alert] = []
        pred_baseline = baseline["prediction"]

        if isinstance(y_pred, pd.DataFrame):
            pred_items = [(col, y_pred[col].dropna().values) for col in y_pred.columns]
        else:
            pred_items = [("_single", np.asarray(y_pred).ravel())]

        for key, values in pred_items:
            ref = pred_baseline.get(key)
            if ref is None or not ref["bin_edges"]:
                continue

            values = values.astype(float)
            suffix = "" if key == "_single" else f"_{key}"

            # Prediction PSI
            psi = compute_psi(values, ref["bin_edges"], ref["bin_proportions"])
            metrics[f"pred_psi{suffix}"] = psi
            if psi >= psi_threshold:
                alerts.append(
                    Alert(
                        category="prediction_drift",
                        feature=key,
                        metric_name="pred_psi",
                        metric_value=psi,
                        threshold=psi_threshold,
                        severity="alert",
                        message=f"{key} 예측 PSI={psi:.4f} >= {psi_threshold}",
                    )
                )

            # Extrapolation ratio
            ref_min, ref_max = ref["min"], ref["max"]
            if len(values) > 0:
                extrap_ratio = float(np.mean((values < ref_min) | (values > ref_max)))
            else:
                extrap_ratio = 0.0
            metrics[f"extrapolation_ratio{suffix}"] = extrap_ratio
            if extrap_ratio >= extrap_threshold:
                alerts.append(
                    Alert(
                        category="prediction_drift",
                        feature=key,
                        metric_name="extrapolation_ratio",
                        metric_value=extrap_ratio,
                        threshold=extrap_threshold,
                        severity="warning",
                        message=f"{key} 외삽 비율={extrap_ratio:.4f} >= {extrap_threshold}",
                    )
                )

            # Mean shift
            mean_shift = abs(float(np.mean(values)) - ref["mean"])
            metrics[f"mean_shift{suffix}"] = mean_shift

        logger.info(
            "[MONITOR] prediction_drift 평가 완료 - alerts=%d",
            len(alerts),
        )
        return MonitorReport(metrics=metrics, alerts=alerts)


# Self-registration
from ..registry import MonitorRegistry

MonitorRegistry.register("prediction_drift", PredictionDriftMonitor)
