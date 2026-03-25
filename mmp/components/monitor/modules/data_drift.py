"""DataDriftMonitor - 피처 분포 변화 감지."""

import numpy as np
import pandas as pd

from mmp.components.monitor.base import Alert, BaseMonitor, MonitorReport
from mmp.components.monitor.utils import EPSILON, NUM_BINS, compute_psi
from mmp.settings import Settings
from mmp.utils.core.logger import logger


def _compute_js_divergence(series: pd.Series, ref_proportions: dict[str, float]) -> float:
    """Jensen-Shannon divergence를 계산한다."""
    current_counts = series.value_counts(normalize=True)
    all_keys = set(ref_proportions.keys()) | set(current_counts.index)
    p = np.array([ref_proportions.get(k, 0.0) for k in all_keys])
    q = np.array([current_counts.get(k, 0.0) for k in all_keys])
    # 정규화
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum
    m = 0.5 * (p + q)
    # KL divergence with epsilon
    eps = EPSILON
    kl_pm = float(np.sum(p * np.log((p + eps) / (m + eps))))
    kl_qm = float(np.sum(q * np.log((q + eps) / (m + eps))))
    return 0.5 * kl_pm + 0.5 * kl_qm


class DataDriftMonitor(BaseMonitor):
    """피처 분포 드리프트 모니터."""

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
        df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        numerical: dict[str, dict] = {}
        categorical: dict[str, dict] = {}

        for col in df.columns:
            series = df[col]
            missing_rate = float(series.isna().mean())

            if pd.api.types.is_numeric_dtype(series):
                valid = series.dropna().values.astype(float)
                if len(valid) == 0:
                    numerical[col] = {
                        "bin_edges": [],
                        "bin_proportions": [],
                        "mean": 0.0,
                        "std": 0.0,
                        "missing_rate": missing_rate,
                    }
                    continue
                counts, bin_edges = np.histogram(valid, bins=NUM_BINS)
                total = counts.sum()
                proportions = (counts / total).tolist() if total > 0 else [0.0] * NUM_BINS
                numerical[col] = {
                    "bin_edges": [float(e) for e in bin_edges],
                    "bin_proportions": proportions,
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "missing_rate": missing_rate,
                }
            else:
                value_counts = series.dropna().value_counts(normalize=True)
                categorical[col] = {
                    "value_counts": {str(k): float(v) for k, v in value_counts.items()},
                    "missing_rate": missing_rate,
                }

        meta = {
            "columns": list(df.columns),
            "row_count": len(df),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
        }
        logger.debug("[MONITOR] data_drift baseline 계산 완료 - %d개 피처", len(df.columns))
        return {"numerical": numerical, "categorical": categorical, "meta": meta}

    def evaluate(self, X_inference, y_pred, baseline: dict) -> MonitorReport:
        monitoring_config = getattr(self.settings.recipe, "monitoring", None)
        if monitoring_config:
            cfg = monitoring_config.data_drift
            psi_warning = cfg.psi_threshold_warning
            psi_alert = cfg.psi_threshold_alert
            missing_threshold = cfg.missing_rate_threshold
        else:
            psi_warning = 0.1
            psi_alert = 0.2
            missing_threshold = 0.05

        df = X_inference if isinstance(X_inference, pd.DataFrame) else pd.DataFrame(X_inference)
        metrics: dict[str, float] = {}
        alerts: list[Alert] = []

        baseline_cols = set(baseline["meta"]["columns"])
        current_cols = set(df.columns)

        # Schema drift
        missing_cols = baseline_cols - current_cols
        new_cols = current_cols - baseline_cols
        if missing_cols:
            alerts.append(
                Alert(
                    category="schema",
                    feature="overall",
                    metric_name="missing_columns",
                    metric_value=float(len(missing_cols)),
                    threshold=0.0,
                    severity="alert",
                    message=f"누락된 컬럼: {sorted(missing_cols)}",
                )
            )
        if new_cols:
            alerts.append(
                Alert(
                    category="schema",
                    feature="overall",
                    metric_name="new_columns",
                    metric_value=float(len(new_cols)),
                    threshold=0.0,
                    severity="warning",
                    message=f"새로운 컬럼: {sorted(new_cols)}",
                )
            )

        # Numerical PSI
        psi_values: list[float] = []
        for col, ref in baseline["numerical"].items():
            if col not in df.columns:
                continue
            series = df[col].dropna().values.astype(float)
            if not ref["bin_edges"]:
                continue
            psi = compute_psi(series, ref["bin_edges"], ref["bin_proportions"])
            psi_values.append(psi)
            metrics[f"psi_{col}"] = psi

            if psi >= psi_alert:
                alerts.append(
                    Alert(
                        category="data_drift",
                        feature=col,
                        metric_name="psi",
                        metric_value=psi,
                        threshold=psi_alert,
                        severity="alert",
                        message=f"{col} PSI={psi:.4f} >= {psi_alert}",
                    )
                )
            elif psi >= psi_warning:
                alerts.append(
                    Alert(
                        category="data_drift",
                        feature=col,
                        metric_name="psi",
                        metric_value=psi,
                        threshold=psi_warning,
                        severity="warning",
                        message=f"{col} PSI={psi:.4f} >= {psi_warning}",
                    )
                )

            # Missing rate delta
            current_missing = float(df[col].isna().mean())
            baseline_missing = ref["missing_rate"]
            delta = abs(current_missing - baseline_missing)
            metrics[f"missing_delta_{col}"] = delta
            if delta >= missing_threshold:
                alerts.append(
                    Alert(
                        category="data_drift",
                        feature=col,
                        metric_name="missing_rate_delta",
                        metric_value=delta,
                        threshold=missing_threshold,
                        severity="warning",
                        message=f"{col} 결측률 변화={delta:.4f} >= {missing_threshold}",
                    )
                )

        # Categorical JS divergence
        for col, ref in baseline["categorical"].items():
            if col not in df.columns:
                continue
            js = _compute_js_divergence(df[col], ref["value_counts"])
            metrics[f"js_divergence_{col}"] = js

            # Missing rate delta
            current_missing = float(df[col].isna().mean())
            baseline_missing = ref["missing_rate"]
            delta = abs(current_missing - baseline_missing)
            metrics[f"missing_delta_{col}"] = delta
            if delta >= missing_threshold:
                alerts.append(
                    Alert(
                        category="data_drift",
                        feature=col,
                        metric_name="missing_rate_delta",
                        metric_value=delta,
                        threshold=missing_threshold,
                        severity="warning",
                        message=f"{col} 결측률 변화={delta:.4f} >= {missing_threshold}",
                    )
                )

        # Overall PSI stats
        if psi_values:
            metrics["mean_psi"] = float(np.mean(psi_values))
            metrics["max_psi"] = float(np.max(psi_values))
        else:
            metrics["mean_psi"] = 0.0
            metrics["max_psi"] = 0.0

        logger.info(
            "[MONITOR] data_drift 평가 완료 - mean_psi=%.4f, alerts=%d",
            metrics["mean_psi"],
            len(alerts),
        )
        return MonitorReport(metrics=metrics, alerts=alerts)


# Self-registration
from ..registry import MonitorRegistry

MonitorRegistry.register("data_drift", DataDriftMonitor)
