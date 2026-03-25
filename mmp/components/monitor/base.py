"""BaseMonitor - 모니터링 기본 인터페이스"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmp.settings import Settings


@dataclass
class Alert:
    """모니터링 임계치 초과 항목."""

    category: str
    feature: str
    metric_name: str
    metric_value: float
    threshold: float
    severity: str
    message: str


@dataclass
class MonitorReport:
    """모니터링 평가 결과."""

    metrics: dict[str, float] = field(default_factory=dict)
    alerts: list[Alert] = field(default_factory=list)

    @property
    def status(self) -> str:
        for a in self.alerts:
            if a.severity == "alert":
                return "alert"
        for a in self.alerts:
            if a.severity == "warning":
                return "warning"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "metrics": self.metrics,
            "alerts": [vars(a) for a in self.alerts],
            "alert_count": len(self.alerts),
        }

    def merge(self, other: "MonitorReport") -> "MonitorReport":
        """다른 MonitorReport의 metrics와 alerts를 self에 합친다 (in-place)."""
        self.metrics.update(other.metrics)
        self.alerts.extend(other.alerts)
        return self


class BaseMonitor(ABC):
    """모든 Monitor가 따라야 할 표준 계약."""

    def __init__(self, settings: "Settings"):
        self.settings = settings

    @abstractmethod
    def compute_baseline(
        self,
        X_train,
        X_test,
        y_test_pred,
        y_test_true,
        metadata: dict,
    ) -> dict:
        """학습 시 호출 — baseline 통계를 JSON-serializable dict로 반환."""
        pass

    @abstractmethod
    def evaluate(self, X_inference, y_pred, baseline: dict) -> MonitorReport:
        """추론 시 호출 — baseline 대비 드리프트를 평가."""
        pass
