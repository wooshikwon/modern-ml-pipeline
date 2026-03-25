"""Monitor Registry - Self-registration pattern for monitors."""

from __future__ import annotations

from typing import Dict, Type

from mmp.components.base_registry import BaseRegistry

from .base import BaseMonitor


class MonitorRegistry(BaseRegistry[BaseMonitor]):
    """컴포넌트 레벨 모니터 레지스트리"""

    _registry: Dict[str, Type[BaseMonitor]] = {}
    _base_class = BaseMonitor
