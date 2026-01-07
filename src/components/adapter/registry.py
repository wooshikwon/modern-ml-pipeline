"""
Adapter Registry - Self-registration pattern for data adapters.
각 어댑터 모듈에서 자동으로 자신을 등록하여 의존성을 줄입니다.
"""

from typing import Dict, Type

from src.components.base_registry import BaseRegistry

from .base import BaseAdapter


class AdapterRegistry(BaseRegistry[BaseAdapter]):
    """Data Adapter 등록 및 관리 클래스"""

    _registry: Dict[str, Type[BaseAdapter]] = {}
    _base_class = BaseAdapter
