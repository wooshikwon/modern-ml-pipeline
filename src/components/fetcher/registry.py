"""
Fetcher Registry - Self-registration pattern for feature fetchers.
"""

from __future__ import annotations

from typing import Dict, Type

from src.components.base_registry import BaseRegistry

from .base import BaseFetcher


class FetcherRegistry(BaseRegistry[BaseFetcher]):
    """컴포넌트 레벨 피처 페처 레지스트리"""

    _registry: Dict[str, Type[BaseFetcher]] = {}
    _base_class = BaseFetcher
