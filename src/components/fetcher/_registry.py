from __future__ import annotations
from typing import Dict, Type
from src.interface import BaseFetcher
from src.utils.system.logger import logger

class FetcherRegistry:
    """컴포넌트 레벨 피처 페처 레지스트리 (엔진 의존성 제거)."""
    _fetchers: Dict[str, Type[BaseFetcher]] = {}

    @classmethod
    def register(cls, fetcher_type: str, fetcher_class: Type[BaseFetcher]):
        if not issubclass(fetcher_class, BaseFetcher):
            raise TypeError(f"{fetcher_class.__name__} must be a subclass of BaseFetcher")
        cls.fetchers[fetcher_type] = fetcher_class
        logger.debug(f"[components] Fetcher registered: {fetcher_type} -> {fetcher_class.__name__}")

    @classmethod
    def create(cls, fetcher_type: str, *args, **kwargs) -> BaseFetcher:
        fetcher_class = cls.fetchers.get(fetcher_type)
        if not fetcher_class:
            available = list(cls.fetchers.keys())
            raise ValueError(f"Unknown fetcher type: '{fetcher_type}'. Available types: {available}")
        logger.debug(f"[components] Creating fetcher instance: {fetcher_type}")
        return fetcher_class(*args, **kwargs)

    @classmethod
    def get_available_types(cls) -> list[str]:
        """등록된 모든 fetcher type 목록 반환."""
        return list(cls.fetchers.keys())

    @classmethod 
    def get_fetcher_class(cls, fetcher_type: str) -> Type[BaseFetcher]:
        """Fetcher type에 해당하는 Fetcher 클래스 반환."""
        fetcher_class = cls.fetchers.get(fetcher_type)
        if not fetcher_class:
            available = list(cls.fetchers.keys())
            raise ValueError(f"Unknown fetcher type: '{fetcher_type}'. Available types: {available}")
        return fetcher_class