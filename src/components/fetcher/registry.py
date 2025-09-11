from __future__ import annotations
from typing import Dict, Type
from src.interface import BaseFetcher
from src.utils.system.console_manager import get_console

class FetcherRegistry:
    """컴포넌트 레벨 피처 페처 레지스트리 (엔진 의존성 제거)."""
    fetchers: Dict[str, Type[BaseFetcher]] = {}

    @classmethod
    def register(cls, fetcher_type: str, fetcher_class: Type[BaseFetcher]):
        if not issubclass(fetcher_class, BaseFetcher):
            raise TypeError(f"{fetcher_class.__name__} must be a subclass of BaseFetcher")
        cls.fetchers[fetcher_type] = fetcher_class
        console = get_console()
        try:
            console.debug(f"[components] Fetcher registered: {fetcher_type} -> {fetcher_class.__name__}",
                         rich_message=f"📝 Fetcher registered: [cyan]{fetcher_type}[/cyan] → [green]{fetcher_class.__name__}[/green]")
        except AttributeError:
            # debug 메서드가 없으면 info 사용
            console.info(f"[components] Fetcher registered: {fetcher_type} -> {fetcher_class.__name__}",
                        rich_message=f"📝 Fetcher registered: [cyan]{fetcher_type}[/cyan] → [green]{fetcher_class.__name__}[/green]")

    @classmethod
    def create(cls, fetcher_type: str, *args, **kwargs) -> BaseFetcher:
        fetcher_class = cls.fetchers.get(fetcher_type)
        if not fetcher_class:
            available = list(cls.fetchers.keys())
            raise ValueError(f"Unknown fetcher type: '{fetcher_type}'. Available types: {available}")
        console = get_console()
        try:
            console.debug(f"[components] Creating fetcher instance: {fetcher_type}",
                         rich_message=f"🔧 Creating fetcher: [cyan]{fetcher_type}[/cyan]")
        except AttributeError:
            # debug 메서드가 없으면 info 사용
            console.info(f"[components] Creating fetcher instance: {fetcher_type}",
                        rich_message=f"🔧 Creating fetcher: [cyan]{fetcher_type}[/cyan]")
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