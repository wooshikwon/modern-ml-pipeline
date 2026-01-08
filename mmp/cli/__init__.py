"""CLI Module Public API

CLI 시작 속도 개선을 위해 app은 lazy import로 필요할 때만 로드합니다.
"""

__all__ = ["app"]


def __getattr__(name: str):
    """Lazy import for app."""
    if name == "app":
        from .main_commands import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
