"""CLI Utilities Module

CLI 명령어 실행에 필요한 유틸리티 함수들을 제공합니다.

무거운 모듈(RecipeBuilder, InteractiveConfigBuilder 등)은 lazy import로
필요할 때만 로드하여 CLI 시작 속도를 개선합니다.
"""

# 가벼운 모듈만 즉시 import
from .cli_progress import CLIProgress
from .interactive_ui import InteractiveUI
from .template_engine import TemplateEngine

__all__ = [
    "TemplateEngine",
    "RecipeBuilder",
    "InteractiveConfigBuilder",
    "InteractiveUI",
    "SystemChecker",
    "CLIProgress",
]


def __getattr__(name: str):
    """Lazy import for heavy modules."""
    if name == "RecipeBuilder":
        from .recipe_builder import RecipeBuilder
        return RecipeBuilder
    elif name == "InteractiveConfigBuilder":
        from .config_builder import InteractiveConfigBuilder
        return InteractiveConfigBuilder
    elif name == "SystemChecker":
        from .system_checker import SystemChecker
        return SystemChecker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
