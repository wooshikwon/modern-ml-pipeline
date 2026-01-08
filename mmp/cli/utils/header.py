"""
CLI 공통 헤더 유틸리티

대화형 명령어(init, get-config, get-recipe, list)에서 사용하는
공통 헤더 출력 기능을 제공합니다.
"""

import shutil
import sys

from rich.console import Console

# mmp.settings 전체 로드를 피하기 위해 메타데이터에서 직접 버전 조회
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("modern-ml-pipeline")
except Exception:
    __version__ = "unknown"

# 공통 콘솔 인스턴스
_console = Console()

# UI 최대 폭
UI_MAX_WIDTH = 80


def _get_line_width() -> int:
    """터미널 폭 기반 라인 폭 계산 (최대값 제한)."""
    terminal_width = shutil.get_terminal_size().columns
    return min(terminal_width, UI_MAX_WIDTH)


def print_command_header(command_title: str, description: str = "") -> None:
    """
    대화형 명령어 공통 헤더 출력.

    Args:
        command_title: 명령어 제목 (예: "Init Project", "Get Config")
        description: 명령어 설명 (예: "Interactive project initializer")
    """
    sys.stdout.write(f"\nmmp v{__version__}\n\n")
    if description:
        sys.stdout.write(f"{command_title}: {description}\n\n")
    else:
        sys.stdout.write(f"{command_title}\n\n")
    sys.stdout.flush()


def print_simple_header(title: str) -> None:
    """
    단순 헤더 출력 (list 명령어용).

    Args:
        title: 제목 (예: "Adapters", "Models by Task")
    """
    sys.stdout.write(f"\nmmp v{__version__}\n\n")
    sys.stdout.write(f"{title}\n")
    sys.stdout.flush()


def print_divider() -> None:
    """구분선 출력."""
    line_width = _get_line_width()
    _console.print(f"[dim]{'─' * line_width}[/dim]")


def print_section(tag: str, title: str, style: str = "cyan", newline: bool = True) -> None:
    """
    섹션 헤더 출력.

    Args:
        tag: 태그 (예: "SUMMARY", "TODO", "OK", "NEXT")
        title: 섹션 제목
        style: Rich 스타일 (cyan, green, yellow, blue 등)
        newline: 섹션 앞에 빈 줄 추가 여부
    """
    if newline:
        _console.print()
    _console.print(f"[bold {style}][{tag}][/bold {style}] {title}")


def print_item(prefix: str, value: str, indent: int = 2) -> None:
    """
    항목 출력.

    Args:
        prefix: 접두사 (예: "FILE", "TASK")
        value: 값
        indent: 들여쓰기 칸 수
    """
    spaces = " " * indent
    _console.print(f"{spaces}[dim][{prefix}][/dim] {value}")
