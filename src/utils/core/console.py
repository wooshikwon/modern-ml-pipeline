from __future__ import annotations

"""
CLI 전용 Console 모듈.
- Rich 기반의 CLI 출력 기능만 제공
- 파이프라인 로깅은 logger 모듈 사용
"""

import os
import sys
from contextlib import contextmanager
from typing import Any, Dict

from rich.console import Console as RichConsole
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.utils.core.logger import logger


class Console:
    """
    CLI 전용 콘솔 클래스.
    - Rich 출력 및 프로그레스 바
    - 환경 자동 감지(test/plain/rich)
    """

    def __init__(self, settings: Any = None):
        self.mode: str = self._detect_output_mode(settings)
        if self.mode == "test":
            self.console = RichConsole(width=200, soft_wrap=False)
        else:
            self.console = RichConsole()
        self.progress_bars: Dict[str, Any] = {}

    def print(self, *args, **kwargs) -> None:
        """Rich console 출력"""
        try:
            self.console.print(*args, **kwargs)
        except Exception:
            builtins_print = __builtins__.get("print") if isinstance(__builtins__, dict) else print
            builtins_print(*args)

    def _detect_output_mode(self, settings: Any) -> str:
        """출력 모드 감지"""
        if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in os.environ.get("_", ""):
            return "test"
        if self.is_ci_environment() or not sys.stdout.isatty():
            return "plain"
        if settings and hasattr(settings, "console_mode"):
            return getattr(settings, "console_mode")
        return "rich"

    def is_ci_environment(self) -> bool:
        """CI 환경 여부 확인"""
        return any(env in os.environ for env in ["CI", "GITHUB_ACTIONS", "JENKINS_URL"])

    def get_console_mode(self) -> str:
        """현재 콘솔 모드 반환"""
        if self.is_ci_environment() or not sys.stdout.isatty():
            return "plain"
        return "rich"

    @contextmanager
    def progress_tracker(
        self, task_id: str, total: int, description: str, show_progress: bool = True
    ):
        """CLI 프로그레스 바 표시. 프로그레스 바 불가 환경에서는 로거로 fallback"""
        if not show_progress or self.mode in ["plain", "test"]:
            logger.info(f"[Progress] {description}")
            yield lambda current=0: None
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            task = progress.add_task(description, total=total)
            self.progress_bars[task_id] = (progress, task)

            def update_progress(current: int):
                progress.update(task, completed=current)

            try:
                yield update_progress
            finally:
                progress.update(task, completed=total)
                if task_id in self.progress_bars:
                    del self.progress_bars[task_id]


def get_console(settings: Any = None) -> Console:
    """Console 인스턴스 생성"""
    return Console(settings)


def get_rich_console() -> RichConsole:
    """Rich Console 인스턴스 반환 (CLI 직접 접근용)"""
    return RichConsole()


# CLI helper 전역 인스턴스
_module_console = Console()


def cli_success_panel(content: str, title: str = "성공", border_style: str = "green") -> None:
    from rich.panel import Panel

    panel = Panel(content, title=title, border_style=border_style)
    _module_console.console.print(panel)


def cli_command_start(command_name: str, description: str = "") -> None:
    """CLI 명령어 시작 메시지 출력"""
    if description:
        _module_console.console.print(f"🚀 {command_name}: {description}", style="bold blue")
    else:
        _module_console.console.print(f"🚀 {command_name}", style="bold blue")


def cli_command_error(command_name: str, error: str, suggestion: str = "") -> None:
    """CLI 명령어 에러 메시지 출력"""
    _module_console.console.print(f"❌ {command_name} 실행 중 오류 발생: {error}", style="bold red")
    if suggestion:
        _module_console.console.print(f"   💡 제안: {suggestion}", style="blue")


def cli_step_complete(step_name: str, details: str = "", duration: float = None) -> None:
    """CLI 단계 완료 메시지 출력"""
    duration_str = f" ({duration:.1f}s)" if duration else ""
    detail_str = f" - {details}" if details else ""
    _module_console.console.print(f"✅ {step_name} 완료{duration_str}{detail_str}", style="green")


def cli_info(message: str) -> None:
    """CLI 정보 메시지 출력"""
    _module_console.console.print(f"ℹ️ {message}", style="bold blue")
