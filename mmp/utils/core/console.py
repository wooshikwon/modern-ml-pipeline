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

from mmp.utils.core.logger import logger


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



# CLI helper 전역 인스턴스
_module_console = Console()
