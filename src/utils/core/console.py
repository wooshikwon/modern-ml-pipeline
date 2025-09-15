from __future__ import annotations

"""
Unified Console for logging and Rich output.
- Single class: Console
- Environment-aware (test/plain/rich)
- Provides legacy-named methods used across the codebase to minimize call-site churn
- Includes CLI helper functions previously provided at module level
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Callable
import os
import sys
import time

from rich.console import Console as RichConsole
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from src.utils.core.logger import logger


class Console:
    """
    단일 통합 콘솔 클래스.
    - Rich 출력 및 로거 연계
    - 환경 자동 감지(test/plain/rich)
    - 기존 프로젝트에서 사용하던 메서드명 유지(호출부 최소 수정)
    """

    def __init__(self, settings: Any = None):
        self.console = RichConsole()
        self.current_pipeline: Optional[str] = None
        self.progress_bars: Dict[str, Any] = {}
        self.iteration_counters: Dict[str, int] = {}
        self.active_progress: Optional[Progress] = None
        self.mode: str = self._detect_output_mode(settings)

    # ========== 기본 로그 ==========
    def log(
        self,
        message: str,
        *,
        level: str = "info",
        details: str = "",
        context: Optional[Dict[str, Any]] = None,
        operation_type: Optional[str] = None,
    ) -> None:
        # Logger 기록
        log_func = getattr(logger, level, logger.info)
        log_func(message)

        # 콘솔 출력
        if self.mode in ["rich", "test"]:
            if level == "error":
                self.log_error_with_context(message, context=context)
            elif level == "warning":
                self.log_warning_with_context(message, context=context)
            else:
                # operation_type 별 이모지 매핑
                op_emoji = {
                    None: "📝",
                    "general": "📝",
                    "data": "📊",
                    "model": "🤖",
                    "file": "📁",
                    "db": "🗄️",
                }.get(operation_type, "📝")
                self.console.print(f"{op_emoji} {message}")
                if details:
                    self.console.print(f"   [dim]{details}[/dim]")
        elif self.mode == "plain":
            prefix = level.upper()
            print(f"{prefix}: {message}")
            if details:
                print(f"  {details}")

    # ========== 환경 감지 ==========
    def _detect_output_mode(self, settings: Any) -> str:
        # Test environment detection
        if os.environ.get('PYTEST_CURRENT_TEST') or 'pytest' in os.environ.get('_', ''):
            return "test"
        # CI / non-tty
        if self.is_ci_environment() or not sys.stdout.isatty():
            return "plain"
        # settings override
        if settings and hasattr(settings, 'console_mode'):
            return getattr(settings, 'console_mode')
        return "rich"

    def is_ci_environment(self) -> bool:
        return any(env in os.environ for env in ['CI', 'GITHUB_ACTIONS', 'JENKINS_URL'])

    # ========== 파이프라인/프로그레스 ==========
    @contextmanager
    def pipeline_context(self, name: str, description: str):
        self.current_pipeline = name
        self.log_milestone(f"{name}", "start")
        if description:
            self.console.print(description)
        self.console.print()
        try:
            yield
        finally:
            self.log_milestone(f"{name} completed", "finish")
            self.current_pipeline = None
            self.console.print()

    @contextmanager
    def progress_tracker(self, task_id: str, total: int, description: str, show_progress: bool = True):
        if not show_progress or self.mode in ["plain", "test"]:
            # text-only mode for tests/plain
            self.log_milestone(description, "optimization")
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

    def cleanup_completed_tasks(self) -> None:
        completed = [task_id for task_id, (progress, task) in self.progress_bars.items() if task.finished]
        for task_id in completed:
            del self.progress_bars[task_id]

    # ========== 표/테이블/런 정보 ==========
    def display_metrics_table(self, metrics: Dict[str, float], title: str = "Metrics") -> None:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta", justify="right")
        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        self.console.print(table)
        self.console.print()

    def display_unified_metrics_table(self, metrics: Dict[str, Any], performance_summary: str = None) -> None:
        table = Table(title="모델 성능 통합 요약", show_header=True, header_style="bold magenta")
        table.add_column("지표", style="cyan", no_wrap=True)
        table.add_column("값", style="magenta", justify="right")
        table.add_column("해석", style="green")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                continue
            interpretation = self._get_metric_interpretation(metric, value)
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}", interpretation)
            else:
                table.add_row(metric, str(value), interpretation)
        self.console.print(table)
        if performance_summary:
            self.console.print(f"\n📈 [bold blue]전체 성능 요약:[/bold blue]")
            self.console.print(f"   {performance_summary}")

    def _get_metric_interpretation(self, metric_name: str, value: float) -> str:
        if isinstance(value, (int, float)):
            lower = metric_name.lower()
            if "accuracy" in lower or "f1" in lower:
                if value >= 0.9:
                    return "우수"
                if value >= 0.8:
                    return "좋음"
                if value >= 0.7:
                    return "보통"
                return "개선 필요"
            if "loss" in lower or "error" in lower:
                if value <= 0.1:
                    return "우수"
                if value <= 0.2:
                    return "좋음"
                if value <= 0.5:
                    return "보통"
                return "개선 필요"
        return "분석 불가"

    def display_run_info(self, run_id: str, model_uri: str = None, tracking_uri: str = None) -> None:
        self.console.print()
        self.console.print(f"🎯 Run ID: [bold cyan]{run_id}[/bold cyan]")
        if model_uri:
            self.console.print(f"📦 Model URI: [bold green]{model_uri}[/bold green]")
        if tracking_uri:
            if "experiments" in tracking_uri:
                self.console.print(f"🔗 MLflow URI: [link]{tracking_uri}[/link]")
            else:
                self.console.print(f"🔗 Tracking URI: [link]{tracking_uri}[/link]")
        self.console.print()

    # ========== 페이즈/마일스톤/주기적 로그 ==========
    def log_milestone(self, message: str, level: str = "info") -> None:
        emoji_map = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "start": "🚀",
            "data": "📊",
            "model": "🤖",
            "optimization": "🎯",
            "mlflow": "📤",
            "finish": "🏁",
        }
        emoji = emoji_map.get(level, "📝")
        self.console.print(f"{emoji} {message}")

    def log_phase(self, phase_name: str, emoji: str = "📝") -> None:
        self.console.print()
        self.console.print(f"{emoji} {phase_name}")

    def log_periodic(self, process_id: str, iteration: int, data: Dict[str, Any], every_n: int = 10) -> None:
        if (iteration + 1) % every_n == 0 or iteration == 0:
            if process_id == "optuna_trials":
                trial_num = data.get("trial", iteration + 1)
                total_trials = data.get("total_trials", "?")
                score = data.get("score", 0.0)
                best_score = data.get("best_score", score)
                params = data.get("params", {})
                status = "🔥" if score >= best_score else ""
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                self.console.print(
                    f"Trial {trial_num}/{total_trials}: score={score:.4f} (best: {best_score:.4f}) {status} | {param_str}"
                )
            else:
                self.console.print(f"[{process_id}] Iteration {iteration + 1}: {data}")

    # ========== 도메인 특화 로그 ==========
    def log_component_init(self, component_name: str, status: str = "success") -> None:
        emoji = "✅" if status == "success" else "❌" if status == "error" else "🔄"
        self.console.print(f"{emoji} {component_name} initialized")

    def log_processing_step(self, step_name: str, details: str = "") -> None:
        self.console.print(f"   🔄 {step_name}")
        if details:
            self.console.print(f"      [dim]{details}[/dim]")

    def log_warning_with_context(self, message: str, context: Dict[str, Any] = None) -> None:
        self.console.print(f"⚠️  [yellow]{message}[/yellow]")
        if context:
            for key, value in context.items():
                self.console.print(f"      [dim]{key}: {value}[/dim]")

    def log_database_operation(self, operation: str, details: str = "") -> None:
        self.console.print(f"🗄️  {operation}")
        if details:
            self.console.print(f"      [dim]{details}[/dim]")

    def log_feature_engineering(self, step: str, columns: List[str], result_info: str = "") -> None:
        self.console.print(f"🔬 {step}")
        if columns:
            cols_display = ', '.join(columns[:5])
            if len(columns) > 5:
                cols_display += f"... (+{len(columns)-5} more)"
            self.console.print(f"   [dim]Columns: {cols_display}[/dim]")
        if result_info:
            self.console.print(f"   [dim]Result: {result_info}[/dim]")

    def log_data_operation(self, operation: str, shape: Optional[tuple] = None, details: str = "") -> None:
        shape_str = f" ({shape[0]} rows, {shape[1]} columns)" if shape else ""
        self.console.print(f"📊 {operation}{shape_str}")
        if details:
            self.console.print(f"   [dim]{details}[/dim]")

    def log_model_operation(self, operation: str, model_info: str = "") -> None:
        self.console.print(f"🤖 {operation}")
        if model_info:
            self.console.print(f"   [dim]{model_info}[/dim]")

    def log_file_operation(self, operation: str, file_path: str, details: str = "") -> None:
        from pathlib import Path
        filename = Path(file_path).name if file_path else "file"
        self.console.print(f"📁 {operation}: [cyan]{filename}[/cyan]")
        if details:
            self.console.print(f"   [dim]{details}[/dim]")

    def log_error_with_context(self, error_message: str, context: Dict[str, Any] = None, suggestion: str = None) -> None:
        self.console.print(f"❌ [red]Error: {error_message}[/red]")
        if context:
            for key, value in context.items():
                self.console.print(f"   [dim]{key}: {value}[/dim]")
        if suggestion:
            self.console.print(f"   [blue]💡 Suggestion: {suggestion}[/blue]")

    def log_validation_result(self, item: str, status: str, details: str = "") -> None:
        emoji = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
        color = "green" if status == "pass" else "red" if status == "fail" else "yellow"
        self.console.print(f"{emoji} [{color}]{item}[/{color}]")
        if details:
            self.console.print(f"   [dim]{details}[/dim]")

    def log_connection_status(self, service: str, status: str, details: str = "") -> None:
        emoji = "🔗" if status == "connected" else "❌" if status == "failed" else "🔄"
        color = "green" if status == "connected" else "red" if status == "failed" else "yellow"
        self.console.print(f"{emoji} [{color}]{service}: {status.title()}[/{color}]")
        if details:
            self.console.print(f"   [dim]{details}[/dim]")

    def log_pipeline_connection(self, from_component: str, to_component: str, data_info: str = "") -> None:
        self.console.print(f"🔗 {from_component} → {to_component}")
        if data_info:
            self.console.print(f"   [dim]Data: {data_info}[/dim]")

    def log_performance_guidance(self, metric_name: str, value: float, guidance: str) -> None:
        color = "green" if ("좋습니다" in guidance or "우수" in guidance) else "yellow" if ("보통" in guidance) else "red"
        self.console.print(f"📊 {metric_name}: [bold]{value:.4f}[/bold]")
        self.console.print(f"   💡 [{color}]{guidance}[/{color}]")

    def log_artifacts_progress(self, artifacts: List[str]) -> None:
        self.log_phase("MLflow Experiment Tracking", "📤")
        total = len(artifacts)
        with self.progress_tracker("mlflow_artifacts", total, f"Uploading {total} artifacts") as update:
            for i, artifact in enumerate(artifacts):
                time.sleep(0.1)
                update(i + 1)
                self.console.print(f"✅ {artifact} logged")

    # ===== 레거시 시그니처 호환 메서드 =====
    def info(self, message: str, rich_message: str = None, **rich_kwargs) -> None:
        """Unified info logging (Korean-first policy: ignore rich_message)"""
        logger.info(message)
        if self.mode in ["rich", "test"]:
            self.log_milestone(message, "info")
        elif self.mode == "plain":
            print(f"INFO: {message}")

    def warning(self, message: str, rich_message: str = None, context: Dict[str, Any] = None) -> None:
        """Unified warning logging with optional context"""
        logger.warning(message)
        if self.mode in ["rich", "test"]:
            self.log_warning_with_context(message, context=context)
        elif self.mode == "plain":
            print(f"WARNING: {message}")
            if context:
                for key, value in context.items():
                    print(f"  {key}: {value}")

    def error(self, message: str, rich_message: str = None, context: Dict[str, Any] = None, suggestion: str = None) -> None:
        """Unified error logging with context and suggestion"""
        logger.error(message)
        if self.mode in ["rich", "test"]:
            self.log_error_with_context(message, context=context, suggestion=suggestion)
        elif self.mode == "plain":
            print(f"ERROR: {message}")
            if context:
                print(f"  Context: {context}")
            if suggestion:
                print(f"  Suggestion: {suggestion}")

    def debug(self, message: str, rich_message: str = None, **rich_kwargs) -> None:
        """Unified debug logging (dim style in rich/test)"""
        logger.debug(message)
        if self.mode in ["rich", "test"]:
            self.console.print(f"🔍 {message}", style="dim", **rich_kwargs)
        elif self.mode == "plain":
            print(f"DEBUG: {message}")

    def component_init(self, component_name: str, status: str = "success") -> None:
        """Alias for log_component_init"""
        self.log_component_init(component_name, status)

    def data_operation(self, operation: str, shape: Optional[tuple] = None, details: str = "") -> None:
        """Alias for log_data_operation"""
        self.log_data_operation(operation, shape, details)


# ===== 모듈 전역 인스턴스 및 헬퍼 =====

def get_console(settings: Any = None) -> Console:
    return Console(settings)


# CLI helper functions (kept for compatibility with existing callers)
_module_console = Console()


def cli_print(message: str, style: str = None, emoji: str = None) -> None:
    if emoji:
        message = f"{emoji} {message}"
    if style:
        _module_console.console.print(message, style=style)
    else:
        _module_console.console.print(message)


def cli_success(message: str) -> None:
    cli_print(message, style="bold green", emoji="✅")


def cli_error(message: str) -> None:
    cli_print(message, style="bold red", emoji="❌")


def cli_warning(message: str) -> None:
    cli_print(message, style="bold yellow", emoji="⚠️")


def cli_info(message: str) -> None:
    cli_print(message, style="bold blue", emoji="ℹ️")


def cli_success_panel(content: str, title: str = "성공", border_style: str = "green") -> None:
    from rich.panel import Panel
    panel = Panel(content, title=title, border_style=border_style)
    _module_console.console.print(panel)


def cli_command_start(command_name: str, description: str = "") -> None:
    if description:
        _module_console.console.print(f"🚀 {command_name}: {description}", style="bold blue")
    else:
        _module_console.console.print(f"🚀 {command_name}", style="bold blue")


def cli_command_success(command_name: str, details: List[str] = None) -> None:
    _module_console.console.print(f"✅ {command_name}이 성공적으로 완료되었습니다.", style="bold green")
    if details:
        for detail in details:
            _module_console.console.print(f"  • {detail}", style="cyan")


def cli_command_error(command_name: str, error: str, suggestion: str = "") -> None:
    _module_console.console.print(f"❌ {command_name} 실행 중 오류 발생: {error}", style="bold red")
    if suggestion:
        _module_console.console.print(f"   💡 제안: {suggestion}", style="blue")


def cli_step_start(step_name: str, emoji: str = "🔄") -> None:
    _module_console.console.print(f"{emoji} {step_name} 시작...", style="cyan")


def cli_step_complete(step_name: str, details: str = "", duration: float = None) -> None:
    duration_str = f" ({duration:.1f}s)" if duration else ""
    detail_str = f" - {details}" if details else ""
    _module_console.console.print(f"✅ {step_name} 완료{duration_str}{detail_str}", style="green")


def cli_file_created(file_type: str, file_path: str, details: str = "") -> None:
    _module_console.console.print(f"📄 {file_type}: [cyan]{file_path}[/cyan]")
    if details:
        _module_console.console.print(f"   [dim]{details}[/dim]")


def cli_directory_created(dir_path: str, file_count: int = 0) -> None:
    file_info = f" ({file_count} files)" if file_count > 0 else ""
    _module_console.console.print(f"📂 디렉토리 생성: [blue]{dir_path}[/blue]{file_info}")


def cli_next_steps(steps: List[str], title: str = "다음 단계") -> None:
    _module_console.console.print(f"\n💡 {title}:", style="bold blue")
    for i, step in enumerate(steps, 1):
        _module_console.console.print(f"  {i}. [cyan]{step}[/cyan]")


def cli_validation_result(item: str, status: str, details: str = "") -> None:
    emoji_map = {
        "pass": "✅",
        "fail": "❌",
        "warning": "⚠️",
        "success": "✅",
        "error": "❌",
    }
    color_map = {
        "pass": "green",
        "fail": "red",
        "warning": "yellow",
        "success": "green",
        "error": "red",
    }
    emoji = emoji_map.get(status, "📝")
    color = color_map.get(status, "white")
    _module_console.console.print(f"{emoji} [{color}]{item}[/{color}]")
    if details:
        _module_console.console.print(f"   [dim]{details}[/dim]")


def cli_validation_summary(results: List[Dict[str, Any]], title: str = "검증 결과") -> None:
    _module_console.console.print(f"\n📋 {title}:", style="bold blue")
    for result in results:
        cli_validation_result(result.get('item', ''), result.get('status', ''), result.get('details', ''))


def cli_connection_test(service: str, status: str, details: str = "") -> None:
    emoji_map = {"connected": "🔗", "failed": "❌", "connecting": "🔄"}
    color_map = {"connected": "green", "failed": "red", "connecting": "yellow"}
    emoji = emoji_map.get(status, "🔗")
    color = color_map.get(status, "white")
    _module_console.console.print(f"{emoji} [{color}]{service}: {status.title()}[/{color}]")
    if details:
        _module_console.console.print(f"   [dim]{details}[/dim]")


def cli_system_check_header(config_path: str, env_name: str = None) -> None:
    env_info = f" (env: {env_name})" if env_name else ""
    _module_console.console.print(f"🔍 시스템 체크: [cyan]{config_path}[/cyan]{env_info}", style="bold blue")


def cli_template_processing(template_name: str, output_path: str, context: Dict = None) -> None:
    _module_console.console.print(f"🎨 템플릿 렌더링: [magenta]{template_name}[/magenta] → [cyan]{output_path}[/cyan]")
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        _module_console.console.print(f"   [dim]Context: {context_str}[/dim]")


def cli_usage_example(command: str, examples: List[str]) -> None:
    _module_console.console.print(f"\n💡 사용 예시:", style="bold blue")
    for example in examples:
        _module_console.console.print(f"  [cyan]{command} {example}[/cyan]")


def cli_troubleshooting_tip(issue: str, solution: str) -> None:
    _module_console.console.print(f"🔧 문제: {issue}", style="yellow")
    _module_console.console.print(f"   해결: [green]{solution}[/green]")


def cli_process_status(process: str, current: int, total: int, details: str = "") -> None:
    detail_str = f" - {details}" if details else ""
    _module_console.console.print(f"{process}: [{current}/{total}]{detail_str}")