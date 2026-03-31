"""
Modern ML Pipeline CLI - Main Commands Router

이 모듈은 CLI의 메인 진입점으로, Typer 앱 생성과 커맨드 라우팅을 담당한다.
실제 비즈니스 로직은 각 커맨드 모듈(commands/)에 위치하며, 여기서는
"어떤 커맨드가 있고, 어떤 순서로 실행되는지"만 정의한다.

CLI 실행 흐름:
    mmp --quiet train --config my.yaml
    │   │       │
    │   │       └── @app.command()로 등록된 서브커맨드 → train_command() 실행
    │   └────────── @app.callback()의 글로벌 옵션 → main(quiet=True) 먼저 실행
    └────────────── pyproject.toml [project.scripts]가 mmp.__main__:main을 가리킴
"""

import os

# MLflow 출력 억제 (import 전에 설정 필요)
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
os.environ["MLFLOW_LOGGING_LEVEL"] = "ERROR"

import logging
import warnings
from importlib.metadata import version as get_pkg_version

import typer
from rich.console import Console
from rich.text import Text
from typing_extensions import Annotated

# 모든 라이브러리 경고를 진입점에서 차단
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="scipy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pytorch_tabnet")
# 특정 메시지 패턴 차단
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
warnings.filterwarnings("ignore", message=".*json_schema_extra.*")
warnings.filterwarnings("ignore", message=".*scipy.sparse.base.*")

from mmp.cli.commands.get_config_command import get_config_command
from mmp.cli.commands.get_recipe_command import get_recipe_command
from mmp.cli.commands.inference_command import batch_inference_command

# Command imports
from mmp.cli.commands.init_command import init_command

# List commands
from mmp.cli.commands.list_commands import (
    list_adapters,
    list_evaluators,
    list_metrics,
    list_models,
    list_preprocessors,
)
from mmp.cli.commands.serve_command import serve_api_command
from mmp.cli.commands.system_check_command import system_check_command
from mmp.cli.commands.train_command import train_command
from mmp.cli.commands.validate_command import validate_command
from mmp.utils.core.logger import setup_log_level

# ASCII Art Banner (Simple version for terminal compatibility)
ASCII_BANNER = """
  __  __   __  __   ____
 |  \\/  | |  \\/  | |  _ \\
 | |\\/| | | |\\/| | | |_) |
 | |  | | | |  | | |  __/
 |_|  |_| |_|  |_| |_|
"""

console = Console()


def show_banner():
    """Display ASCII art banner when CLI is invoked."""
    banner_text = Text(ASCII_BANNER, style="bold cyan")
    console.print(banner_text)
    console.print("[bold yellow]Modern ML Pipeline[/bold yellow] - ML 워크플로우를 위한 CLI 도구\n")


def _get_version() -> str:
    """
    패키지 메타데이터에서 버전 정보를 가져옴.

    Returns:
        str: 설치된 패키지 버전, 실패 시 "unknown"
    """
    try:
        return get_pkg_version("modern-ml-pipeline")
    except Exception:
        return "unknown"


# Typer 앱 인스턴스 — CLI의 최상위 객체.
# 여기서는 앱 자체의 메타 설정(도움말 텍스트, 옵션 이름 등)만 정의하고,
# 글로벌 옵션(--version, --quiet)은 아래 @app.callback()에서 처리한다.
app = typer.Typer(
    help=(
        "🚀 Modern ML Pipeline - Unified CLI Interface\n\n"
        "[dim]AI/LLM agents: Read AGENT.md for schema reference and working examples.[/dim]"
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    """
    --version 옵션의 콜백. Typer가 --version을 파싱하면 이 함수를 "call back" 해준다.
    버전을 출력한 뒤 즉시 종료하므로, 서브커맨드는 실행되지 않는다.
    """
    if value:
        version = _get_version()
        typer.echo(f"modern-ml-pipeline {version}")
        typer.echo("AI/LLM agents: Read AGENT.md for schema reference and working examples.")
        raise typer.Exit()


# @app.callback() — 모든 서브커맨드 실행 전에 Typer가 먼저 호출하는 함수.
# 여기서 --version, --quiet 같은 글로벌 옵션을 처리한다.
# 옵션을 넘기지 않으면 기본값(False)으로 들어와 아무 일도 하지 않는다.
#
# callback이 필요한 이유: --quiet은 특정 커맨드(train, serve 등)가 아니라
# 앱 자체에 붙는 옵션이다. callback 없이는 이런 "앱 레벨 옵션"을 정의할 곳이 없어서,
# 모든 커맨드에 --quiet을 중복 추가하거나 글로벌 옵션을 포기해야 한다.
@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=version_callback, help="Show version")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="요약 출력 (진행 상태만)")
    ] = False,
) -> None:
    """
    Modern ML Pipeline CLI - 글로벌 전처리.

    모든 서브커맨드(train, serve-api 등)가 실행되기 전에 이 함수가 먼저 호출되어
    로그 레벨 등 공통 설정을 적용한다.
    """
    from mmp.utils.core.logger import CLI_LEVEL

    if quiet:
        setup_log_level(CLI_LEVEL)
    else:
        # 기본: 상세 출력 (K8s 로그 등에서 유용)
        setup_log_level(logging.DEBUG)


# ═══════════════════════════════════════════════════
# 서브커맨드 등록
# ═══════════════════════════════════════════════════
# app.command("name")(func)는 @app.command("name") 데코레이터와 동일하다.
# 커맨드 함수가 다른 모듈에 정의되어 있으므로 데코레이터 대신 이 방식을 사용한다.

# --- Project Management ---

app.command("init", help="프로젝트 초기화 - 기본 디렉토리 구조 및 파일 생성")(init_command)
app.command("get-config", help="대화형으로 환경별 설정 파일 생성")(get_config_command)
app.command("get-recipe", help="대화형 모델 선택 및 레시피 생성")(get_recipe_command)
app.command("system-check", help="현재 config 파일 기반 시스템 연결 상태 검사")(
    system_check_command
)


# --- ML Pipeline ---

app.command("validate", help="Recipe + Config 사전 검증 (학습 없이)")(validate_command)
app.command("train", help="학습 파이프라인 실행")(train_command)
app.command("batch-inference", help="배치 추론 실행")(batch_inference_command)
app.command("serve-api", help="API 서버 실행")(serve_api_command)


# --- List Commands (중첩 서브커맨드) ---
# list_app은 별도의 Typer 인스턴스로, app.add_typer()를 통해 하위 커맨드 그룹이 된다.
# 이로써 `mmp list models`, `mmp list adapters`처럼 2단계 커맨드 구조를 만든다.
#   mmp list models
#   │    │    └── list_app의 서브커맨드
#   │    └────── app의 서브커맨드 (= list_app 전체)
#   └─────────── 루트 앱

list_app = typer.Typer(
    help="컴포넌트 목록 조회 (models, adapters, preprocessors 등)",
    invoke_without_command=True,
)


@list_app.callback()
def list_callback(ctx: typer.Context) -> None:
    """서브커맨드 없이 호출 시 도움말 표시"""
    if ctx.invoked_subcommand is None:
        console = Console()
        console.print(ctx.get_help())
        raise typer.Exit(0)


list_app.command("adapters", help="사용 가능한 데이터 어댑터 목록")(list_adapters)
list_app.command("evaluators", help="사용 가능한 평가자 목록")(list_evaluators)
list_app.command("metrics", help="Task별 사용 가능한 평가 메트릭 목록")(list_metrics)
list_app.command("preprocessors", help="사용 가능한 전처리기 목록")(list_preprocessors)
list_app.command("models", help="사용 가능한 모델 목록")(list_models)

# list_app을 "list"라는 이름으로 루트 앱에 연결
app.add_typer(list_app, name="list")


if __name__ == "__main__":
    # 이 파일을 직접 실행(python main_commands.py)할 때의 진입점.
    # 일반적으로는 mmp.__main__.py 또는 콘솔 스크립트를 통해 실행된다.
    app()
