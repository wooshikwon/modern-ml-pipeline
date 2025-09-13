"""
Modern ML Pipeline CLI - Main Commands Router
단순 라우팅만 담당하는 메인 CLI 진입점

"""

import typer
import tomllib
from pathlib import Path
from typing_extensions import Annotated
from rich.console import Console
from rich.text import Text

# Command imports
from src.cli.commands.init_command import init_command
from src.cli.commands.get_config_command import get_config_command
from src.cli.commands.get_recipe_command import get_recipe_command
from src.cli.commands.system_check_command import system_check_command
from src.cli.commands.train_command import train_command
from src.cli.commands.inference_command import batch_inference_command
from src.cli.commands.serve_command import serve_api_command

# List commands
from src.cli.commands.list_commands import (
    list_adapters,
    list_evaluators,
    list_preprocessors,
    list_models
)

# ASCII Art Banner
ASCII_BANNER = """
███╗   ███╗███╗   ███╗████████╗ 
████╗ ████║████╗ ████║██╔════██╗
██╔████╔██║██╔████╔██║████████╔╝
██║╚██╔╝██║██║╚██╔╝██║██╔═════╝ 
██║ ╚═╝ ██║██║ ╚═╝ ██║██║     
╚═╝     ╚═╝╚═╝     ╚═╝╚═╝     
"""

console = Console()

def show_banner():
    """Display ASCII art banner when CLI is invoked."""
    banner_text = Text(ASCII_BANNER, style="bold cyan")
    console.print(banner_text)
    console.print("[bold yellow]Modern ML Pipeline[/bold yellow] - ML 워크플로우를 위한 CLI 도구\n")

def _get_version() -> str:
    """
    Read version from pyproject.toml file.

    Returns:
        str: Version string from pyproject.toml, defaults to "unknown" if not found
    """
    try:
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        project_data = pyproject_data.get("project")
        if project_data is not None and isinstance(project_data, dict):
            version = project_data.get("version")
            return str(version) if version is not None else "unknown"
        return "unknown"
    except Exception:
        return "unknown"


# Main CLI App with custom epilog for banner
epilog_text = """
[bold cyan]███╗   ███╗███╗   ███╗████████╗[/bold cyan]
[bold cyan]████╗ ████║████╗ ████║██╔════██╗[/bold cyan]
[bold cyan]██╔████╔██║██╔████╔██║████████╔╝[/bold cyan]
[bold cyan]██║╚██╔╝██║██║╚██╔╝██║██╔═════╝[/bold cyan]
[bold cyan]██║ ╚═╝ ██║██║ ╚═╝ ██║██║     [/bold cyan]
[bold cyan]╚═╝     ╚═╝╚═╝     ╚═╝╚═╝     [/bold cyan]

[bold yellow]Modern ML Pipeline[/bold yellow] - ML 워크플로우를 위한 CLI 도구
"""

app = typer.Typer(
    help="🚀 Modern ML Pipeline - Unified CLI Interface",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog=epilog_text
)


def version_callback(value: bool) -> None:
    """
    Callback function for --version option.

    Args:
        value: Boolean flag indicating if version was requested

    Raises:
        typer.Exit: Always exits after displaying version
    """
    if value:
        version = _get_version()
        typer.echo(f"modern-ml-pipeline {version}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=version_callback, help="Show version")
    ] = False,
) -> None:
    """
    Modern ML Pipeline CLI - Main entry point for all commands.

    A robust tool for building and deploying ML pipelines with configuration-driven approach.
    Supports training, inference, serving, and project initialization.

    Args:
        version: Show version information and exit
    """
    pass  # Main callback only handles version option


# ═══════════════════════════════════════════════════
# Project Management Commands
# ═══════════════════════════════════════════════════

app.command("init", help="프로젝트 초기화 - 기본 디렉토리 구조 및 파일 생성")(init_command)
app.command("get-config", help="대화형으로 환경별 설정 파일 생성")(get_config_command)
app.command("get-recipe", help="대화형 모델 선택 및 레시피 생성")(get_recipe_command)
app.command("system-check", help="현재 config 파일 기반 시스템 연결 상태 검사")(system_check_command)


# ═══════════════════════════════════════════════════
# ML Pipeline Commands
# ═══════════════════════════════════════════════════

app.command("train", help="학습 파이프라인 실행")(train_command)
app.command("batch-inference", help="배치 추론 실행")(batch_inference_command)
app.command("serve-api", help="API 서버 실행")(serve_api_command)


# ═══════════════════════════════════════════════════
# List Commands Group
# ═══════════════════════════════════════════════════

list_app = typer.Typer(help="사용 가능한 컴포넌트들의 목록을 확인합니다.")

list_app.command("adapters", help="사용 가능한 데이터 어댑터 목록")(list_adapters)
list_app.command("evaluators", help="사용 가능한 평가자 목록")(list_evaluators)
list_app.command("preprocessors", help="사용 가능한 전처리기 목록")(list_preprocessors)
list_app.command("models", help="사용 가능한 모델 목록")(list_models)

app.add_typer(list_app, name="list")


if __name__ == "__main__":
    app()