# src/cli/commands.py
# Modern ML Pipeline CLI - Unified Command Interface
# CLAUDE.md 원칙 준수: TDD, 타입 힌트, Google Style Docstring

import typer
import json
import tomllib
import yaml
from pathlib import Path
from typing_extensions import Annotated
from typing import Optional, Dict, Any

# Import subprocess for external commands

# Modern CLI commands
from src.cli.commands.system_check_command import system_check_command
from src.cli.commands.get_recipe_command import get_recipe_command
from src.cli.commands.get_config_command import get_config_command
from src.cli.commands.migrate_command import migrate_command
from src.cli.utils.env_loader import load_environment, get_env_name_with_fallback

# Core functionality imports 
from src.settings import (
    load_settings_by_file,
    create_settings_for_inference,
    load_config_files,
)
from src.pipelines import run_training, run_batch_inference
from src.serving import run_api_server
from src.utils.system.logger import setup_logging, logger
from src.utils.deprecation import show_deprecation_warning
from rich.console import Console

console = Console()
from src.components._adapter import AdapterRegistry
from src.components._evaluator import EvaluatorRegistry
from src.components._preprocessor._registry import PreprocessorStepRegistry
from src.utils.system.catalog_parser import load_model_catalog


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


# Main CLI App
app = typer.Typer(
    help="🚀 Modern ML Pipeline - Unified CLI Interface",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich"
)


@app.callback()
def main_callback():
    """
    Main callback executed before any command.
    Checks for legacy structure and shows deprecation warnings.
    """
    import warnings
    import sys
    import os
    from src.cli.commands.migrate_command import check_legacy_structure
    from rich.table import Table
    
    # Enable deprecation warnings if requested
    if os.getenv('SHOW_DEPRECATION_WARNINGS', 'true').lower() == 'true':
        warnings.filterwarnings('always', category=DeprecationWarning)
    
    # Python version check
    if sys.version_info < (3, 8):
        console.print(
            "[red]⚠️ Python 3.7 support is deprecated and will be removed in v2.0[/red]"
        )
    
    # Legacy structure check (only show occasionally to avoid spam)
    # Check if we should show legacy warnings (once per day)
    check_file = Path.home() / ".mmp_legacy_check"
    should_check = True
    
    if check_file.exists():
        import time
        last_check = check_file.stat().st_mtime
        current_time = time.time()
        # Show warning once per day
        should_check = (current_time - last_check) > 86400
    
    if should_check:
        legacy_items = check_legacy_structure()
        
        if legacy_items:
            console.print("\n[yellow]⚠️ Legacy structure detected:[/yellow]")
            table = Table(show_header=True, header_style="bold yellow")
            table.add_column("Found", style="red") 
            table.add_column("Should be", style="green")
            table.add_column("Type")
            
            for old, new, name in legacy_items[:3]:  # Show max 3 items
                table.add_row(old, new, name)
            
            if len(legacy_items) > 3:
                table.add_row("...", "...", f"and {len(legacy_items) - 3} more")
            
            console.print(table)
            console.print("\n[yellow]Run 'mmp migrate' to fix these issues[/yellow]\n")
        
        # Update check file
        check_file.touch()


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
    pass


# === Modern Commands (Phase 1-5) ===

# Phase 1: Get Config Command
app.command("get-config", help="대화형으로 환경별 설정 파일 생성")(get_config_command)
app.command("migrate", help="레거시 프로젝트 구조를 새 구조로 마이그레이션")(migrate_command)

# Phase 3: System Check Command  
app.command("system-check", help="현재 config 파일 기반 시스템 연결 상태 검사")(system_check_command)

# Phase 4: Get Recipe Command
app.command("get-recipe", help="대화형 모델 선택 및 레시피 생성")(get_recipe_command)

# Phase 5: Init Command
@app.command()
def init(
    project_name: Annotated[
        Optional[str], typer.Option(help="프로젝트 이름")
    ] = None,
    with_mmp_dev: Annotated[
        bool, typer.Option("--with-mmp-dev", help="mmp-local-dev 환경 설치")
    ] = False,
) -> None:
    """단순화된 프로젝트 초기화 (mmp-local-dev 통합)"""
    from src.cli.commands.init_command import create_project_structure, clone_mmp_local_dev
    from rich.console import Console
    
    console = Console()
    
    try:
        # Step 1: mmp-local-dev 설치 여부 확인
        if not with_mmp_dev:
            with_mmp_dev = typer.confirm(
                "🐳 mmp-local-dev를 함께 설치하시겠습니까? (PostgreSQL, Redis, MLflow 개발 환경)"
            )
        
        # Step 2: 프로젝트명 입력
        if not project_name:
            project_name = typer.prompt("📁 프로젝트 이름을 입력하세요")
        
        # Step 3: mmp-local-dev clone (상위 디렉토리)
        if with_mmp_dev:
            console.print("🔄 mmp-local-dev를 상위 디렉토리에 설치 중...")
            try:
                clone_mmp_local_dev(Path.cwd().parent)
                console.print("✅ mmp-local-dev가 상위 디렉토리에 설치되었습니다.", style="green")
            except Exception as e:
                console.print(f"⚠️ mmp-local-dev 설치 중 오류 발생: {e}", style="yellow")
                console.print("수동으로 설치하거나 건너뛸 수 있습니다.", style="yellow")
        
        # Step 4: 프로젝트 디렉토리 생성
        project_path = Path.cwd() / project_name
        create_project_structure(project_path, with_mmp_dev=with_mmp_dev)
        
        # Step 5: 성공 메시지
        console.print(f"🎉 프로젝트 '{project_name}'이 생성되었습니다!", style="green bold")
        console.print(f"📂 경로: {project_path.absolute()}")
        
        console.print("\n🚀 다음 단계:", style="cyan bold")
        console.print(f"   cd {project_name}")
        if with_mmp_dev:
            console.print("   cd ../mmp-local-dev && docker-compose up -d")
        console.print("   modern-ml-pipeline get-recipe")
        
    except KeyboardInterrupt:
        console.print("\n❌ 프로젝트 초기화가 취소되었습니다.", style="red")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"❌ 프로젝트 초기화 중 오류 발생: {e}", style="red")
        raise typer.Exit(code=1)


# === Core ML Pipeline Commands ===

@app.command()
def train(
    recipe_file: Annotated[
        str, 
        typer.Option("--recipe-file", "-r", help="실행할 Recipe 파일 경로")
    ],
    env_name: Annotated[
        Optional[str],
        typer.Option("--env-name", "-e", help="사용할 환경 이름")
    ] = None,
    context_params: Annotated[
        Optional[str], 
        typer.Option("--params", "-p", help="Jinja 템플릿 파라미터 (JSON)")
    ] = None,
) -> None:
    """
    지정된 환경과 Recipe로 학습 파이프라인 실행.
    
    Examples:
        mmp train --recipe-file recipes/model.yaml --env-name dev
        mmp train -r recipes/model.yaml -e prod --params '{"date": "2024-01-01"}'
    """
    try:
        # Deprecation warning if env_name not provided
        if env_name is None:
            console.print(
                "[yellow]⚠️ WARNING: Running without --env-name is deprecated![/yellow]\n"
                "[yellow]This feature will be removed in v2.0.[/yellow]\n"
                "[yellow]Please use: mmp train --recipe-file <file> --env-name <env>[/yellow]",
                style="yellow"
            )
            show_deprecation_warning(
                "Running train command without --env-name",
                alternative="mmp train --recipe-file <file> --env-name <env>",
                critical=True
            )
        
        # 환경 이름 결정 및 환경변수 로드
        env_name = get_env_name_with_fallback(env_name)
        load_environment(env_name)
        
        # Settings 생성 (Phase 0에서 구현한 env_name 파라미터 사용)
        params: Optional[Dict[str, Any]] = (
            json.loads(context_params) if context_params else None
        )
        settings = load_settings_by_file(
            recipe_file, 
            context_params=params,
            env_name=env_name  # Phase 0에서 추가한 파라미터
        )
        setup_logging(settings)

        logger.info(f"환경 '{env_name}'에서 학습을 시작합니다.")
        logger.info(f"Recipe: {recipe_file}")
        computed = settings.recipe.model.computed
        run_name = computed.get("run_name", "unknown") if computed else "unknown"
        logger.info(f"Run Name: {run_name}")
        
        run_training(settings=settings, context_params=params)

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"학습 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def batch_inference(
    run_id: Annotated[
        str, 
        typer.Option("--run-id", help="추론에 사용할 MLflow Run ID")
    ],
    env_name: Annotated[
        Optional[str],
        typer.Option("--env-name", "-e", help="사용할 환경 이름")
    ] = None,
    context_params: Annotated[
        Optional[str], 
        typer.Option("--params", "-p", help="Jinja 템플릿 파라미터 (JSON)")
    ] = None,
) -> None:
    """
    지정된 환경에서 배치 추론 실행.
    
    Examples:
        mmp batch-inference --run-id abc123 --env-name prod
        mmp batch-inference --run-id abc123 -e dev --params '{"date": "2024-01-01"}'
    """
    try:
        # Deprecation warning if env_name not provided
        if env_name is None:
            console.print(
                "[yellow]⚠️ WARNING: Running without --env-name is deprecated![/yellow]\n"
                "[yellow]This feature will be removed in v2.0.[/yellow]\n"
                "[yellow]Please use: mmp batch-inference --run-id <id> --env-name <env>[/yellow]",
                style="yellow"
            )
            show_deprecation_warning(
                "Running batch-inference command without --env-name",
                alternative="mmp batch-inference --run-id <id> --env-name <env>",
                critical=True
            )
        
        # 환경 이름 결정 및 환경변수 로드
        env_name = get_env_name_with_fallback(env_name)
        load_environment(env_name)
        
        params: Optional[Dict[str, Any]] = (
            json.loads(context_params) if context_params else None
        )
        
        # Config 로드 및 Settings 생성 (Phase 0 env_name 지원)
        config_data = load_config_files(env_name=env_name)
        settings = create_settings_for_inference(config_data)
        setup_logging(settings)

        logger.info(f"환경 '{env_name}'에서 배치 추론을 시작합니다.")
        logger.info(f"Run ID: {run_id}")
        
        run_batch_inference(
            settings=settings,
            run_id=run_id,
            context_params=params or {},
        )
    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"배치 추론 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def serve_api(
    run_id: Annotated[
        str, 
        typer.Option("--run-id", help="서빙할 모델의 MLflow Run ID")
    ],
    env_name: Annotated[
        Optional[str],
        typer.Option("--env-name", "-e", help="사용할 환경 이름")
    ] = None,
    host: Annotated[
        str, 
        typer.Option("--host", help="바인딩할 호스트")
    ] = "0.0.0.0",
    port: Annotated[
        int, 
        typer.Option("--port", help="바인딩할 포트")
    ] = 8000,
) -> None:
    """
    지정된 환경에서 API 서버 실행.
    
    Examples:
        mmp serve-api --run-id abc123 --env-name prod
        mmp serve-api --run-id abc123 -e dev --host localhost --port 8080
    """
    try:
        # Deprecation warning if env_name not provided
        if env_name is None:
            console.print(
                "[yellow]⚠️ WARNING: Running without --env-name is deprecated![/yellow]\n"
                "[yellow]This feature will be removed in v2.0.[/yellow]\n"
                "[yellow]Please use: mmp serve-api --run-id <id> --env-name <env>[/yellow]",
                style="yellow"
            )
            show_deprecation_warning(
                "Running serve-api command without --env-name",
                alternative="mmp serve-api --run-id <id> --env-name <env>",
                critical=True
            )
        
        # 환경 이름 결정 및 환경변수 로드
        env_name = get_env_name_with_fallback(env_name)
        load_environment(env_name)
        
        # Config 로드 및 Settings 생성 (Phase 0 env_name 지원)
        config_data = load_config_files(env_name=env_name)
        settings = create_settings_for_inference(config_data)
        setup_logging(settings)

        logger.info(f"환경 '{env_name}'에서 API 서버를 시작합니다.")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Server: {host}:{port}")
        
        run_api_server(settings=settings, run_id=run_id, host=host, port=port)
    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"API 서버 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)




# === List Commands Group ===

list_app = typer.Typer(help="사용 가능한 컴포넌트들의 '별명(type)' 목록을 확인합니다.")


@list_app.command("adapters")
def list_adapters() -> None:
    """사용 가능한 모든 데이터 어댑터의 별명 목록을 출력합니다."""
    typer.echo("✅ Available Adapters:")
    available_items = sorted(AdapterRegistry.list_adapters().keys())
    for item in available_items:
        typer.echo(f"- {item}")


@list_app.command("evaluators")
def list_evaluators() -> None:
    """사용 가능한 모든 평가자의 별명 목록을 출력합니다."""
    typer.echo("✅ Available Evaluators:")
    available_items = sorted(EvaluatorRegistry.get_available_tasks())
    for item in available_items:
        typer.echo(f"- {item}")


@list_app.command("preprocessors")
def list_preprocessors() -> None:
    """사용 가능한 모든 전처리기 블록의 별명 목록을 출력합니다."""
    typer.echo("✅ Available Preprocessor Steps:")
    available_items = sorted(PreprocessorStepRegistry._steps.keys())
    for item in available_items:
        typer.echo(f"- {item}")


def _load_catalog_from_directory() -> Dict[str, Any]:
    """src/models/catalog/ 디렉토리에서 모델 카탈로그를 로드합니다."""
    catalog_dir = Path(__file__).parent.parent / "models" / "catalog"
    if not catalog_dir.exists():
        return {}
    
    catalog = {}
    
    # 각 카테고리 디렉토리를 순회
    for category_dir in catalog_dir.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            catalog[category_name] = []
            
            # 각 모델 YAML 파일을 순회
            for model_file in category_dir.glob("*.yaml"):
                try:
                    with open(model_file, "r", encoding="utf-8") as f:
                        model_data = yaml.safe_load(f)
                        if model_data:
                            catalog[category_name].append(model_data)
                except Exception as e:
                    logger.warning(f"모델 파일 로드 실패: {model_file}, 오류: {e}")
                    continue
    
    return catalog


@list_app.command("models")
def list_models() -> None:
    """src/models/catalog/ 디렉토리에 등록된 사용 가능한 모델 목록을 출력합니다."""
    typer.echo("✅ Available Models from Catalog:")
    
    # 새로운 디렉토리 구조에서 로드 시도
    catalog_dir = Path(__file__).parent.parent / "models" / "catalog"
    if catalog_dir.exists():
        model_catalog = _load_catalog_from_directory()
    else:
        # Fallback: 기존 catalog.yaml 파일에서 로드
        model_catalog = load_model_catalog()
    
    if not model_catalog:
        typer.secho(
            "Error: src/models/catalog/ 디렉토리나 catalog.yaml 파일을 찾을 수 없거나 내용이 비어있습니다.",
            fg="red",
        )
        raise typer.Exit(1)

    for category, models in model_catalog.items():
        typer.secho(f"\n--- {category} ---", fg=typer.colors.CYAN)
        for model_info in models:
            class_path = model_info.get('class_path', 'Unknown')
            library = model_info.get('library', 'Unknown')
            typer.echo(f"- {class_path} ({library})")


# Register list command group
app.add_typer(list_app, name="list")

if __name__ == "__main__":
    app()