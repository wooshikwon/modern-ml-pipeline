# src/cli/commands.py

import typer
import json
import shutil
import tomllib
from pathlib import Path
from typing_extensions import Annotated
from typing import Optional, Dict, Any
# importlib.resources는 필요시에만 import
import subprocess
import inspect
import importlib
from jinja2 import Template
from datetime import datetime

from src.settings import load_settings_by_file, create_settings_for_inference, load_config_files
from src.pipelines import run_training, run_batch_inference
from src.serving import run_api_server
from src.utils.system.logger import setup_logging, logger
from src.engine import AdapterRegistry, EvaluatorRegistry
from src.components._preprocessor._registry import PreprocessorStepRegistry
from src.utils.system.catalog_parser import load_model_catalog
from src.settings.compatibility_maps import TASK_METRIC_COMPATIBILITY


def _get_templates_directory() -> Path:
    """
    Get the project templates directory path.
    
    Returns:
        Path: Path to the project templates directory
        
    Raises:
        FileNotFoundError: If templates directory cannot be found
    """
    # 먼저 패키지 리소스로 시도
    try:
        from src.cli.project_templates import TEMPLATES_DIR
        if TEMPLATES_DIR.exists():
            return TEMPLATES_DIR
    except ImportError:
        pass
    
    # fallback: 현재 파일 기준 경로
    fallback_path = Path(__file__).parent / "project_templates"
    if fallback_path.exists():
        return fallback_path
    
    raise FileNotFoundError(
        "프로젝트 템플릿 디렉토리를 찾을 수 없습니다. "
        "패키지 설치가 올바르지 않을 수 있습니다."
    )


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


app = typer.Typer(
    help="Modern ML Pipeline - A robust tool for building and deploying ML.",
    rich_markup_mode="markdown"
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
    version: Annotated[bool, typer.Option("--version", callback=version_callback, help="Show version")] = False,
) -> None:
    """
    Modern ML Pipeline CLI - Main entry point for all commands.
    
    A robust tool for building and deploying ML pipelines with configuration-driven approach.
    Supports training, inference, serving, and project initialization.
    
    Args:
        version: Show version information and exit
    """
    pass

# --- Guide Command ---

@app.command()
def guide(
    model_path: Annotated[str, typer.Argument(help="레시피 가이드를 생성할 모델의 전체 클래스 경로 (예: sklearn.ensemble.RandomForestClassifier)")]
) -> None:
    """
    지정된 모델 클래스에 대한 모범적인 레시피(recipe) 템플릿을 생성합니다.
    """
    try:
        # 1. 모델 클래스 동적 로드 및 인트로스펙션
        module_path, class_name = model_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        params = {}
        for name, p in inspect.signature(model_class.__init__).parameters.items():
            if name not in ('self', 'args', 'kwargs'):
                # 기본값이 없는 파라미터는 주석 처리된 플레이스홀더로 처리
                default_value = p.default if p.default != inspect.Parameter.empty else '# EDIT_ME'
                # 문자열 기본값에 따옴표 추가
                if isinstance(default_value, str):
                    value_str = f"'{default_value}'"
                else:
                    value_str = default_value

                params[name] = {
                     'type': str(p.annotation) if p.annotation != inspect.Parameter.empty else 'Any',
                     'default': p.default if p.default != inspect.Parameter.empty else 'N/A',
                     'value': value_str
                 }

        # 2. Task-type 추정 (간단한 규칙 기반)
        task_type = "classification" if "Classifier" in class_name else "regression" if "Regressor" in class_name else "unknown"
        
        # 3. Jinja2 템플릿 렌더링
        templates_dir = _get_templates_directory()
        template_path = templates_dir / "guideline_recipe.yaml.j2"
        template = Template(template_path.read_text())
        
        rendered_recipe = template.render(
            model_class_path=model_path,
            creation_date=datetime.now().strftime("%Y-%m-%d"),
            model_name=class_name.lower(),
            hyperparameters=params,
            task_type=task_type,
            recommended_metrics=TASK_METRIC_COMPATIBILITY.get(task_type, [])
        )
        
        typer.secho(f"--- Recipe 가이드: {model_path} ---\n", fg=typer.colors.CYAN)
        typer.echo(rendered_recipe)

    except (ImportError, AttributeError):
        typer.secho(f"오류: 모델 클래스 '{model_path}'를 찾을 수 없습니다.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"예상치 못한 오류 발생: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


# --- Main Commands ---

@app.command()
def train(
    recipe_file: Annotated[str, typer.Option(help="실행할 Recipe 파일 경로")],
    context_params: Annotated[Optional[str], typer.Option(help='Jinja 템플릿에 전달할 파라미터 (JSON)')] = None,
) -> None:
    """
    지정된 Recipe를 사용하여 학습 파이프라인을 실행합니다.
    """
    try:
        params: Optional[Dict[str, Any]] = json.loads(context_params) if context_params else None
        settings = load_settings_by_file(recipe_file, context_params=params)
        setup_logging(settings)
        
        logger.info(f"'{recipe_file}' 레시피로 학습을 시작합니다.")
        computed = settings.recipe.model.computed
        run_name = computed.get('run_name', 'unknown') if computed else 'unknown'
        logger.info(f"Run Name: {run_name}")
        run_training(settings=settings, context_params=params)
        
    except Exception as e:
        logger.error(f"학습 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def batch_inference(
    run_id: Annotated[str, typer.Option(help="추론에 사용할 MLflow Run ID")],
    context_params: Annotated[Optional[str], typer.Option(help='Jinja 템플릿에 전달할 파라미터 (JSON)')] = None,
) -> None:
    """
    지정된 `run_id`의 모델을 사용하여 배치 추론을 실행합니다.
    """
    try:
        params: Optional[Dict[str, Any]] = json.loads(context_params) if context_params else None
        # 추론 시점에는 config 파일만 로드하여 Settings 생성
        config_data = load_config_files()
        settings = create_settings_for_inference(config_data)
        setup_logging(settings)
        
        logger.info(f"Run ID '{run_id}'로 배치 추론을 시작합니다.")
        run_batch_inference(
            settings=settings,
            run_id=run_id,
            context_params=params or {},
        )
    except Exception as e:
        logger.error(f"배치 추론 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def serve_api(
    run_id: Annotated[str, typer.Option(help="서빙할 모델의 MLflow Run ID")],
    host: Annotated[str, typer.Option(help="바인딩할 호스트")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="바인딩할 포트")] = 8000,
) -> None:
    """
    지정된 `run_id`의 모델로 FastAPI 서버를 실행합니다.
    """
    try:
        # 추론 시점에는 config 파일만 로드하여 Settings 생성
        config_data = load_config_files()
        settings = create_settings_for_inference(config_data)
        setup_logging(settings)
        
        logger.info(f"Run ID '{run_id}'로 API 서버를 시작합니다.")
        run_api_server(settings=settings, run_id=run_id, host=host, port=port)
    except Exception as e:
        logger.error(f"API 서버 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def init(
    dir: Annotated[str, typer.Option(help="프로젝트 구조를 생성할 디렉토리")] = "."
) -> None:
    """
    현재 디렉토리에 `config/`와 `recipes/` 폴더 및 예제 파일을 생성합니다.
    """
    typer.echo("프로젝트 구조 초기화를 시작합니다...")
    
    try:
        # 템플릿 디렉토리를 robust한 방식으로 찾기
        source_path = _get_templates_directory()
        destination_path = Path(dir)
        
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        typer.secho(f"✅ 성공: '{destination_path.resolve()}'에 기본 설정 파일들이 생성되었습니다.", fg=typer.colors.GREEN)
        
    except FileNotFoundError as e:
        typer.secho(f"❌ 오류: {e}", fg=typer.colors.RED)
        typer.secho("패키지가 올바르게 설치되었는지 확인하세요.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"❌ 오류: 프로젝트 초기화 중 문제가 발생했습니다: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.command()
def validate(
    recipe_file: Annotated[str, typer.Option(help="검증할 Recipe 파일 경로")]
) -> None:
    """
    지정된 Recipe 파일과 관련 설정 파일들의 유효성을 검증합니다.
    """
    typer.echo(f"'{recipe_file}' 설정 파일 검증을 시작합니다...")
    try:
        load_settings_by_file(recipe_file)
        typer.secho("✅ 성공: 모든 설정 파일이 유효합니다.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho("❌ 오류: 설정 파일 검증에 실패했습니다.", fg=typer.colors.RED)
        typer.echo(e)
        raise typer.Exit(code=1)

@app.command(name="test-contract")
def test_contract() -> None:
    """
    `tests/integration/test_dev_contract.py`를 실행하여 인프라 계약을 테스트합니다.
    """
    typer.echo("인프라 계약 테스트를 시작합니다...")
    try:
        result = subprocess.run(
            ["pytest", "tests/integration/test_dev_contract.py"],
            capture_output=True, text=True, check=True
        )
        typer.echo(result.stdout)
        typer.secho("✅ 성공: 모든 인프라 계약 테스트를 통과했습니다.", fg=typer.colors.GREEN)
    except subprocess.CalledProcessError as e:
        typer.secho("❌ 오류: 인프라 계약 테스트에 실패했습니다.", fg=typer.colors.RED)
        typer.echo(e.stdout)
        typer.echo(e.stderr)
        raise typer.Exit(code=1)
    except FileNotFoundError:
        typer.secho("❌ 오류: `pytest`를 찾을 수 없습니다. 테스트 의존성이 설치되었는지 확인하세요.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.command(name="self-check")
def self_check(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="상세한 검사 결과 출력")] = False
) -> None:
    """
    시스템 환경과 의존성을 종합적으로 검사하여 건강 상태를 보고합니다.
    
    Environment, MLflow, External Services 등의 연결성과 설정을 검증하고,
    문제 발견 시 해결 방법을 제안합니다.
    """
    from src.health.checker import HealthCheckOrchestrator
    from src.health.reporter import HealthReporter
    from src.health.models import HealthCheckConfig
    
    typer.echo("🏥 시스템 건강 상태 검사를 시작합니다...\n")
    
    try:
        # Health check 설정 생성
        config = HealthCheckConfig(
            verbose=verbose,
            use_colors=True
        )
        
        # Health check 실행
        orchestrator = HealthCheckOrchestrator(config)
        summary = orchestrator.run_all_checks()
        
        # 결과 출력
        reporter = HealthReporter(config)
        reporter.display_summary(summary)
        
        # 전체 상태에 따라 종료 코드 설정
        if not summary.overall_healthy:
            raise typer.Exit(code=1)
            
    except Exception as e:
        typer.secho(f"❌ 건강 상태 검사 중 오류 발생: {e}", fg=typer.colors.RED)
        if verbose:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(code=1) 

# --- List Commands Group ---

list_app = typer.Typer(help="사용 가능한 컴포넌트들의 '별명(type)' 목록을 확인합니다.")
app.add_typer(list_app, name="list")

@list_app.command("adapters")
def list_adapters() -> None:
    """사용 가능한 모든 데이터 어댑터의 별명 목록을 출력합니다."""
    typer.echo("✅ Available Adapters:")
    available_items = sorted(AdapterRegistry._adapters.keys())
    for item in available_items:
        typer.echo(f"- {item}")

@list_app.command("evaluators")
def list_evaluators() -> None:
    """사용 가능한 모든 평가자의 별명 목록을 출력합니다."""
    typer.echo("✅ Available Evaluators:")
    available_items = sorted(EvaluatorRegistry._evaluators.keys())
    for item in available_items:
        typer.echo(f"- {item}")

@list_app.command("preprocessors")
def list_preprocessors() -> None:
    """사용 가능한 모든 전처리기 블록의 별명 목록을 출력합니다."""
    typer.echo("✅ Available Preprocessor Steps:")
    available_items = sorted(PreprocessorStepRegistry._steps.keys())
    for item in available_items:
        typer.echo(f"- {item}")

@list_app.command("models")
def list_models() -> None:
    """src/models/catalog.yaml에 등록된 사용 가능한 모델 목록을 출력합니다."""
    typer.echo("✅ Available Models from Catalog:")
    model_catalog = load_model_catalog()
    if not model_catalog:
        typer.secho("Error: src/models/catalog.yaml 파일을 찾을 수 없거나 내용이 비어있습니다.", fg="red")
        raise typer.Exit(1)
    
    for category, models in model_catalog.items():
        typer.secho(f"\n--- {category} ---", fg=typer.colors.CYAN)
        for model_info in models:
            typer.echo(f"- {model_info['class_path']} ({model_info['library']})") 