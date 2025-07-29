# src/cli/commands.py

import typer
import json
import shutil
from pathlib import Path
from typing_extensions import Annotated
from typing import Optional
import subprocess

from src.settings import Settings, load_settings_by_file, create_settings_for_inference, load_config_files
from src.pipelines import run_training, run_batch_inference
from src.serving import run_api_server
from src.utils.system.logger import setup_logging, logger
from src.engine import AdapterRegistry, EvaluatorRegistry, PreprocessorStepRegistry
from src.utils.system.catalog_parser import load_model_catalog

app = typer.Typer(
    help="Modern ML Pipeline - A robust tool for building and deploying ML.",
    rich_markup_mode="markdown"
)

# --- Main Commands ---

@app.command()
def train(
    recipe_file: Annotated[str, typer.Option(help="실행할 Recipe 파일 경로")],
    context_params: Annotated[Optional[str], typer.Option(help='Jinja 템플릿에 전달할 파라미터 (JSON)')] = None,
):
    """
    지정된 Recipe를 사용하여 학습 파이프라인을 실행합니다.
    """
    try:
        params = json.loads(context_params) if context_params else None
        settings = load_settings_by_file(recipe_file, context_params=params)
        setup_logging(settings)
        
        logger.info(f"'{recipe_file}' 레시피로 학습을 시작합니다.")
        logger.info(f"Run Name: {settings.recipe.model.computed['run_name']}")
        run_training(settings=settings, context_params=params)
        
    except Exception as e:
        logger.error(f"학습 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def batch_inference(
    run_id: Annotated[str, typer.Option(help="추론에 사용할 MLflow Run ID")],
    context_params: Annotated[Optional[str], typer.Option(help='Jinja 템플릿에 전달할 파라미터 (JSON)')] = None,
):
    """
    지정된 `run_id`의 모델을 사용하여 배치 추론을 실행합니다.
    """
    try:
        params = json.loads(context_params) if context_params else None
        # 추론 시점에는 config 파일만 로드하여 Settings 생성
        config_data = load_config_files()
        settings = create_settings_for_inference(config_data)
        setup_logging(settings)
        
        logger.info(f"Run ID '{run_id}'로 배치 추론을 시작합니다.")
        run_batch_inference(
            settings=settings,
            run_id=run_id,
            context_params=params,
        )
    except Exception as e:
        logger.error(f"배치 추론 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def serve_api(
    run_id: Annotated[str, typer.Option(help="서빙할 모델의 MLflow Run ID")],
    host: Annotated[str, typer.Option(help="바인딩할 호스트")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="바인딩할 포트")] = 8000,
):
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
):
    """
    현재 디렉토리에 `config/`와 `recipes/` 폴더 및 예제 파일을 생성합니다.
    """
    typer.echo("프로젝트 구조 초기화를 시작합니다...")
    
    source_path = Path(__file__).parent / "project_templates"
    destination_path = Path(dir)
    
    try:
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        typer.secho(f"✅ 성공: '{destination_path.resolve()}'에 기본 설정 파일들이 생성되었습니다.", fg=typer.colors.GREEN)
        
    except Exception as e:
        typer.secho(f"❌ 오류: 프로젝트 초기화 중 문제가 발생했습니다: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.command()
def validate(
    recipe_file: Annotated[str, typer.Option(help="검증할 Recipe 파일 경로")]
):
    """
    지정된 Recipe 파일과 관련 설정 파일들의 유효성을 검증합니다.
    """
    typer.echo(f"'{recipe_file}' 설정 파일 검증을 시작합니다...")
    try:
        load_settings_by_file(recipe_file)
        typer.secho(f"✅ 성공: 모든 설정 파일이 유효합니다.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"❌ 오류: 설정 파일 검증에 실패했습니다.", fg=typer.colors.RED)
        typer.echo(e)
        raise typer.Exit(code=1)

@app.command(name="test-contract")
def test_contract():
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

# --- List Commands Group ---

list_app = typer.Typer(help="사용 가능한 컴포넌트들의 '별명(type)' 목록을 확인합니다.")
app.add_typer(list_app, name="list")

@list_app.command("adapters")
def list_adapters():
    """사용 가능한 모든 데이터 어댑터의 별명 목록을 출력합니다."""
    typer.echo("✅ Available Adapters:")
    available_items = sorted(AdapterRegistry._adapters.keys())
    for item in available_items:
        typer.echo(f"- {item}")

@list_app.command("evaluators")
def list_evaluators():
    """사용 가능한 모든 평가자의 별명 목록을 출력합니다."""
    typer.echo("✅ Available Evaluators:")
    available_items = sorted(EvaluatorRegistry._evaluators.keys())
    for item in available_items:
        typer.echo(f"- {item}")

@list_app.command("preprocessors")
def list_preprocessors():
    """사용 가능한 모든 전처리기 블록의 별명 목록을 출력합니다."""
    typer.echo("✅ Available Preprocessor Steps:")
    available_items = sorted(PreprocessorStepRegistry._steps.keys())
    for item in available_items:
        typer.echo(f"- {item}")

@list_app.command("models")
def list_models():
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