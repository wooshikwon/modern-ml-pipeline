import typer
import json
from pathlib import Path
from typing_extensions import Annotated
from typing import Optional
import subprocess

from src.settings import Settings, load_settings_by_file, load_config_files
from src.pipelines.train_pipeline import run_training
from src.pipelines.inference_pipeline import run_batch_inference
from serving.api import run_api_server
from src.utils.system.logger import setup_logging, logger

# --- 기본 설정 파일 내용 ---
# (init 커맨드에서 사용될 기본 설정 파일의 내용을 여기에 정의합니다)
DEFAULT_DATA_ADAPTERS_YAML = """
# config/data_adapters.yaml
# 어떤 기술을 사용할지 정의합니다.
data_adapters:
  # Phase 2 통합 어댑터 사용을 기본으로 설정합니다.
  default_loader: storage
  default_storage: storage
  adapters:
    storage:
      class_name: StorageAdapter
      config: {}
    sql:
      class_name: SqlAdapter
      config:
        # 이 URI는 base.yaml이나 dev.yaml에서 덮어써야 합니다.
        connection_uri: "postgresql://user:pass@localhost:5432/db"
"""

DEFAULT_BASE_YAML = """
# config/base.yaml
# 어디에 연결할지 등 인프라 정보를 정의합니다.
environment:
  app_env: ${APP_ENV:local}

mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI:./mlruns}
  experiment_name: "Default-Experiment"
"""

DEFAULT_RECIPE_YAML = """
# recipes/example_recipe.yaml
# 모델의 논리를 정의합니다.
model:
  class_path: "sklearn.ensemble.RandomForestClassifier"
  hyperparameters:
    n_estimators: 100
    max_depth: 10

  loader:
    name: "default_loader" # 필수 필드 추가
    # 로컬 파일 시스템을 사용하는 예제
    source_uri: "data/raw/your_data.parquet"
  
  # 피처 증강은 필요 시 여기에 추가합니다.
  # augmenter: ...
  
  preprocessor:
    name: "default_preprocessor" # 필수 필드 추가
    params: # 필수 필드 추가
      exclude_cols: ["user_id"]
  
  data_interface:
    task_type: "classification"
    target_col: "target"
"""


app = typer.Typer(
    help="Modern ML Pipeline - A robust tool for building and deploying ML.",
    rich_markup_mode="markdown"
)

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
        logger.info(f"Run Name: {settings.recipe.model.computed['run_name']}")  # 🔄 Phase 1: 새 구조 반영
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
        
        # 추론 시점에는 모델 설정이 없으므로, 유효성 검사를 통과하기 위한 최소한의 더미 값을 추가
        if "model" not in config_data:
            config_data["model"] = {
                "class_path": "dummy.path",
                "loader": {"name": "dummy", "source_uri": "dummy"},
                "data_interface": {"task_type": "dummy"},
                "hyperparameters": {}
            }
            
        settings = Settings(**config_data)
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

        # 추론 시점에는 모델 설정이 없으므로, 유효성 검사를 통과하기 위한 최소한의 더미 값을 추가
        if "model" not in config_data:
            config_data["model"] = {
                "class_path": "dummy.path",
                "loader": {"name": "dummy", "source_uri": "dummy"},
                "data_interface": {"task_type": "dummy"},
                "hyperparameters": {}
            }

        settings = Settings(**config_data)
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
    base_path = Path(dir)
    config_path = base_path / "config"
    recipes_path = base_path / "recipes"
    
    try:
        config_path.mkdir(parents=True, exist_ok=True)
        recipes_path.mkdir(parents=True, exist_ok=True)
        
        (config_path / "data_adapters.yaml").write_text(DEFAULT_DATA_ADAPTERS_YAML)
        (config_path / "base.yaml").write_text(DEFAULT_BASE_YAML)
        (recipes_path / "example_recipe.yaml").write_text(DEFAULT_RECIPE_YAML)
        
        typer.secho(f"✅ 성공: '{base_path.resolve()}'에 기본 설정 파일들이 생성되었습니다.", fg=typer.colors.GREEN)
        
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


if __name__ == "__main__":
    app()
