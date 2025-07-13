import typer
import json
from typing_extensions import Annotated
from typing import Optional

from src.settings.settings import load_settings, load_settings_by_file
from src.pipelines.train_pipeline import run_training
from src.pipelines.inference_pipeline import run_batch_inference
from serving.api import run_api_server
from src.utils.system.logger import setup_logging, logger

app = typer.Typer(help="현대적인 ML 파이프라인 CLI 도구")

@app.command()
def train(
    recipe_file: Annotated[str, typer.Option(help="Recipe 파일 경로 (확장자 제외)")],
    context_params: Annotated[Optional[str], typer.Option(help='실행 컨텍스트 파라미터 (JSON 문자열)')] = None,
):
    """
    지정��� 모델 이름의 레시피를 사용하여 학습 파이프라인을 실행합니다.
    """
    try:
        settings = load_settings_by_file(recipe_file)
        setup_logging(settings)
        params = json.loads(context_params) if context_params else {}
        
        logger.info(f"'{recipe_file}' 레시피로 학습을 시작합니다.")
        logger.info(f"생성될 Run Name: {settings.model.computed['run_name']}")
        run_training(settings=settings, context_params=params)
    except Exception as e:
        logger.error(f"학습 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def batch_inference(
    run_id: Annotated[str, typer.Option(help="추론에 사용할 MLflow Run ID")],
    context_params: Annotated[Optional[str], typer.Option(help='실행 컨텍스트 파라미터 (JSON 문자열)')] = None,
):
    """
    지정된 run_id의 모델을 사용하여 배치 추론을 실행합니다.
    예시: python main.py batch-inference --run-id "abc123def456"
    """
    try:
        params = json.loads(context_params) if context_params else {}
        logger.info(f"Run ID '{run_id}'로 배치 추론을 시작합니다.")
        
        run_batch_inference(
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
    지정된 run_id의 모델로 FastAPI 서버를 실행합니다.
    예시: python main.py serve-api --run-id "abc123def456"
    """
    try:
        logger.info(f"Run ID '{run_id}'로 API 서버를 시작합니다.")
        run_api_server(run_id=run_id, host=host, port=port)
    except Exception as e:
        logger.error(f"API 서버 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
