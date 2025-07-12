import typer
import json
from typing_extensions import Annotated
from typing import Optional

from src.settings.settings import load_settings
from src.pipelines.train_pipeline import run_training
from src.pipelines.inference_pipeline import run_batch_inference
from serving.api import run_api_server
from src.utils.logger import setup_logging, logger

app = typer.Typer(help="현대적인 ML 파이프라인 CLI 도구")

@app.command()
def train(
    model_name: Annotated[str, typer.Option(help="Recipe 파일과 동일한 모델 이름")],
    context_params: Annotated[Optional[str], typer.Option(help='실행 컨텍스트 파라미터 (JSON 문자열)')] = None,
):
    """
    지정��� 모델 이름의 레시피를 사용하여 학습 파이프라인을 실행합니다.
    """
    try:
        settings = load_settings(model_name)
        setup_logging(settings)
        params = json.loads(context_params) if context_params else {}
        logger.info(f"'{model_name}' 모델 학습을 시작합니다. 컨텍스트: {params}")
        run_training(settings=settings, context_params=params)
    except Exception as e:
        logger.error(f"학습 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def batch_inference(
    model_name: Annotated[str, typer.Option(help="추론에 사용할 모델의 레시피 이름")],
    run_id: Annotated[str, typer.Option(help="아티팩트를 가져올 MLflow Run ID")],
    context_params: Annotated[Optional[str], typer.Option(help='실행 컨텍스트 파라미터 (JSON 문자열)')] = None,
):
    """
    지정된 run_id의 아티팩트를 사용하여 배치 추론을 실행하고,
    결과를 Wrapper에 내장된 레시피 스냅샷의 설정에 따라 저장합니다.
    """
    try:
        # 배치 추론 시에는 전체 settings가 필요 없으므로, 로깅만 간단히 설정
        # setup_logging() # 이 부분은 로거를 어떻게 처리할지 정책에 따라 결정
        params = json.loads(context_params) if context_params else {}
        logger.info(f"'{model_name}' 모델의 배치 추론을 시작합니다. (Run ID: {run_id}, 컨텍스트: {params})")
        run_batch_inference(
            model_name=model_name,
            run_id=run_id,
            context_params=params,
        )
    except Exception as e:
        logger.error(f"배치 추론 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def serve_api(
    model_name: Annotated[str, typer.Option(help="서빙할 모델의 이름")] = "xgboost_x_learner",
    host: Annotated[str, typer.Option(help="바인딩할 호스트")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="바인딩할 포트")] = 8000,
):
    """
    지정된 모델로 FastAPI 서버를 실행합니다.
    """
    try:
        settings = load_settings(model_name)
        setup_logging(settings) # 로거 설정 주입
        logger.info(f"'{model_name}' 모델을 서빙하는 API 서버를 시작합니다.")
        run_api_server(settings=settings, host=host, port=port)
    except Exception as e:
        logger.error(f"API 서버 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
