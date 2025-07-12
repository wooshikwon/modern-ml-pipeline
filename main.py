import typer
from typing_extensions import Annotated

from src.settings.settings import load_settings
from src.pipelines.train_pipeline import run_training
from src.pipelines.inference_pipeline import run_batch_inference
from serving.api import run_api_server
from src.utils.logger import setup_logging, logger

app = typer.Typer(help="현대적인 ML 파이프라인 CLI 도구")

@app.command()
def train(
    model_name: Annotated[str, typer.Option(help="Recipe 파일과 동일한 모델 이름")],
    loader_name: Annotated[str, typer.Option(help="사용할 데이터 로더의 이름")] = "user_features",
):
    """
    지정된 모델 이름의 레시피를 사용하여 학습 파이프라인을 실행합니다.
    """
    try:
        settings = load_settings(model_name)
        setup_logging(settings) # 로거 설정 주입
        logger.info(f"'{model_name}' 모델 학습을 시작합니다.")
        run_training(settings=settings, loader_name=loader_name)
    except Exception as e:
        logger.error(f"학습 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def batch_inference(
    model_name: Annotated[str, typer.Option(help="추론에 사용할 모델의 레시피 이름")],
    loader_name: Annotated[str, typer.Option(help="사용할 데이터 로더의 이름")] = "user_features",
    model_stage: Annotated[str, typer.Option(help="사용할 모델의 스테이지 (e.g., 'Production')")] = "Production",
    output_table_id: Annotated[str, typer.Option(help="결과를 저장할 BigQuery 테이블 ID")],
):
    """
    지정된 모델과 데이터로 배치 추론을 실행하고 결과를 BigQuery에 저장합니다.
    """
    try:
        settings = load_settings(model_name)
        setup_logging(settings) # 로거 설정 주입
        logger.info(f"'{model_name}' 모델 설정으로 배치 추론을 시작합니다.")
        run_batch_inference(
            settings=settings,
            model_name=model_name,
            model_stage=model_stage,
            loader_name=loader_name,
            output_table_id=output_table_id
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
