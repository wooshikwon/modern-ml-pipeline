"""
Serve Command Implementation
모듈화된 API 서빙 실행 명령
"""

import logging

import typer
from typing_extensions import Annotated

from src.cli.utils import CLIProgress
from src.cli.utils.system_checker import CheckStatus, SystemChecker
from src.serving import run_api_server
from src.settings import SettingsFactory, __version__
from src.utils.core.logger import get_current_log_file

logger = logging.getLogger(__name__)


def serve_api_command(
    run_id: Annotated[str, typer.Option("--run-id", help="서빙할 모델의 MLflow Run ID")],
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config 파일 경로")],
    host: Annotated[str, typer.Option("--host", help="바인딩할 호스트")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="바인딩할 포트")] = 8000,
) -> None:
    """
    API 서버 실행.

    MLflow에 저장된 모델을 REST API로 서빙합니다.
    Config 파일을 직접 지정하여 서버를 구성합니다.

    Args:
        run_id: MLflow Run ID (서빙할 모델)
        config_path: Config YAML 파일 경로
        host: API 서버 호스트 (기본: 0.0.0.0)
        port: API 서버 포트 (기본: 8000)

    Examples:
        mmp serve-api --run-id abc123 --config configs/prod.yaml
        mmp serve-api --run-id abc123 -c configs/dev.yaml --host localhost --port 8080

    API Endpoints:
        - GET /health: 헬스 체크
        - POST /predict: 예측 요청
        - GET /model/info: 모델 정보

    Raises:
        typer.Exit: 파일을 찾을 수 없거나 실행 중 오류 발생 시
    """
    progress = CLIProgress(total_steps=3)
    progress.header(__version__)

    try:
        # 1. 설정 로드
        progress.step_start("Loading settings")
        settings = SettingsFactory.for_serving(run_id=run_id, config_path=config_path)
        progress.step_done(f"run_id: {run_id[:8]}...")

        # 2. 패키지 의존성 체크
        progress.step_start("Checking package deps")
        config_dict = settings.config.model_dump() if hasattr(settings.config, "model_dump") else {}
        recipe_dict = (
            settings.recipe.model_dump()
            if hasattr(settings, "recipe")
            and settings.recipe
            and hasattr(settings.recipe, "model_dump")
            else {}
        )
        env_name = getattr(getattr(settings.config, "environment", None), "name", "local")

        checker = SystemChecker(config_dict, env_name, config_path, recipe=recipe_dict)
        dep_result = checker.check_package_dependencies()

        if dep_result.status == CheckStatus.FAILED:
            progress.step_fail()
            logger.error(f"  {dep_result.message}")
            logger.error(f"  Solution: {dep_result.solution}")
            raise typer.Exit(code=1)
        progress.step_done("verified")

        # 3. 서버 시작
        progress.step_start("Starting API server")
        progress.step_done()

        # 서버 정보 출력
        logger.info(f"\n      API Server:   http://{host}:{port}")
        logger.info(f"      API Docs:     http://{host}:{port}/docs")
        logger.info(f"      Health Check: http://{host}:{port}/health\n")

        # API 서버 실행 (블로킹)
        run_api_server(settings=settings, run_id=run_id, host=host, port=port)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        logger.info("\n  Server stopped by user")
        raise typer.Exit(code=0)
    except FileNotFoundError as e:
        progress.step_fail()
        logger.error(f"  File not found: {e}")
        logger.error("  Check the file path or use a valid Run ID")
        log_file = get_current_log_file()
        if log_file:
            logger.error(f"  See: {log_file}")
        raise typer.Exit(code=1)
    except ValueError as e:
        progress.step_fail()
        logger.error(f"  Configuration error: {e}")
        log_file = get_current_log_file()
        if log_file:
            logger.error(f"  See: {log_file}")
        raise typer.Exit(code=1)
    except Exception as e:
        progress.step_fail()
        logger.error(f"  {e}")
        log_file = get_current_log_file()
        if log_file:
            logger.error(f"  See: {log_file}")
        raise typer.Exit(code=1)
