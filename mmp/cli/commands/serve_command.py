"""
Serve Command Implementation
모듈화된 API 서빙 실행 명령
"""

import logging

import typer
from typing_extensions import Annotated

from mmp.cli.utils import CLIProgress
from mmp.cli.utils.env_loader import load_env_for_config
from mmp.cli.utils.system_checker import CheckStatus, SystemChecker
from mmp.serving import run_api_server
from mmp.settings import SettingsFactory, __version__
from mmp.utils.core.logger import get_current_log_file, log_error, log_sys

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
        # 0. 환경변수 로드 (config 파일명에서 env_name 추출)
        load_env_for_config(config_path)

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
            missing = dep_result.details.get("missing", []) if dep_result.details else []
            log_error(dep_result.message, "Dependencies")
            if missing:
                log_error(f"Missing: {', '.join(missing)}", "Dependencies")
            log_error(f"Solution: {dep_result.solution}", "Dependencies")
            raise typer.Exit(code=1)
        progress.step_done("verified")

        # 3. 서버 시작
        progress.step_start("Starting API server")
        log_sys(f"API Server:   http://{host}:{port}")
        log_sys(f"API Docs:     http://{host}:{port}/docs")
        log_sys(f"Health Check: http://{host}:{port}/health")
        progress.step_done()

        # API 서버 실행 (블로킹)
        run_api_server(settings=settings, run_id=run_id, host=host, port=port)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        progress.step_fail()
        log_sys("Server stopped by user")
        raise typer.Exit(code=0)
    except FileNotFoundError as e:
        progress.step_fail()
        log_error(f"File not found: {e}", "CLI")
        log_sys("Check the file path or use a valid Run ID")
        log_file = get_current_log_file()
        if log_file:
            log_sys(f"See: {log_file}")
        raise typer.Exit(code=1)
    except ValueError as e:
        progress.step_fail()
        log_error(f"Configuration error: {e}", "CLI")
        log_file = get_current_log_file()
        if log_file:
            log_sys(f"See: {log_file}")
        raise typer.Exit(code=1)
    except Exception as e:
        progress.step_fail()
        log_error(str(e), "CLI")
        log_file = get_current_log_file()
        if log_file:
            log_sys(f"See: {log_file}")
        raise typer.Exit(code=1)
