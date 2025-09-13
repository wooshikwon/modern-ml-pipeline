"""
Serve Command Implementation
모듈화된 API 서빙 실행 명령

"""

import typer
from typing_extensions import Annotated

from src.settings import SettingsFactory
from src.serving import run_api_server
from src.utils.core.console_manager import (
    cli_command_start, cli_command_error,
    cli_step_complete, get_rich_console
)


def serve_api_command(
    run_id: Annotated[
        str, 
        typer.Option("--run-id", help="서빙할 모델의 MLflow Run ID")
    ],
    config_path: Annotated[
        str,
        typer.Option("--config-path", "-c", help="Config 파일 경로")
    ],
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
    API 서버 실행 (Phase 5.3 리팩토링).
    
    MLflow에 저장된 모델을 REST API로 서빙합니다.
    Config 파일을 직접 지정하여 서버를 구성합니다.
    
    Args:
        run_id: MLflow Run ID (서빙할 모델)
        config_path: Config YAML 파일 경로
        host: API 서버 호스트 (기본: 0.0.0.0)
        port: API 서버 포트 (기본: 8000)
    
    Examples:
        mmp serve-api --run-id abc123 --config-path configs/prod.yaml
        mmp serve-api --run-id abc123 -c configs/dev.yaml --host localhost --port 8080
        
    API Endpoints:
        - GET /health: 헬스 체크
        - POST /predict: 예측 요청
        - GET /model/info: 모델 정보
        
    Raises:
        typer.Exit: 파일을 찾을 수 없거나 실행 중 오류 발생 시
    """
    try:
        cli_command_start("API Server", "모델 서빙 서버 시작")

        # 1. Settings 생성 (for serving)
        console = get_rich_console()
        with console.progress_tracker("setup", 2, "서버 환경 설정") as update:
            settings = SettingsFactory.for_serving(
                run_id=run_id,
                config_path=config_path
            )
            update(2)  # 설정 완료

        cli_step_complete("설정", f"Config: {config_path}, Run ID: {run_id}")

        # 2. 서버 시작 정보 표시
        console.log_milestone(f"🌐 API Server: http://{host}:{port}", "success")
        console.log_milestone(f"📜 API Documentation: http://{host}:{port}/docs", "info")
        console.log_milestone(f"🔍 Health Check: http://{host}:{port}/health", "info")

        # 3. API 서버 실행 (로그 출력은 run_api_server에서 처리)
        run_api_server(
            settings=settings,
            run_id=run_id,
            host=host,
            port=port
        )

    except FileNotFoundError as e:
        cli_command_error("API Server", f"파일을 찾을 수 없습니다: {e}",
                         "파일 경로를 확인하거나 올바른 Run ID를 사용하세요")
        raise typer.Exit(code=1)
    except ValueError as e:
        cli_command_error("API Server", f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        cli_command_error("API Server", f"실행 중 오류 발생: {e}")
        raise typer.Exit(code=1)