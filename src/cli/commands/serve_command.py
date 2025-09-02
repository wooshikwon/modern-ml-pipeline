"""
Serve Command Implementation
모듈화된 API 서빙 실행 명령

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 단일 책임 원칙
"""

import typer
from typing_extensions import Annotated
from pathlib import Path

from src.settings import create_settings_for_inference, load_config_files
from src.serving import run_api_server
from src.utils.system.logger import setup_logging, logger
from src.cli.utils.config_loader import load_environment


def serve_api_command(
    run_id: Annotated[
        str, 
        typer.Option("--run-id", help="서빙할 모델의 MLflow Run ID")
    ],
    env_name: Annotated[
        str,
        typer.Option("--env-name", "-e", help="환경 이름 (필수)")
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
    API 서버 실행 (v2.0).
    
    MLflow에 저장된 모델을 REST API로 서빙합니다.
    환경 설정에 따라 인증, 로깅, 모니터링 등이 구성됩니다.
    
    Args:
        run_id: MLflow Run ID (서빙할 모델)
        env_name: 사용할 환경 이름 (configs/{env_name}.yaml)
        host: API 서버 호스트 (기본: 0.0.0.0)
        port: API 서버 포트 (기본: 8000)
    
    Examples:
        mmp serve-api --run-id abc123 --env-name prod
        mmp serve-api --run-id abc123 -e dev --host localhost --port 8080
        
    API Endpoints:
        - GET /health: 헬스 체크
        - POST /predict: 예측 요청
        - GET /model/info: 모델 정보
        
    Raises:
        typer.Exit: 파일을 찾을 수 없거나 실행 중 오류 발생 시
    """
    try:
        # 1. 환경변수 로드
        env_file = Path(f".env.{env_name}")
        if env_file.exists():
            load_environment(env_name)
            logger.info(f"환경 변수 로드: .env.{env_name}")
        
        # 2. Config 로드 및 Settings 생성
        config_data = load_config_files(env_name=env_name)
        settings = create_settings_for_inference(config_data)
        setup_logging(settings)

        # 3. 서버 정보 로깅
        logger.info(f"환경 '{env_name}'에서 API 서버를 시작합니다.")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Server: {host}:{port}")
        logger.info(f"API Documentation: http://{host}:{port}/docs")
        
        # 4. API 서버 실행
        run_api_server(
            settings=settings, 
            run_id=run_id, 
            host=host, 
            port=port
        )
        
    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"API 서버 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)