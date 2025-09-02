"""
Inference Command Implementation
모듈화된 배치 추론 실행 명령

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 단일 책임 원칙
"""

import json
from typing import Optional, Dict, Any
import typer
from typing_extensions import Annotated
from pathlib import Path

from src.settings import create_settings_for_inference, load_config_files
from src.pipelines import run_batch_inference
from src.utils.system.logger import setup_logging, logger
from src.cli.utils.config_loader import load_environment


def batch_inference_command(
    run_id: Annotated[
        str, 
        typer.Option("--run-id", help="추론에 사용할 MLflow Run ID")
    ],
    env_name: Annotated[
        str,
        typer.Option("--env-name", "-e", help="환경 이름 (필수)")
    ],
    context_params: Annotated[
        Optional[str], 
        typer.Option("--params", "-p", help="Jinja 템플릿 파라미터 (JSON)")
    ] = None,
) -> None:
    """
    배치 추론 실행 (v2.0).
    
    MLflow에 저장된 모델을 사용하여 배치 추론을 수행합니다.
    환경 설정을 통해 데이터 소스와 추론 결과 저장 위치를 결정합니다.
    
    Args:
        run_id: MLflow Run ID (학습된 모델)
        env_name: 사용할 환경 이름 (configs/{env_name}.yaml)
        context_params: 추가 파라미터 (JSON 형식)
    
    Examples:
        mmp batch-inference --run-id abc123 --env-name prod
        mmp batch-inference --run-id abc123 -e dev --params '{"date": "2024-01-01"}'
        
    Raises:
        typer.Exit: 파일을 찾을 수 없거나 실행 중 오류 발생 시
    """
    try:
        # 1. 환경변수 로드
        env_file = Path(f".env.{env_name}")
        if env_file.exists():
            load_environment(env_name)
            logger.info(f"환경 변수 로드: .env.{env_name}")
        
        # 2. 파라미터 파싱
        params: Optional[Dict[str, Any]] = (
            json.loads(context_params) if context_params else None
        )
        
        # 3. Config 로드 및 Settings 생성
        config_data = load_config_files(env_name=env_name)
        settings = create_settings_for_inference(config_data)
        setup_logging(settings)

        # 4. 추론 정보 로깅
        logger.info(f"환경 '{env_name}'에서 배치 추론을 시작합니다.")
        logger.info(f"Run ID: {run_id}")
        
        # 5. 배치 추론 실행
        run_batch_inference(
            settings=settings,
            run_id=run_id,
            context_params=params or {},
        )
        
        logger.info("✅ 배치 추론이 성공적으로 완료되었습니다.")
        
    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"배치 추론 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)