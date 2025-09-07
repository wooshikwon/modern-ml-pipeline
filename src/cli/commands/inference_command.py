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
from src.pipelines.inference_pipeline import run_inference_pipeline
from src.utils.system.logger import setup_logging, logger
from src.cli.utils.config_loader import load_environment


def batch_inference_command(
    run_id: Annotated[
        str, 
        typer.Option("--run-id", help="추론에 사용할 MLflow Run ID")
    ],
    config_path: Annotated[
        str,
        typer.Option("--config-path", "-c", help="Config 파일 경로")
    ],
    data_path: Annotated[
        str,
        typer.Option("--data-path", "-d", help="추론 데이터 파일 경로")
    ],
    context_params: Annotated[
        Optional[str], 
        typer.Option("--params", "-p", help="Jinja 템플릿 파라미터 (JSON)")
    ] = None,
) -> None:
    """
    배치 추론 실행 (Phase 5.3 리팩토링).
    
    MLflow에 저장된 모델을 사용하여 배치 추론을 수행합니다.
    --data-path로 추론 데이터를 직접 지정합니다.
    
    Args:
        run_id: MLflow Run ID (학습된 모델)
        config_path: Config YAML 파일 경로
        data_path: 추론 데이터 파일 경로
        context_params: 추가 파라미터 (JSON 형식)
    
    Examples:
        mmp batch-inference --run-id abc123 --config-path configs/prod.yaml --data-path data/inference.csv
        mmp batch-inference --run-id abc123 -c configs/dev.yaml -d queries/inference.sql --params '{"date": "2024-01-01"}'
        
    Raises:
        typer.Exit: 파일을 찾을 수 없거나 실행 중 오류 발생 시
    """
    try:        
        # 1. 파라미터 파싱
        params: Optional[Dict[str, Any]] = (
            json.loads(context_params) if context_params else None
        )
        
        # 2. Config 로드 및 Settings 생성
        config_data = load_config_files(config_path=config_path)
        settings = create_settings_for_inference(config_data)
        setup_logging(settings)

        # 3. 추론 정보 로깅
        logger.info(f"Config: {config_path}")
        logger.info(f"Data: {data_path}")
        logger.info(f"Run ID: {run_id}")
        
        # 4. 배치 추론 실행 (data_path 직접 전달)
        run_inference_pipeline(
            settings=settings,
            run_id=run_id,
            data_path=data_path,
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