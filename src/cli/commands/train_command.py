"""
Train Command Implementation
모듈화된 학습 파이프라인 실행 명령

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

from src.settings import load_settings
from src.pipelines import run_training
from src.utils.system.logger import setup_logging, logger
from src.cli.utils.config_loader import load_environment


def train_command(
    recipe_file: Annotated[
        str, 
        typer.Option("--recipe-file", "-r", help="실행할 Recipe 파일 경로")
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
    학습 파이프라인 실행 (v2.0).
    
    환경 설정과 Recipe를 조합하여 모델 학습을 수행합니다.
    Recipe는 환경 독립적이며, --env-name으로 지정한 환경 설정과 결합됩니다.
    
    Args:
        recipe_file: Recipe YAML 파일 경로
        env_name: 사용할 환경 이름 (configs/{env_name}.yaml)
        context_params: 추가 파라미터 (JSON 형식)
    
    Examples:
        mmp train --recipe-file recipes/model.yaml --env-name dev
        mmp train -r recipes/model.yaml -e prod --params '{"date": "2024-01-01"}'
        
    Raises:
        typer.Exit: 파일을 찾을 수 없거나 실행 중 오류 발생 시
    """
    try:
        # 1. 환경변수 로드
        env_file = Path(f".env.{env_name}")
        if env_file.exists():
            load_environment(env_name)
            logger.info(f"환경 변수 로드: .env.{env_name}")
        
        # 2. Settings 생성 (config + recipe 조합)
        params: Optional[Dict[str, Any]] = (
            json.loads(context_params) if context_params else None
        )
        settings = load_settings(
            recipe_file, 
            env_name  # v2.0에서 필수 파라미터
        )
        setup_logging(settings)

        # 3. 학습 정보 로깅
        logger.info(f"환경 '{env_name}'에서 학습을 시작합니다.")
        logger.info(f"Recipe: {recipe_file}")
        computed = settings.recipe.model.computed
        run_name = computed.get("run_name", "unknown") if computed else "unknown"
        logger.info(f"Run Name: {run_name}")
        
        # 4. 학습 파이프라인 실행
        run_training(settings=settings, context_params=params)
        
        logger.info("✅ 학습이 성공적으로 완료되었습니다.")

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"학습 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)