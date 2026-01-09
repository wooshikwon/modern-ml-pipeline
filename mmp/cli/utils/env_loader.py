"""
환경변수 파일 로더

config 파일명에서 env_name을 추출하여 대응되는 .env.{env_name} 파일을 로드합니다.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env_for_config(config_path: str) -> bool:
    """
    config 파일명에서 env_name을 추출하여 .env.{env_name} 파일 로드.

    Args:
        config_path: config 파일 경로 (예: "configs/local.yaml")

    Returns:
        bool: .env 파일 로드 성공 여부

    Examples:
        - config_path="configs/local.yaml" → .env.local 로드
        - config_path="configs/dev.yaml" → .env.dev 로드
        - config_path="configs/production.yaml" → .env.production 로드
    """
    from dotenv import load_dotenv

    env_name = Path(config_path).stem
    env_file = Path.cwd() / f".env.{env_name}"

    if env_file.exists():
        load_dotenv(env_file, override=True)
        logger.debug(f"환경변수 로드 완료: {env_file}")
        return True

    logger.debug(f"환경변수 파일 없음: {env_file} (환경변수가 이미 설정되어 있다면 무시 가능)")
    return False
