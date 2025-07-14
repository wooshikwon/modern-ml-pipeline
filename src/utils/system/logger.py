import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

# 순환 참조를 피하기 위해 타입 힌트만 임포트
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.settings import Settings

# 전역 로거 객체
logger = logging.getLogger(__name__)

def setup_logging(settings: "Settings"):
    """
    주입된 설정(settings) 객체를 기반으로 전역 로거를 설정합니다.
    """
    # logging.getLogger()는 이름 없는 루트 로거를 가져옵니다.
    root_logger = logging.getLogger()

    # 핸들러 중복 등록 방지
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    # 실행 모드에 따른 로그 레벨 및 핸들러 설정
    if settings.environment.app_env == "local":
        root_logger.setLevel(logging.DEBUG)
        log_dir = Path(__file__).resolve().parent.parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"
        handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=30, encoding="utf-8")
    else: # "dev", "prod" 등
        root_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)

    # 포매터 설정 및 핸들러 추가
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    logger.info(f"로거 설정 완료. 실행 환경: {settings.environment.app_env}") 