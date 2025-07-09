import logging
import sys
from pathlib import Path

# logging.handlers의 대표적 종류(더 많음)
# 1. RotatingFileHandler : 로그 파일 크기가 커지면 자동으로 파일을 덮어쓰는 핸들러#
# 2. TimedRotatingFileHandler : 특정 시간마다 새 로그 파일을 생성하는 핸들러
# 3. WatchedFileHandler : 파일 변경 시 새 로그 파일을 생성하는 핸들러
# 4. StreamHandler : 콘솔에 로그를 출력하는 핸들러

# 여기서는 일자별 로그 파일이 생성되는 방식으로 - RotatingFileHandler 사용
from logging.handlers import TimedRotatingFileHandler
from config.settings import settings

"""
1. 이 프로젝트(애플리케이션)의 전역 로거를 설정
    - 로거 설정 시, 로그 파일 경로를 설정하고, 로그 파일 크기를 설정하고, 로그 파일 로테이션 기간을 설정한다.
    - 로그 파일 경로는 config.yaml 파일에 정의된 환경 변수를 참조하여 설정한다.
    - 로그 파일 로테이션은 하루 단위로 생성되며, 30일까지 유지된다.
"""

def setup_logging():
    # logging.getLogger() 라는 ()안에 이름 없는 로거를 설정함으로써, 해당 객체가 파이썬 로깅 계층 구조의 '최상단'이 된다.
        # 최상단이라는 것의 의미는, 다른 파이썬 파일에서 logging.getlogger("다른 이름") 으로 로거를 생성하면, 그 로거는 이름 없는 root_logger의 하위에 위치한다.
        # 하위 로거는 별도 설정이 없다면. 상위 로거의 설정을 따라간다.
    root_logger = logging.getLogger()

    # 특정 파일에서 setup_logging을 할 때, 핸들러가 중복 등록되어 로그가 여러번 출력되는 현상을 막기 위해 setup_logging을 불러올 때마다, 기존 핸들러를 모두 제거한다.
    if root_logger.handlers:
        root_logger.handlers.clear()

    # run_mode가 local일 때, 로그 파일은 로컬 경로에 생성되며, 하루 단위로 생성된다.
    if settings.environment.run_mode == "local":
        root_logger.setLeveL(logging.DEBUG)
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"
        handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=30, encoding="utf-8")
    # run_mode가 그 외일 때, 로그 파일은 콘솔에 출력된다.
    else:
        root_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)

    # 위에서 설정한 규칙들을 logging의 methods를 통해 적용한다.
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

# 모듈을 import 할 때, 자동으로 '전역 로거 설정'이 되도록 한다. 
# 이 프로젝트에서는 pipelines/ 파일에서 from src.utils.logger import logger을 통해 1회 **로거 설정**을 한다.
# 다른 개별 파일에서는 import logging, logger = logging.getLogger(__name__)을 통해 해당 파일의 별도 로거 객체를 **생성**한다.
setup_logging()

