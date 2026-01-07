import logging
import logging.handlers
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.settings import Settings

# Pydantic 및 라이브러리 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
warnings.filterwarnings("ignore", message=".*json_schema_extra.*")
warnings.filterwarnings("ignore", message=".*Field.*deprecated.*")
# MLflow 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", message=".*Add type hints to the.*")
warnings.filterwarnings("ignore", message=".*Failed to resolve installed pip version.*")
warnings.filterwarnings("ignore", message=".*artifact_path.*deprecated.*")

# CLI 전용 로그 레벨 정의 (INFO=20, WARNING=30 사이)
# 기본: 상세 출력 (DEBUG), -q 옵션 시 CLI_LEVEL 이상만 출력 (요약 모드)
CLI_LEVEL = 25
logging.addLevelName(CLI_LEVEL, "CLI")

# 전역 로거 객체
logger = logging.getLogger(__name__)

# 현재 세션의 로그 파일 경로 저장 (MLflow 업로드용)
_current_log_file: Optional[Path] = None


class TerminalFormatter(logging.Formatter):
    """
    터미널용 포맷터: CLI 진행 상태와 연동하여 들여쓰기 적용.

    MMP_CLI_LINE_ACTIVE: 들여쓰기 활성화 여부
    MMP_CLI_NEEDS_NEWLINE: 줄바꿈 필요 여부 (기본 모드에서 인라인 텍스트 후)
    """

    def format(self, record: logging.LogRecord) -> str:
        import os
        message = record.getMessage()

        # CLI 진행 라인이 활성 상태면 들여쓰기 적용
        if os.environ.get("MMP_CLI_LINE_ACTIVE") == "1":
            # 기본 모드: 인라인 텍스트 후 줄바꿈 필요
            if os.environ.get("MMP_CLI_NEEDS_NEWLINE") == "1":
                return f"\n  {message}"
            # verbose 모드: 이미 줄바꿈됨, 들여쓰기만
            return f"  {message}"

        return message


class FileFormatter(logging.Formatter):
    """파일용 포맷터: 타임스탬프 및 레벨 포함"""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )


def get_current_log_file() -> Optional[Path]:
    """현재 세션의 로그 파일 경로 반환"""
    return _current_log_file


def setup_log_level(level: int) -> None:
    """
    런타임 로그 레벨 변경 (-v, -q 옵션 지원).
    핸들러가 없으면 기본 콘솔 핸들러를 추가하여 즉시 출력 가능하게 함.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 핸들러가 없으면 기본 콘솔 핸들러 추가 (setup_logging 호출 전 로그 출력용)
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(TerminalFormatter())
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(level)

    # 외부 라이브러리 로그 억제 (Popen 등 불필요한 출력 방지)
    logging.getLogger("git").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logging.getLogger("git.util").setLevel(logging.WARNING)


# 표준화된 카테고리별 로깅 함수
# 카테고리: PIPE(파이프라인), FACT(팩토리), DATA(데이터), PREP(전처리),
#          TRAIN(학습), EVAL(평가), MLFLOW(MLflow), SYS(시스템)


def log_pipe(message: str, phase: str = None) -> None:
    """파이프라인 단계 로그"""
    if phase:
        logger.info(f"[PIPE:{phase}] {message}")
    else:
        logger.info(f"[PIPE] {message}")


def log_fact(message: str, component: str = None) -> None:
    """Factory 컴포넌트 생성 로그 (상세 로그, DEBUG 레벨)"""
    if component:
        logger.debug(f"[FACT:{component}] {message}")
    else:
        logger.debug(f"[FACT] {message}")


def log_data(message: str, component: str = None) -> None:
    """데이터 처리 로그"""
    if component:
        logger.info(f"[DATA:{component}] {message}")
    else:
        logger.info(f"[DATA] {message}")


def log_data_debug(message: str, component: str = None) -> None:
    """데이터 처리 상세 로그 (DEBUG)"""
    if component:
        logger.debug(f"[DATA:{component}] {message}")
    else:
        logger.debug(f"[DATA] {message}")


def log_prep(message: str, step: str = None) -> None:
    """전처리 로그"""
    if step:
        logger.info(f"[PREP:{step}] {message}")
    else:
        logger.info(f"[PREP] {message}")


def log_prep_debug(message: str, step: str = None) -> None:
    """전처리 상세 로그 (DEBUG)"""
    if step:
        logger.debug(f"[PREP:{step}] {message}")
    else:
        logger.debug(f"[PREP] {message}")


def log_train(message: str) -> None:
    """학습 로그"""
    logger.info(f"[TRAIN] {message}")


def log_train_debug(message: str) -> None:
    """학습 상세 로그 (DEBUG)"""
    logger.debug(f"[TRAIN] {message}")


def log_eval(message: str) -> None:
    """평가 로그"""
    logger.info(f"[EVAL] {message}")


def log_eval_debug(message: str) -> None:
    """평가 상세 로그 (DEBUG)"""
    logger.debug(f"[EVAL] {message}")


def log_mlflow(message: str) -> None:
    """MLflow 관련 로그"""
    logger.info(f"[MLFLOW] {message}")


def log_sys(message: str) -> None:
    """시스템/환경 로그"""
    logger.info(f"[SYS] {message}")


def log_sys_debug(message: str) -> None:
    """시스템/환경 상세 로그 (DEBUG)"""
    logger.debug(f"[SYS] {message}")


def log_config(message: str) -> None:
    """설정 로드 관련 로그 (Recipe, Config 파일 로드 시 사용)"""
    logger.info(f"[CONFIG] {message}")


def log_cli(message: str) -> None:
    """CLI 단계 로그 (기본 모드에서 터미널에 항상 출력)"""
    logger.log(CLI_LEVEL, message)


def log_component(component: str, message: str) -> None:
    """컴포넌트 생성/초기화 로그 (호환성 유지, FACT 형식 사용, DEBUG 레벨)"""
    logger.debug(f"[FACT:{component}] {message}")


def log_model_op(message: str, details: str = None) -> None:
    """모델 작업 로그 (호환성 유지, TRAIN 형식 사용)"""
    if details:
        logger.info(f"[TRAIN] {message} - {details}")
    else:
        logger.info(f"[TRAIN] {message}")


def log_data_op(message: str, details: str = None) -> None:
    """데이터 작업 로그 (호환성 유지, DATA 형식 사용)"""
    if details:
        logger.info(f"[DATA] {message} - {details}")
    else:
        logger.info(f"[DATA] {message}")


def log_milestone(milestone_type: str, message: str) -> None:
    """마일스톤 로그 (호환성 유지, PIPE 형식 사용)"""
    logger.info(f"[PIPE:{milestone_type}] {message}")


def log_phase(phase_name: str) -> None:
    """파이프라인 페이즈 로그 (호환성 유지, PIPE 형식 사용)"""
    logger.info(f"[PIPE] {phase_name}")


def log_pipeline_start(pipeline_name: str, details: str = None) -> None:
    """파이프라인 시작 로그 (호환성 유지, PIPE 형식 사용)"""
    if details:
        logger.info(f"[PIPE] {pipeline_name} 시작 - {details}")
    else:
        logger.info(f"[PIPE] {pipeline_name} 시작")


def setup_logging(settings: "Settings", verbose: bool = None):
    """
    주입된 설정(settings) 객체를 기반으로 전역 로거를 설정합니다.
    모든 환경에서 파일 로깅을 수행하며, 실험별로 로그 파일을 분리합니다.

    Args:
        settings: Settings 객체
        verbose: True면 터미널에도 상세 로그 출력 (-v 옵션)
                 None이면 main_commands.py에서 설정한 레벨 자동 감지
    """
    global _current_log_file

    root_logger = logging.getLogger()

    # verbose 자동 감지 (main_commands.py의 -v 옵션이 먼저 실행됨)
    if verbose is None:
        verbose = root_logger.level == logging.DEBUG

    # 핸들러 중복 등록 방지
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    # Config에서 로깅 설정 가져오기
    logging_config = getattr(getattr(settings, "config", None), "logging", None)
    env_obj = getattr(getattr(settings, "config", None), "environment", None)
    env_name = getattr(env_obj, "name", None) or "local"

    # 로깅 설정 기본값
    if logging_config:
        base_path = getattr(logging_config, "base_path", "./logs")
        retention_days = getattr(logging_config, "retention_days", 30)
    else:
        base_path = "./logs"
        retention_days = 30

    # 파일 로그 레벨: DEBUG 레벨 이상 모두 기록 (상세 로그 보존)
    file_log_level = logging.DEBUG

    # 터미널 로그 레벨: 기본은 DEBUG, -q 옵션 시 CLI_LEVEL(25)로 요약
    console_log_level = CLI_LEVEL if not verbose else logging.DEBUG

    # 루트 로거는 가장 낮은 레벨로 설정 (모든 로그 캡처)
    root_logger.setLevel(logging.DEBUG)

    # 로그 디렉토리 생성
    log_dir = Path(base_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 실험별 로그 파일명 생성
    recipe_name = _extract_recipe_name(settings)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{env_name}_{recipe_name}_{timestamp}.log"
    log_file = log_dir / log_filename
    _current_log_file = log_file

    # 파일 핸들러: DEBUG 이상 모든 로그 기록
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_log_level)

    # 콘솔 핸들러: 기본은 CLI_LEVEL(25) 이상만, -v 옵션 시 DEBUG까지
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)

    # 핸들러별 포맷터 설정
    file_handler.setFormatter(FileFormatter())
    console_handler.setFormatter(TerminalFormatter())

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 외부 라이브러리 로그 억제
    # Git 관련 로그 (Popen 출력 방지)
    logging.getLogger("git").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logging.getLogger("git.util").setLevel(logging.WARNING)

    # MLflow 내부 경고 억제 (verbose 여부와 무관하게 ERROR 이상만)
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("mlflow.tracking").setLevel(logging.ERROR)
    logging.getLogger("mlflow.utils").setLevel(logging.ERROR)
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
    logging.getLogger("mlflow.models").setLevel(logging.ERROR)
    logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
    logging.getLogger("mlflow.models.utils").setLevel(logging.ERROR)
    logging.getLogger("mlflow.pyfunc").setLevel(logging.ERROR)

    # Optuna 내부 로그 억제 (우리 형식으로 대체)
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("optuna.trial").setLevel(logging.WARNING)

    # Google Cloud / BigQuery 관련 로그 억제
    logging.getLogger("google.cloud.bigquery").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("google.auth.transport").setLevel(logging.WARNING)
    logging.getLogger("google.auth.transport.requests").setLevel(logging.WARNING)

    # HTTP 클라이언트 로그 억제
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # 오래된 로그 파일 정리
    _cleanup_old_logs(log_dir, retention_days)


def _extract_recipe_name(settings: "Settings") -> str:
    """Settings에서 recipe 이름 추출"""
    try:
        recipe = getattr(settings, "recipe", None)
        if recipe:
            name = getattr(recipe, "name", None)
            if name:
                return name
    except Exception:
        pass
    return "unknown"


def _cleanup_old_logs(log_dir: Path, retention_days: int) -> None:
    """지정된 보관 일수보다 오래된 로그 파일 삭제"""
    try:
        import os
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=retention_days)

        for log_file in log_dir.glob("*.log"):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            if file_mtime < cutoff_time:
                log_file.unlink()
                logger.debug(f"오래된 로그 파일 삭제: {log_file}")
    except Exception as e:
        logger.warning(f"로그 파일 정리 중 오류: {e}")
