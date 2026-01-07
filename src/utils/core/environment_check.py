"""
Development Environment Compatibility Check
   - Architecture Excellence

이 모듈은 개발환경의 호환성을 검증하여 예상치 못한 오류를 사전에 방지합니다.

주요 검증 항목:
- Python 버전 호환성 (3.11.x 권장, 3.12 미지원)
- 필수 패키지 호환성
- 환경 변수 설정 확인
- 기본 디렉토리 구조 검증

실행 환경을 확인하고 캡처하는 함수 모음.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from src.utils.core.logger import logger


class EnvironmentChecker:
    """개발환경 호환성 검증 클래스"""

    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def check_python_version(self) -> bool:
        """Python 버전 호환성 검증"""
        current_version = sys.version_info

        # Python 3.11.x 권장 (causalml 호환성 고려)
        if current_version.major != 3:
            self.errors.append(
                f"Python 3.11.x가 필요합니다. 현재: Python {current_version.major}.{current_version.minor}"
            )
            return False

        if current_version.minor < 11:
            self.errors.append(
                f"Python 3.11 이상이 필요합니다. 현재: Python {current_version.major}.{current_version.minor}"
            )
            return False

        if current_version.minor == 12:
            self.warnings.append(
                "Python 3.12는 causalml과 호환성 문제가 있을 수 있습니다. Python 3.11.x 사용을 권장합니다."
            )

        if current_version.minor > 12:
            self.warnings.append(
                f"Python {current_version.major}.{current_version.minor}는 테스트되지 않았습니다. Python 3.11.x 사용을 권장합니다."
            )

        logger.info(
            f"Python 버전 확인: {current_version.major}.{current_version.minor}.{current_version.micro}"
        )
        return True

    def check_required_packages(self) -> bool:
        """필수 패키지 호환성 검증"""
        required_packages = [
            "pandas",
            "numpy",
            "scikit-learn",
            "mlflow",
            "pydantic",
            "fastapi",
            "uvicorn",
            "typer",
            "pyyaml",
            "python-dotenv",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.errors.append(f"필수 패키지가 설치되지 않았습니다: {missing_packages}")
            return False

        logger.info("필수 패키지 확인 완료")
        return True

    def check_optional_packages(self) -> bool:
        """선택적 패키지 호환성 검증"""
        optional_packages = {
            "redis": "실시간 Feature Store 지원",
            "causalml": "인과추론 모델 지원",
            "optuna": "하이퍼파라미터 자동 최적화",
            "xgboost": "XGBoost 모델 지원",
            "lightgbm": "LightGBM 모델 지원",
        }

        missing_optional = []
        for package, description in optional_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(f"{package} ({description})")

        if missing_optional:
            self.warnings.append(f"선택적 패키지가 설치되지 않았습니다: {missing_optional}")

        logger.info("선택적 패키지 확인 완료")
        return True

    def check_directory_structure(self) -> bool:
        """기본 디렉토리 구조 검증"""
        base_dir = Path(__file__).resolve().parent.parent.parent.parent

        required_dirs = [
            "config",
            "recipes",
            "src",
            "data",
            "serving",
            "tests",
        ]

        missing_dirs = []
        for dir_name in required_dirs:
            if not (base_dir / dir_name).exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            self.errors.append(f"필수 디렉토리가 없습니다: {missing_dirs}")
            return False

        logger.info("디렉토리 구조 확인 완료")
        return True

    def check_environment_variables(self) -> bool:
        """환경변수 설정 확인"""
        env_name = os.getenv("ENV_NAME", "local")

        if env_name == "local":
            # LOCAL 환경에서는 환경변수 검증 생략
            logger.info("LOCAL 환경: 환경변수 검증 생략")
            return True

        # DEV/PROD 환경에서는 추가 검증
        if env_name in ["dev", "prod"]:
            critical_vars = []

            # PostgreSQL 연결 필요 시
            if not os.getenv("POSTGRES_PASSWORD"):
                critical_vars.append("POSTGRES_PASSWORD")

            if critical_vars:
                self.warnings.append(f"중요한 환경변수가 설정되지 않았습니다: {critical_vars}")

        logger.info(f"환경변수 확인 완료 (ENV_NAME: {env_name})")
        return True

    def check_system_compatibility(self) -> bool:
        """시스템 호환성 검증"""
        system_info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        }

        # macOS Apple Silicon 호환성 체크
        if system_info["system"] == "Darwin" and system_info["machine"] == "arm64":
            self.warnings.append(
                "Apple Silicon Mac 감지. 일부 패키지는 Rosetta 2가 필요할 수 있습니다."
            )

        logger.info(f"시스템 호환성 확인: {system_info['system']} {system_info['machine']}")
        return True

    def run_full_check(self) -> Tuple[bool, List[str], List[str]]:
        """전체 환경 검증 실행"""
        logger.info("개발환경 호환성 검증 시작...")

        checks = [
            self.check_python_version,
            self.check_required_packages,
            self.check_optional_packages,
            self.check_directory_structure,
            self.check_environment_variables,
            self.check_system_compatibility,
        ]

        success = True
        for check in checks:
            if not check():
                success = False

        # 결과 출력
        if self.errors:
            logger.error("환경 검증 실패:")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning("환경 검증 경고:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        if success and not self.errors:
            logger.info("개발환경 호환성 검증 완료")

        return success, self.errors, self.warnings


# 편의 함수
def check_environment() -> bool:
    """환경 검증 편의 함수"""
    checker = EnvironmentChecker()
    success, errors, warnings = checker.run_full_check()
    return success


# 모듈 import 시 자동 검증 (개발 환경에서만)
if __name__ == "__main__":
    check_environment()


def get_pip_requirements() -> List[str]:
    """
    'uv pip freeze'를 사용하여 현재 환경의 의존성을 캡처합니다.
    ['pandas==2.0.0', 'scikit-learn==1.3.0']와 같은 문자열 리스트를 반환합니다.
    """
    try:
        result = subprocess.run(
            ["uv", "pip", "freeze"], capture_output=True, text=True, check=True, encoding="utf-8"
        )
        # 빈 줄은 제거하되 주석/특수 라인은 그대로 유지
        raw_lines = result.stdout.splitlines()
        requirements = [line for line in raw_lines if line.strip()]
        logger.info(f"현재 환경에서 {len(requirements)}개의 패키지 의존성을 캡처했습니다.")
        return requirements
    except FileNotFoundError:
        logger.warning(
            "'uv' 명령어를 찾을 수 없어 pip 의존성을 캡처할 수 없습니다. 모델 아티팩트에 포함되지 않습니다."
        )
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"pip 의존성 캡처 중 오류 발생: {e.stderr}")
        return []
