"""
패키지 버전 호환성 체크 모듈

학습 시 사용된 패키지 버전과 현재 환경의 버전을 비교하여
호환성 문제를 사전에 경고합니다.
"""

import os
import re
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import mlflow

from mmp.utils.core.logger import log_warn, logger

# artifact 다운로드 타임아웃 (초)
ARTIFACT_DOWNLOAD_TIMEOUT = 3

# 직렬화에 영향을 주는 핵심 패키지
CRITICAL_PACKAGES = [
    "numpy",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "pandas",
]


def check_package_versions(run_id: str) -> List[str]:
    """
    학습 시 패키지 버전과 현재 환경 비교.

    MLflow artifact에서 requirements.txt를 읽어 핵심 패키지 버전을 비교합니다.
    버전 불일치 시 경고 메시지를 반환하지만, 실행은 계속됩니다.

    Args:
        run_id: MLflow Run ID

    Returns:
        경고 메시지 리스트 (비어있으면 호환)
    """
    # 환경변수로 체크 비활성화 가능
    if os.environ.get("MMP_SKIP_VERSION_CHECK", "0") == "1":
        logger.debug("MMP_SKIP_VERSION_CHECK=1로 버전 체크 비활성화됨")
        return []

    try:
        # MLflow artifact에서 학습 시 버전 조회
        stored_versions = _get_stored_versions(run_id)
        if not stored_versions:
            logger.debug(f"버전 정보를 찾을 수 없음: run_id={run_id[:8]}...")
            return []

        # 현재 환경 버전 수집
        current_versions = _get_current_versions()

        # 비교 및 경고 생성
        warnings = []
        for pkg in CRITICAL_PACKAGES:
            stored = stored_versions.get(pkg)
            current = current_versions.get(pkg)

            if stored and current:
                mismatch_type = _check_version_mismatch(stored, current)
                if mismatch_type == "major":
                    warnings.append(
                        f"{pkg}: 학습={stored}, 현재={current} "
                        f"(메이저 버전 불일치 - 모델 로드 실패 가능)"
                    )
                elif mismatch_type == "minor":
                    # 마이너 버전 불일치는 INFO 레벨로 로그만
                    logger.info(f"[VERSION] {pkg}: 학습={stored}, 현재={current} (마이너 버전 불일치)")

        return warnings

    except Exception as e:
        # 버전 체크 실패해도 실행은 계속
        logger.debug(f"버전 체크 실패 (무시하고 진행): {e}")
        return []


def log_version_warnings(run_id: str) -> None:
    """
    버전 경고를 로그로 출력하는 편의 함수.

    Args:
        run_id: MLflow Run ID
    """
    warnings = check_package_versions(run_id)
    for warning in warnings:
        log_warn(warning, "VERSION")


@lru_cache(maxsize=32)
def _get_stored_versions(run_id: str) -> Dict[str, str]:
    """
    MLflow artifact에서 학습 시 패키지 버전 조회.

    캐싱되어 동일 run_id에 대해 한 번만 다운로드합니다.
    타임아웃 적용으로 네트워크 지연 시에도 무한 대기 방지.
    daemon 스레드 사용으로 Python 종료 시 블로킹 방지.

    Args:
        run_id: MLflow Run ID

    Returns:
        패키지명 -> 버전 딕셔너리
    """
    result_holder: Dict[str, Dict[str, str]] = {}

    def download() -> None:
        try:
            req_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="model/requirements.txt",
            )
            result_holder["result"] = _parse_requirements_file(req_path)
        except Exception as e:
            logger.debug(f"requirements.txt 다운로드 실패: {e}")

    # daemon=True: Python 종료 시 스레드가 자동 종료되어 블로킹 방지
    thread = threading.Thread(target=download, daemon=True)
    thread.start()
    thread.join(timeout=ARTIFACT_DOWNLOAD_TIMEOUT)

    if thread.is_alive():
        logger.debug(
            f"requirements.txt 다운로드 타임아웃 ({ARTIFACT_DOWNLOAD_TIMEOUT}초)"
        )
        return {}

    return result_holder.get("result", {})


def _get_current_versions() -> Dict[str, str]:
    """
    현재 환경의 핵심 패키지 버전 수집.

    Returns:
        패키지명 -> 버전 딕셔너리
    """
    versions = {}

    for pkg in CRITICAL_PACKAGES:
        try:
            if pkg == "scikit-learn":
                import sklearn

                versions[pkg] = sklearn.__version__
            elif pkg == "numpy":
                import numpy

                versions[pkg] = numpy.__version__
            elif pkg == "pandas":
                import pandas

                versions[pkg] = pandas.__version__
            elif pkg == "xgboost":
                import xgboost

                versions[pkg] = xgboost.__version__
            elif pkg == "lightgbm":
                import lightgbm

                versions[pkg] = lightgbm.__version__
            elif pkg == "catboost":
                import catboost

                versions[pkg] = catboost.__version__
            elif pkg == "torch":
                import torch

                versions[pkg] = torch.__version__
        except ImportError:
            # 패키지가 설치되지 않은 경우 무시
            pass

    return versions


def _parse_requirements_file(filepath: str) -> Dict[str, str]:
    """
    requirements.txt 파일 파싱.

    Args:
        filepath: requirements.txt 파일 경로

    Returns:
        패키지명 -> 버전 딕셔너리
    """
    versions = {}

    # 패키지==버전 또는 패키지>=버전 형식 파싱
    pattern = re.compile(r"^([a-zA-Z0-9_-]+)[=<>!~]+(.+)$")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = pattern.match(line)
            if match:
                pkg_name = match.group(1).lower().replace("_", "-")
                version = match.group(2).strip()

                # 핵심 패키지만 저장
                normalized_name = pkg_name.replace("-", "_")
                for critical_pkg in CRITICAL_PACKAGES:
                    if normalized_name == critical_pkg.replace("-", "_"):
                        versions[critical_pkg] = version
                        break

    return versions


def _check_version_mismatch(stored: str, current: str) -> Optional[str]:
    """
    버전 불일치 유형 확인.

    Args:
        stored: 저장된 버전
        current: 현재 버전

    Returns:
        "major", "minor", None (호환)
    """
    stored_parts = _parse_version(stored)
    current_parts = _parse_version(current)

    if not stored_parts or not current_parts:
        return None

    # 메이저 버전 비교
    if stored_parts[0] != current_parts[0]:
        return "major"

    # 마이너 버전 비교
    if len(stored_parts) > 1 and len(current_parts) > 1:
        if stored_parts[1] != current_parts[1]:
            return "minor"

    return None


def _parse_version(version: str) -> Optional[Tuple[int, ...]]:
    """
    버전 문자열 파싱.

    Args:
        version: "1.26.4" 형식의 버전 문자열

    Returns:
        (1, 26, 4) 형식의 튜플, 파싱 실패 시 None
    """
    try:
        # 버전에서 숫자 부분만 추출 (예: "2.0.0rc1" -> "2.0.0")
        version_clean = re.match(r"^(\d+(?:\.\d+)*)", version)
        if not version_clean:
            return None

        parts = version_clean.group(1).split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return None
