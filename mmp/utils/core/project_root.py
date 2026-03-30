"""
프로젝트 루트 자동 탐지

CWD에서 상위로 올라가며 mmp 프로젝트 루트를 찾는다.
cargo가 Cargo.toml을, git이 .git/을 찾는 것과 동일한 패턴.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MMP_MARKERS = ("configs", "recipes")


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """
    CWD에서 상위로 올라가며 mmp 프로젝트 루트를 탐지한다.

    탐지 기준 (우선순위):
    1. configs/ + recipes/ 디렉토리가 모두 존재하는 경로
    2. 홈 디렉토리나 파일시스템 루트에 도달하면 중단

    Args:
        start: 탐색 시작 경로 (기본: cwd)

    Returns:
        프로젝트 루트 경로. 찾지 못하면 None.
    """
    current = (start or Path.cwd()).resolve()
    home = Path.home().resolve()

    while True:
        if all((current / marker).is_dir() for marker in _MMP_MARKERS):
            if current != Path.cwd().resolve():
                logger.debug(f"프로젝트 루트 감지: {current} (cwd: {Path.cwd()})")
            return current

        parent = current.parent
        if parent == current or current == home:
            return None
        current = parent


def resolve_project_path(relative_path: str, start: Optional[Path] = None) -> Path:
    """
    상대 경로를 프로젝트 루트 기준으로 해소한다.

    1. 절대 경로이면 그대로 반환
    2. CWD 기준으로 존재하면 그대로 반환
    3. 프로젝트 루트 기준으로 존재하면 루트 기준 경로 반환
    4. 모두 실패하면 원본 경로 반환 (호출자가 에러 처리)

    Args:
        relative_path: 해소할 경로
        start: 프로젝트 루트 탐색 시작점

    Returns:
        해소된 Path
    """
    path = Path(relative_path)

    if path.is_absolute():
        return path

    if path.exists():
        return path

    root = find_project_root(start)
    if root is not None:
        rooted = root / path
        if rooted.exists():
            return rooted

    return path
