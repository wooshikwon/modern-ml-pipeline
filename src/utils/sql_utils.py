"""
SQL 유틸리티 모듈

Jinja2 템플릿을 사용하여 SQL 파일을 로드하고 파라미터를 주입하는 기능을 제공합니다.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Union, Optional

import jinja2

from src.utils.logger import logger


def render_sql(file_path: str, params: Union[Dict[str, Any], SimpleNamespace, None] = None, project_root: Optional[Path] = None) -> str:
    """
    SQL 파일을 로드하고 Jinja2 템플릿을 사용하여 파라미터를 주입합니다.

    Args:
        file_path (str): .sql 파일의 경로 (절대경로 또는 상대경로)
        params (Union[Dict, SimpleNamespace, None], optional):
            SQL 쿼리에 주입할 파라미터. 딕셔너리 또는 SimpleNamespace 객체.
            Defaults to None.
        project_root (Optional[Path], optional):
            프로젝트 루트 디렉토리. 지정하지 않으면 파일 위치 기준으로 자동 감지.
            Defaults to None.
    
    Returns:
        str: 파라미터가 렌더링된 최종 SQL 쿼리 문자열
    """
    if params is None:
        params = {}

    try:
        # 프로젝트 루트 디렉토리 찾기
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent
        
        # Jinja2 Environment 설정
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(project_root)),
            # SQL에서 유용한 추가 설정
            trim_blocks=True,           # 블록 태그 후 첫 번째 개행 제거
            lstrip_blocks=True,         # 블록 태그 앞 공백 제거
            keep_trailing_newline=True  # 파일 끝 개행 유지
        )

        template = env.get_template(file_path)

        # 파라미터 타입에 따른 처리
        if isinstance(params, SimpleNamespace):
            params_dict = vars(params)
            logger.debug(f"Converting SimpleNamespace to dict: {list(params_dict.keys())}")
        elif isinstance(params, dict):
            params_dict = params
            logger.debug(f"Using dict parameters: {list(params_dict.keys())}")
        else:
            logger.warning(f"Unexpected params type: {type(params)}. Converting to dict.")
            params_dict = dict(params) if hasattr(params, '__iter__') else {}

        # 템플릿 렌더링
        rendered_sql = template.render(params_dict)

        logger.info(f"SQL template rendered successfully. Query length: {len(rendered_sql)} characters")
        logger.debug(f"Rendered SQL preview: {rendered_sql[:200]}...")

        return rendered_sql

    except FileNotFoundError:
        logger.error(f"SQL file not found at path: {file_path}")
        raise
    except jinja2.TemplateNotFound as e:
        logger.error(f"SQL template not found: {e}")
        raise
    except jinja2.TemplateError as e:
        logger.error(f"Template rendering error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while rendering SQL template: {e}")
        raise


def validate_sql_file(file_path: str) -> bool:
    """
    SQL 파일의 유효성을 검사합니다.

    Args:
        file_path (str): 검사할 SQL 파일의 경로

    Returns:
        bool: 파일이 유효하면 True, 그렇지 않으면 False

    Examples:
        >>> if validate_sql_file("src/sql/query.sql"):
        ...     sql = render_sql("src/sql/query.sql")
    """
    try:
        sql_path = Path(file_path)

        # 절대 경로가 아닌 경우 프로젝트 루트 기준으로 변환
        if not sql_path.is_absolute():
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent
            sql_path = project_root / sql_path

        # 파일 존재 및 확장자 확인
        if not sql_path.exists():
            logger.warning(f"SQL file does not exist: {sql_path}")
            return False

        if sql_path.suffix.lower() != '.sql':
            logger.warning(f"File is not a SQL file: {sql_path}")
            return False

        # 파일이 읽을 수 있는지 확인
        try:
            sql_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Cannot read SQL file {sql_path}: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating SQL file {file_path}: {e}")
        return False


def get_sql_file_info(file_path: str) -> Dict[str, Any]:
    """
    SQL 파일의 정보를 반환합니다.

    Args:
        file_path (str): SQL 파일의 경로

    Returns:
        Dict[str, Any]: 파일 정보 딕셔너리
            - exists: 파일 존재 여부
            - size: 파일 크기 (바이트)
            - modified: 마지막 수정 시간
            - absolute_path: 절대 경로

    Examples:
        >>> info = get_sql_file_info("src/sql/query.sql")
        >>> print(f"File size: {info['size']} bytes")
    """
    try:
        sql_path = Path(file_path)

        # 절대 경로가 아닌 경우 프로젝트 루트 기준으로 변환
        if not sql_path.is_absolute():
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent
            sql_path = project_root / sql_path

        if sql_path.exists():
            stat = sql_path.stat()
            return {
                'exists': True,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'absolute_path': str(sql_path.absolute())
            }
        else:
            return {
                'exists': False,
                'size': 0,
                'modified': None,
                'absolute_path': str(sql_path.absolute())
            }

    except Exception as e:
        logger.error(f"Error getting SQL file info for {file_path}: {e}")
        return {
            'exists': False,
            'size': 0,
            'modified': None,
            'absolute_path': file_path,
            'error': str(e)
        }
