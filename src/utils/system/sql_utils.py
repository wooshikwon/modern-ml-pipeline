"""
SQL 유틸리티 모듈
- Jinja2 템플릿을 사용하여 SQL 파일을 렌더링합니다.
- sqlparse를 사용하여 SQL 쿼리를 파싱합니다.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional, List

import jinja2
import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import DML

from src.utils.system.logger import logger


def render_sql(
    file_path: str, params: Union[Dict[str, Any], None] = None
) -> str:
    """
    SQL 파일을 로드하고 Jinja2 템플릿을 사용하여 파라미터를 주입합니다.
    """
    params = params or {}
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(project_root)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        template = env.get_template(file_path)
        rendered_sql = template.render(params)
        logger.info(f"SQL 템플릿 렌더링 성공: {file_path}")
        return rendered_sql
    except jinja2.TemplateError as e:
        logger.error(f"SQL 템플릿 렌더링 오류: {e}")
        raise
    except Exception as e:
        logger.error(f"SQL 템플릿 처리 중 예상치 못한 오류: {e}")
        raise


def get_selected_columns(sql_query: str) -> List[str]:
    """
    주어진 SQL 쿼리 문자열을 파싱하여 SELECT 절에 있는 컬럼명들을 추출합니다.
    'SELECT a, b, c FROM ...' -> ['a', 'b', 'c']
    'SELECT t1.a, t2.b as my_b, c FROM ...' -> ['a', 'my_b', 'c']
    """
    columns: List[str] = []
    parsed = sqlparse.parse(sql_query)[0]

    select_seen = False
    for token in parsed.tokens:
        if token.ttype is DML and token.normalized == "SELECT":
            select_seen = True
            continue
        if not select_seen:
            continue

        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                alias = identifier.get_alias()
                columns.append(alias if alias else identifier.get_real_name())
        elif isinstance(token, Identifier):
            alias = token.get_alias()
            columns.append(alias if alias else token.get_real_name())
        
        if columns:
            break
            
    logger.info(f"SQL에서 {len(columns)}개 컬럼 추출: {columns}")
    return columns


def parse_select_columns(sql_snapshot: str) -> List[str]:
    """
    🆕 Blueprint v17.0: loader_sql_snapshot에서 API 입력 스키마용 컬럼 추출
    
    SELECT 절에서 컬럼을 추출하되, event_timestamp 등 시간 컬럼은 제외
    주로 PK: user_id, product_id, session_id 등을 API 입력으로 사용
    """
    try:
        columns = get_selected_columns(sql_snapshot)
        
        # API 입력에서 제외할 컬럼들 (시간 관련 컬럼)
        excluded_columns = {"event_timestamp", "timestamp", "created_at", "updated_at"}
        
        # PK 용도의 컬럼만 필터링
        api_columns = [col for col in columns if col.lower() not in excluded_columns]
        
        logger.info(f"API 스키마용 컬럼 추출 완료: {api_columns}")
        return api_columns
        
    except Exception as e:
        logger.warning(f"SQL 파싱 실패, 빈 목록 반환: {e}")
        return []


def parse_feature_columns(augmenter_sql_snapshot: str) -> tuple[List[str], str]:
    """
    🆕 Blueprint v17.0: augmenter_sql_snapshot에서 피처 컬럼과 JOIN 키 추출
    
    Feature Store 조회를 위한 컬럼 목록과 JOIN 키를 분석
    """
    try:
        columns = get_selected_columns(augmenter_sql_snapshot)
        
        # 일반적인 JOIN 키 패턴들
        join_key_patterns = ["user_id", "member_id", "customer_id", "product_id", "session_id"]
        
        join_key = ""
        for pattern in join_key_patterns:
            if pattern in columns:
                join_key = pattern
                break
        
        if not join_key and columns:
            join_key = columns[0]  # 첫 번째 컬럼을 기본 JOIN 키로 사용
        
        logger.info(f"피처 컬럼 분석 완료: {len(columns)}개, JOIN 키: {join_key}")
        return columns, join_key
        
    except Exception as e:
        logger.warning(f"Augmenter SQL 파싱 실패: {e}")
        return [], ""