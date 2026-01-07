"""
SQL 유틸리티 모듈
- sqlparse를 사용하여 SQL 쿼리를 파싱합니다.
"""

from typing import List

import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import DML

from src.utils.core.logger import logger


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
     loader_sql_snapshot에서 API 입력 스키마용 컬럼 추출

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


def parse_feature_columns(fetcher_sql_snapshot: str) -> tuple[List[str], str]:
    """
     fetcher_sql_snapshot에서 피처 컬럼과 JOIN 키 추출

    Feature Store 조회를 위한 컬럼 목록과 JOIN 키를 분석
    """
    try:
        columns = get_selected_columns(fetcher_sql_snapshot)

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
        logger.warning(f"fetcher SQL 파싱 실패: {e}")
        return [], ""
