"""환경변수 치환 모듈 - ${VAR:default} 패턴 지원

YAML 설정 파일의 환경변수 참조를 실제 값으로 치환한다.
재귀적으로 dict, list, str을 순회하며 ${VAR} 또는 ${VAR:default} 패턴을 처리한다.

사용 예시:
    >>> resolve_env_variables({"host": "${DB_HOST:localhost}", "port": "${DB_PORT:5432}"})
    {'host': 'localhost', 'port': 5432}
"""

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ${VAR} 또는 ${VAR:default} 패턴
_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def resolve_env_variables(data: Any) -> Any:
    """환경변수 치환 - ${VAR:default} 패턴 지원

    전체 문자열이 단일 환경변수 참조인 경우 타입 변환을 시도한다
    (bool, int, float). 부분 치환인 경우 문자열로만 치환한다.

    Args:
        data: 치환 대상. str, dict, list 또는 기타 타입.

    Returns:
        환경변수가 치환된 결과. 원본과 동일한 구조를 유지한다.
    """
    if isinstance(data, str):
        return _resolve_string(data)
    elif isinstance(data, dict):
        return {k: resolve_env_variables(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_env_variables(item) for item in data]
    return data


def _resolve_string(data: str) -> Any:
    """문자열 내 환경변수 치환

    전체 문자열이 단일 ${VAR} 또는 ${VAR:default}이면 타입 변환을 시도한다.
    부분 치환(예: "host:${PORT}")이면 문자열로만 치환한다.
    """
    full_match = _ENV_PATTERN.fullmatch(data)

    if full_match:
        # 전체가 환경변수인 경우 - 타입 변환 시도
        return _resolve_full_match(full_match.group(1), data)
    else:
        # 부분적으로 환경변수가 포함된 경우 - 문자열로만 치환
        return _ENV_PATTERN.sub(_partial_replacer, data)


def _resolve_full_match(expr: str, original: str) -> Any:
    """전체 매치 시 환경변수 치환 + 타입 변환"""
    if ":" in expr:
        var_name, default_value = expr.split(":", 1)
        var_name = var_name.strip()
        default_value = default_value.strip()
        result = os.environ.get(var_name, default_value)
    else:
        var_name = expr.strip()
        result = os.environ.get(var_name)
        if result is None:
            logger.warning(
                "환경변수 '%s'가 설정되지 않았고 기본값도 없습니다. "
                "원본 문자열 '${%s}'이 그대로 사용됩니다. "
                "Pydantic 파싱 시 타입 에러가 발생할 수 있습니다.",
                var_name,
                var_name,
            )
            result = original  # 기존 동작 유지: 원본 반환

    if isinstance(result, str):
        return _coerce_type(result)
    return result


def _coerce_type(value: str) -> Any:
    """문자열을 적절한 Python 타입으로 변환 시도

    변환 우선순위: 빈 문자열 -> bool -> int -> float -> str
    """
    if value == "":
        return ""

    # Boolean 변환
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # 숫자 변환
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except (ValueError, AttributeError):
        return value


def _partial_replacer(match: re.Match) -> str:
    """부분 치환용 replacer - 항상 문자열 반환"""
    expr = match.group(1)

    if ":" in expr:
        var_name, default_value = expr.split(":", 1)
        var_name = var_name.strip()
        default_value = default_value.strip()
        return str(os.environ.get(var_name, default_value))
    else:
        var_name = expr.strip()
        result = os.environ.get(var_name)
        if result is None:
            logger.warning(
                "환경변수 '%s'가 설정되지 않았고 기본값도 없습니다. "
                "원본 문자열 '${%s}'이 그대로 사용됩니다.",
                var_name,
                var_name,
            )
            return match.group(0)
        return str(result)
