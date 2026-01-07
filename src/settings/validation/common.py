"""검증 시스템 공통 타입"""

from typing import List, NamedTuple


class ValidationResult(NamedTuple):
    is_valid: bool
    error_message: str = ""
    warnings: List[str] = []
