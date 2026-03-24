"""검증 시스템 공통 타입"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
