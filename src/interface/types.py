from typing import Protocol, Optional

class MLTaskProtocol(Protocol):
    task_type: str
    target_column: Optional[str]
    treatment_column: Optional[str] 