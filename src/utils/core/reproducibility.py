from __future__ import annotations

# 모듈 레벨에서 의존성 참조를 확보하여 테스트에서 모킹 가능하도록 함
try:  # torch는 선택적 의존성
    import torch as torch  # noqa: F401  (모듈 레벨 심볼 유지)
except Exception:  # pragma: no cover - 미설치 환경 호환
    torch = None  # type: ignore

try:
    from src.utils.core.logger import log_sys, logger  # noqa: F401
except Exception:  # pragma: no cover
    logger = None  # type: ignore
    log_sys = None  # type: ignore


def set_global_seeds(seed: int = 42) -> None:
    """전역 시드 설정을 통해 재현성을 높입니다.
    - random, numpy, (가능 시) torch 에 시드 적용
    """
    # OS 환경 변수 설정
    try:
        import os

        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass

    # Python random
    try:
        import random

        random.seed(seed)
    except Exception:
        pass

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    # PyTorch (있을 때만)
    try:
        if torch is not None:
            try:
                torch.manual_seed(seed)
            except Exception:
                pass
            try:
                cuda_available = False
                if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", None)):
                    cuda_available = bool(torch.cuda.is_available())
                if cuda_available and hasattr(torch.cuda, "manual_seed_all"):
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            try:
                if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        # torch 접근/모킹 시 예외는 안전히 무시
        pass

    # 로깅 (선택적)
    try:
        if log_sys is not None:
            log_sys(f"전역 시드 설정: {seed}")
        elif logger is not None:
            logger.info(f"[SYS] 전역 시드 설정: {seed}")
    except Exception:
        pass
