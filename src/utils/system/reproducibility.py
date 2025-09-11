from __future__ import annotations

def set_global_seeds(seed: int = 42) -> None:
    """전역 시드 설정을 통해 재현성을 높입니다.
    - random, numpy, (가능 시) torch, sklearn에 시드 적용
    """
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass

    try:
        import random
        random.seed(seed)
    except Exception:
        pass

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 결정적 연산(가능한 경우) 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # sklearn 전역 난수 시드에 직접 접근은 없음. 개별 모델에 전달되므로 여기서는 로깅만.
    try:
        from src.utils.system.logger import logger
        logger.info(f"Global seeds set to {seed}")
    except Exception:
        pass 