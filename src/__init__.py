"""Modern ML Pipeline top-level package.

This file ensures `src` is treated as a regular Python package (not a namespace),
which enables tooling like import-linter to analyze intra-package dependencies.
"""

import warnings

# 전역 라이브러리 경고 차단 (최상단 init)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
warnings.filterwarnings("ignore", message=".*json_schema_extra.*")
warnings.filterwarnings("ignore", message=".*Field.*deprecated.*")

__all__ = [
    "components",
    "engine",
    "interface",
    "models",
    "pipelines",
    "serving",
    "settings",
    "utils",
]
