"""Modern ML Pipeline top-level package.

This file ensures `src` is treated as a regular Python package (not a namespace),
which enables tooling like import-linter to analyze intra-package dependencies.
"""

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