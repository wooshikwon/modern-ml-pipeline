"""
Project Templates Package

This package contains template files for the modern-ml-pipeline init command.
Template files include configuration files, recipe examples, and project scaffolding.
"""

from pathlib import Path

# Template directory path for programmatic access
TEMPLATES_DIR = Path(__file__).parent

__all__ = ["TEMPLATES_DIR"]