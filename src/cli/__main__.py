"""
Modern ML Pipeline CLI - Main Entry Point

This module allows the CLI to be run as a Python module:
    python -m src.cli
    
It imports and executes the main Typer app from main_commands.py

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from src.cli.main_commands import app


def main() -> None:
    """
    Main entry point for the CLI when run as a module.
    
    This function is called when the package is executed with:
        python -m src.cli
    
    It delegates to the Typer app for command parsing and execution.
    """
    app()


if __name__ == "__main__":
    main()