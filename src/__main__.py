"""
Modern ML Pipeline - CLI Entry Point

This module serves as the main entry point when the package is run as a module:
    python -m src [command] [options]
    
or when installed as a package:
    ml-pipeline [command] [options]
"""

import sys
from src.cli.main_commands import app


def main():
    """
    Main entry point for the Modern ML Pipeline CLI tool.
    
    This function is called when:
    1. The package is run as a module: python -m src
    2. The console script is invoked: ml-pipeline (after installation)
    """
    # Delegate to the typer application
    app()


if __name__ == "__main__":
    # Direct execution: python src/__main__.py
    main()