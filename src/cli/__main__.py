"""
Modern ML Pipeline CLI - Entry Point

python -m src.cli [command] [options] 형태로 실행 가능
"""

# 모든 import 전에 경고 억제 (Pydantic, MLflow 등)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.cli.main_commands import app

if __name__ == "__main__":
    app()
