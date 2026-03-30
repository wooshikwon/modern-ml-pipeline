"""
Modern ML Pipeline - CLI 최초 진입점

이 파일은 두 가지 경로로 실행된다:
  1. 콘솔 스크립트: pip install 후 `mmp train ...` → pyproject.toml이 이 파일의 main()을 호출
  2. 모듈 실행:    `python -m mmp train ...` → Python이 패키지의 __main__.py를 자동 실행

실행 흐름:

    사용자: mmp --quiet train --config my.yaml
     │
     ▼
    __main__.py (이 파일)
     ├── warnings 차단 (라이브러리 경고가 CLI 출력을 오염하지 않도록)
     ├── from mmp.cli.main_commands import app
     │    └→ 이 import 시점에 main_commands.py가 로드되면서
     │       Typer 앱 생성 + 모든 커맨드 등록이 완료됨
     └── main() → app()
              └→ Typer가 sys.argv를 파싱하여 등록된 커맨드를 실행
"""

import warnings

# 라이브러리 경고 차단 — 모든 import보다 먼저 실행해야
# 이후 로드되는 scipy, pydantic 등의 경고가 사용자 터미널에 노출되지 않는다.
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*scipy.sparse.base.*")
warnings.filterwarnings("ignore", message=".*spmatrix.*")

# 이 import가 main_commands.py를 로드하면서 Typer 앱 생성 + 커맨드 등록이 일어난다.
# 등록만 될 뿐, 아직 어떤 커맨드도 실행되지 않는다.
from mmp.cli.main_commands import app


def main():
    """pyproject.toml [project.scripts]가 가리키는 진입점 함수.
    app()을 호출하면 Typer가 커맨드라인 인자를 파싱하고 등록된 커맨드를 실행한다."""
    app()


if __name__ == "__main__":
    # python -m mmp 또는 python mmp/__main__.py로 직접 실행한 경우
    main()
