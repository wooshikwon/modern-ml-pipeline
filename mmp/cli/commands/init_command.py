"""
Init Command Implementation
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from mmp.cli.utils.header import print_command_header
from mmp.cli.utils.interactive_ui import InteractiveUI
from mmp.cli.utils.template_engine import TemplateEngine


def init_command(project_name: Optional[str] = typer.Argument(None, help="프로젝트 이름")) -> None:
    """
    대화형 프로젝트 초기화.

    사용자에게 프로젝트 이름을 입력받고, 기본 프로젝트 구조를 생성합니다.

    Args:
        project_name: 프로젝트 이름 (선택사항, 없으면 대화형으로 입력)

    생성되는 구조:
        - data/, configs/, recipes/, sql/ 디렉토리
        - docker-compose.yml, Dockerfile
        - pyproject.toml, README.md
        - .gitignore
    """
    ui = InteractiveUI()
    total_steps = 2

    try:
        print_command_header("📦 Init Project", "Interactive project initializer")

        ui.show_panel(
            """새로운 ML 프로젝트를 초기화합니다.

        기본 디렉토리 구조(data, configs, recipes, sql)와
        Docker, pyproject 설정 파일을 생성합니다.""",
            title="Project Initializer",
            style="green",
        )

        # Step 1: 프로젝트명 입력
        ui.show_step(1, total_steps, "프로젝트명 입력")
        if not project_name:
            project_name = ui.text_input(
                "프로젝트 이름을 입력하세요",
                validator=lambda x: len(x) > 0 and x.replace("-", "").replace("_", "").isalnum(),
            )

        # 프로젝트 경로 생성
        project_path = Path.cwd() / project_name

        # 이미 존재하는지 확인
        if project_path.exists():
            if not ui.confirm(f"'{project_name}' 디렉토리가 이미 존재합니다. 계속하시겠습니까?"):
                ui.show_warning("프로젝트 초기화가 취소되었습니다")
                raise typer.Exit(0)

        # Step 2: 프로젝트 구조 생성
        ui.show_step(2, total_steps, "프로젝트 구조 생성")
        ui.show_info(f"프로젝트 '{project_name}'을 생성하는 중...")
        create_project_structure(project_path)

        # 성공 메시지
        ui.show_success(f"프로젝트 '{project_name}'이 생성되었습니다!")
        ui.show_info(f"경로: {project_path.absolute()}")

        # 다음 단계 안내
        next_steps_content = f"""cd {project_name}
mmp get-config        # 환경 설정 생성
mmp get-recipe        # 모델 레시피 생성
mmp train -r recipes/<recipe>.yaml -e <env>  # 학습 실행"""

        ui.show_panel(next_steps_content, title="다음 단계")

    except KeyboardInterrupt:
        ui.show_error("프로젝트 초기화가 취소되었습니다")
        raise typer.Exit(1)
    except Exception as e:
        ui.show_error(f"프로젝트 초기화 중 오류 발생: {e}")
        raise typer.Exit(1)


def create_project_structure(project_path: Path) -> None:
    """
    프로젝트 기본 구조 생성.

    Args:
        project_path: 프로젝트 디렉토리 경로
    """
    # 디렉토리 생성
    directories = ["data", "configs", "recipes", "sql"]
    for dir_name in directories:
        (project_path / dir_name).mkdir(parents=True, exist_ok=True)

    # 템플릿 엔진 초기화
    templates_dir = Path(__file__).parent.parent / "templates"
    template_engine = TemplateEngine(templates_dir)

    # 컨텍스트 준비
    context = {
        "project_name": project_path.name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 템플릿 파일 생성
    files_to_create = [
        ("docker/docker-compose.yml.j2", "docker-compose.yml"),
        ("docker/Dockerfile.j2", "Dockerfile"),
        ("project/pyproject.toml.j2", "pyproject.toml"),
        ("project/README.md.j2", "README.md"),
    ]

    for template_path, output_name in files_to_create:
        template_engine.write_rendered_file(template_path, project_path / output_name, context)

    # 정적 파일 복사 (.gitignore는 템플릿 렌더링 불필요)
    template_engine.copy_static_file("project/.gitignore", project_path / ".gitignore")
