"""
Get-Recipe Command Implementation
"""

import typer
from pathlib import Path

from src.utils.core.console import (
    cli_command_start, cli_command_error, cli_step_complete,
    cli_info, cli_success_panel
)


def get_recipe_command() -> None:
    """
    환경 독립적인 모델 Recipe 생성.
    
    대화형 인터페이스를 통해 Task와 모델을 선택하고,
    환경 설정과 독립적인 Recipe 파일을 생성합니다.
    
    Process:
        1. Recipe 이름 입력
        2. Task 선택 (Classification, Regression, Clustering, Causal)
        3. 모델 선택 (선택한 Task의 사용 가능한 모델들)
        4. 데이터 설정 (source_uri, target_column, entity_schema)
        5. 전처리 및 평가 설정
        6. Recipe 파일 생성 (recipes/{recipe_name}.yaml)
        
    생성된 Recipe는 --env-name 파라미터와 함께 사용:
        mmp train --recipe-file recipes/model.yaml --env-name dev
        
    Raises:
        typer.Exit: 오류 발생 또는 사용자 취소 시
    """
    from src.cli.utils.recipe_builder import RecipeBuilder
    from src.cli.utils.interactive_ui import InteractiveUI
    
    ui = InteractiveUI()
    
    try:
        cli_command_start("Get Recipe", "대화형 Recipe 파일 생성")

        # Welcome message
        ui.show_panel(
            """🚀 환경 독립적인 Recipe 생성을 시작합니다!

        Recipe는 환경 설정과 분리되어 있어,
        다양한 환경에서 재사용할 수 있습니다.""",
                    title="Recipe Generator",
                    style="green"
        )

        # Recipe Builder 초기화
        builder = RecipeBuilder()
        cli_step_complete("초기화", "RecipeBuilder 준비 완료")

        # 대화형 플로우 실행
        cli_info("대화형 Recipe 생성을 시작합니다...")
        selections = builder.run_interactive_flow()

        # Recipe 파일 생성
        cli_step_complete("대화형 플로우", f"Task: {selections.get('task', 'N/A')}, Model: {selections.get('model_class', 'N/A')}")
        cli_info("Recipe 파일을 생성하는 중...")
        recipe_path = builder.generate_recipe_file(selections)

        # 성공 메시지
        cli_step_complete("파일 생성", f"Recipe: {recipe_path}")
        _show_success_message(recipe_path, selections)
        
    except KeyboardInterrupt:
        cli_command_error("Get Recipe", "Recipe 생성이 취소되었습니다")
        raise typer.Exit(0)
    except FileNotFoundError as e:
        cli_command_error("Get Recipe", f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(1)
    except ValueError as e:
        cli_command_error("Get Recipe", f"잘못된 값: {e}")
        raise typer.Exit(1)
    except Exception as e:
        cli_command_error("Get Recipe", f"Recipe 생성 중 오류가 발생했습니다: {e}", "자세한 오류 정보는 로그를 확인하세요")
        raise typer.Exit(1)


def _show_success_message(recipe_path: Path, selections: dict) -> None:
    """
    성공 메시지 표시.
    
    Args:
        recipe_path: 생성된 Recipe 파일 경로
        selections: 사용자 선택 사항
    """
    success_content = f"""✅ Recipe가 성공적으로 생성되었습니다!

    📄 파일 경로: {recipe_path}
    🎯 Task: {selections['task']}
    🤖 모델: {selections['model_class']}
    📚 라이브러리: {selections['model_library']}

    💡 다음 단계:

    1. Recipe 파일 확인 및 수정:
    cat {recipe_path}

    2. 필요한 컬럼명 업데이트:
    - target_column을 실제 타겟 컬럼명으로 변경
    - entity_schema를 실제 엔티티 컬럼들로 변경
    - preprocessor steps의 컬럼명 업데이트

    3. 환경과 함께 학습 실행:
    mmp train --recipe-file {recipe_path} --env-name <환경명>

    예시:
    mmp train -r {recipe_path} -e local
    mmp train -r {recipe_path} -e dev
    mmp train -r {recipe_path} -e prod

    Recipe는 환경과 독립적이므로,
    동일한 Recipe를 여러 환경에서 사용할 수 있습니다!"""
    
    cli_success_panel(success_content, "Recipe 생성 완료")