"""
Get-Recipe Command Implementation
Phase 4: Environment-independent recipe generation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 환경 독립적 설계
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


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
        
        # 대화형 플로우 실행
        console.print("\n[dim]대화형 Recipe 생성을 시작합니다...[/dim]\n")
        selections = builder.run_interactive_flow()
        
        # Recipe 파일 생성
        console.print("\n[dim]Recipe 파일을 생성하는 중...[/dim]")
        recipe_path = builder.generate_recipe_file(selections)
        
        # 성공 메시지
        _show_success_message(recipe_path, selections)
        
    except KeyboardInterrupt:
        ui.show_error("Recipe 생성이 취소되었습니다.")
        raise typer.Exit(0)
    except FileNotFoundError as e:
        ui.show_error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(1)
    except ValueError as e:
        ui.show_error(f"잘못된 값: {e}")
        raise typer.Exit(1)
    except Exception as e:
        ui.show_error(f"Recipe 생성 중 오류가 발생했습니다: {e}")
        console.print("[dim]자세한 오류 정보는 로그를 확인하세요.[/dim]")
        raise typer.Exit(1)


def _show_success_message(recipe_path: Path, selections: dict) -> None:
    """
    성공 메시지 표시.
    
    Args:
        recipe_path: 생성된 Recipe 파일 경로
        selections: 사용자 선택 사항
    """
    success_content = f"""✅ [bold green]Recipe가 성공적으로 생성되었습니다![/bold green]

    📄 [bold cyan]파일 경로:[/bold cyan] {recipe_path}
    🎯 [bold cyan]Task:[/bold cyan] {selections['task']}
    🤖 [bold cyan]모델:[/bold cyan] {selections['model_class']}
    📚 [bold cyan]라이브러리:[/bold cyan] {selections['model_library']}

    💡 [bold yellow]다음 단계:[/bold yellow]

    1. Recipe 파일 확인 및 수정:
    [cyan]cat {recipe_path}[/cyan]
    
    2. 필요한 컬럼명 업데이트:
    - target_column을 실제 타겟 컬럼명으로 변경
    - entity_schema를 실제 엔티티 컬럼들로 변경
    - preprocessor steps의 컬럼명 업데이트

    3. 환경과 함께 학습 실행:
    [cyan]mmp train --recipe-file {recipe_path} --env-name <환경명>[/cyan]
    
    예시:
    [cyan]mmp train -r {recipe_path} -e local[/cyan]
    [cyan]mmp train -r {recipe_path} -e dev[/cyan]
    [cyan]mmp train -r {recipe_path} -e prod[/cyan]

    Recipe는 환경과 독립적이므로, 
    동일한 Recipe를 여러 환경에서 사용할 수 있습니다!"""
    
    panel = Panel(
        success_content,
        title="Recipe Generation Complete",
        border_style="green"
    )
    
    console.print(panel)