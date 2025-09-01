"""
get-recipe command implementation
Phase 4: TDD 기반 대화형 모델 선택 구현

CLAUDE.md 원칙 준수:
- TDD: RED → GREEN → REFACTOR  
- 타입 힌트 필수
- Google Style Docstring
"""

import sys
import typer
from typing import Tuple
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel

from src.settings import ModelCatalog, ModelSpec
from src.cli.utils.recipe_generator import CatalogBasedRecipeGenerator


class InteractiveModelSelector:
    """대화형 모델 선택기."""
    
    def __init__(self, catalog: ModelCatalog) -> None:
        """
        Initialize InteractiveModelSelector.
        
        Args:
            catalog: ModelCatalog instance containing available models
        """
        self.catalog = catalog
        self.console = Console()
    
    def _select_environment(self) -> str:
        """
        환경 선택 UI 표시 및 사용자 입력 처리.
        
        Returns:
            str: 선택된 환경 ('local' 또는 'dev')
        """
        self.console.print("\n[bold blue]1. 환경 선택[/bold blue]")
        self.console.print("1) local - 로컬 개발 환경")
        self.console.print("2) dev - 개발 서버 환경")
        
        choice = Prompt.ask("환경을 선택하세요", choices=["1", "2"])
        return "local" if choice == "1" else "dev"
    
    def _select_task(self) -> str:
        """
        태스크 선택 UI 표시 및 사용자 입력 처리.
        
        Returns:
            str: 선택된 태스크 ('Classification', 'Regression', etc.)
        """
        self.console.print("\n[bold blue]2. 태스크 선택[/bold blue]")
        tasks = list(self.catalog.models.keys())
        
        for i, task in enumerate(tasks, 1):
            self.console.print(f"{i}) {task}")
        
        choices = [str(i) for i in range(1, len(tasks) + 1)]
        choice = Prompt.ask("태스크를 선택하세요", choices=choices)
        return tasks[int(choice) - 1]
    
    def _select_model(self, task: str) -> ModelSpec:
        """
        모델 선택 UI 표시 및 사용자 입력 처리.
        
        Args:
            task: 선택된 태스크명
            
        Returns:
            ModelSpec: 선택된 모델 스펙
            
        Raises:
            ValueError: If task not found in catalog
        """
        if task not in self.catalog.models:
            raise ValueError(f"Task '{task}' not found in catalog")
            
        self.console.print(f"\n[bold blue]3. {task} 모델 선택[/bold blue]")
        models = self.catalog.models[task]
        
        # Create a table for better display
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("번호", style="dim", width=6)
        table.add_column("모델", style="cyan")
        table.add_column("라이브러리", style="green")
        table.add_column("설명", style="yellow")
        
        for i, model in enumerate(models, 1):
            description = getattr(model, 'description', 'No description available')
            table.add_row(str(i), model.class_path, model.library, description)
        
        self.console.print(table)
        
        choices = [str(i) for i in range(1, len(models) + 1)]
        choice = Prompt.ask("모델을 선택하세요", choices=choices)
        return models[int(choice) - 1]
    
    def run_interactive_selection(self) -> Tuple[str, str, ModelSpec]:
        """
        대화형 선택 프로세스 실행.
        
        Returns:
            Tuple[str, str, ModelSpec]: (environment, task, model_spec)
            
        Raises:
            SystemExit: If user cancels selection
        """
        # Welcome banner
        self.console.print(Panel(
            "[bold green]🚀 대화형 모델 선택을 시작합니다![/bold green]", 
            title="Recipe Generator",
            border_style="green"
        ))
        
        try:
            # 3단계 선택 프로세스
            environment = self._select_environment()
            task = self._select_task()
            model_spec = self._select_model(task)
            
            # 선택 확인 패널
            confirmation_text = f"""
            [bold cyan]환경:[/bold cyan] {environment}
            [bold cyan]태스크:[/bold cyan] {task}  
            [bold cyan]모델:[/bold cyan] {model_spec.class_path}
            [bold cyan]라이브러리:[/bold cyan] {model_spec.library}
            """
            self.console.print(Panel(
                confirmation_text,
                title="선택 결과 확인",
                border_style="yellow"
            ))
            
            confirm = Prompt.ask("이 설정으로 레시피를 생성하시겠습니까?", choices=["y", "n"], default="y")
            if confirm == "n":
                self.console.print("[red]선택이 취소되었습니다.[/red]")
                sys.exit(0)
            
            return environment, task, model_spec
            
        except KeyboardInterrupt:
            self.console.print("\n[red]사용자에 의해 선택이 취소되었습니다.[/red]")
            sys.exit(0)
        except Exception as e:
            self.console.print(f"[red]선택 과정에서 오류가 발생했습니다: {e}[/red]")
            raise


# RecipeGenerator 클래스는 CatalogBasedRecipeGenerator로 대체됨


def get_recipe_command() -> None:
    """
    get-recipe CLI 명령어 진입점.
    
    대화형 모델 선택 및 레시피 생성을 수행합니다.
    3단계 프로세스를 통해 사용자가 환경, 태스크, 모델을 선택하고
    선택된 설정으로 YAML 레시피 파일을 자동 생성합니다.
    
    Process:
        1. 환경 선택 (local/dev)
        2. 태스크 선택 (Classification, Regression, etc.)
        3. 모델 선택 (카탈로그에서 사용 가능한 모델들)
        4. 레시피 생성 및 파일 저장
        
    Raises:
        typer.Exit: If any step fails or user cancels
    """
    console = Console()
    
    try:
        # Step 1: ModelCatalog 로드
        console.print("[dim]카탈로그 로딩 중...[/dim]")
        catalog = ModelCatalog.from_yaml()
        console.print("[green]✅ 모델 카탈로그가 성공적으로 로드되었습니다.[/green]")
        
        # Step 2: 대화형 선택 실행
        selector = InteractiveModelSelector(catalog)
        environment, task, model_spec = selector.run_interactive_selection()
        
        # Step 3: 레시피 생성 (새로운 Catalog 기반 생성기 사용)
        console.print("\n[dim]레시피 생성 중...[/dim]")
        generator = CatalogBasedRecipeGenerator()
        recipe_path = generator.generate_recipe(environment, task, model_spec)
        
        # Step 4: 성공 메시지 출력
        success_panel = Panel(
            f"""[bold green]✅ 레시피가 성공적으로 생성되었습니다![/bold green]

[bold cyan]파일 경로:[/bold cyan] {recipe_path}
[bold cyan]환경:[/bold cyan] {environment}
[bold cyan]태스크:[/bold cyan] {task}
[bold cyan]모델:[/bold cyan] {model_spec.class_path}

[dim]다음 단계: mmp train {recipe_path}[/dim]""",
            title="Recipe Generation Complete",
            border_style="green"
        )
        console.print(success_panel)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]사용자에 의해 취소되었습니다.[/yellow]")
        raise typer.Exit(code=0)
    except FileNotFoundError as e:
        console.print(f"[red]❌ 파일을 찾을 수 없습니다: {e}[/red]")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[red]❌ 잘못된 값: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]❌ 레시피 생성 중 예상치 못한 오류가 발생했습니다: {e}[/red]")
        console.print("[dim]자세한 오류 정보는 로그를 확인하세요.[/dim]")
        raise typer.Exit(code=1)