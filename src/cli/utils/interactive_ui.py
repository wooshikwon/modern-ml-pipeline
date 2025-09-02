"""
Interactive UI Components for Modern ML Pipeline CLI
Phase 1: Rich-based interactive UI components

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 사용자 친화적 인터페이스
"""

from typing import List, Optional, Dict, Any, Union
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich import print as rprint
import sys


class InteractiveUI:
    """Rich 라이브러리 기반 대화형 UI 컴포넌트.
    
    사용자와의 대화형 인터페이스를 위한 다양한 UI 컴포넌트 제공.
    """
    
    def __init__(self):
        """InteractiveUI 초기화."""
        self.console = Console()
    
    def select_from_list(
        self, 
        title: str, 
        options: List[str],
        show_numbers: bool = True,
        allow_cancel: bool = True
    ) -> Optional[str]:
        """화살표로 선택 가능한 리스트 표시 및 선택.
        
        Args:
            title: 선택 프롬프트 제목
            options: 선택 가능한 옵션 리스트
            show_numbers: 번호 표시 여부
            allow_cancel: 취소 허용 여부 (0번 옵션)
            
        Returns:
            선택된 옵션 문자열 또는 None (취소 시)
            
        Raises:
            KeyboardInterrupt: Ctrl+C로 취소 시
        """
        self.console.print(f"\n[bold blue]{title}[/bold blue]")
        
        # 옵션 테이블 생성
        table = Table(show_header=False, box=None)
        
        start_idx = 0 if allow_cancel else 1
        
        if allow_cancel:
            table.add_row("[dim]0)[/dim]", "[dim italic]취소[/dim italic]")
        
        for i, option in enumerate(options, 1):
            if show_numbers:
                table.add_row(f"[cyan]{i})[/cyan]", f"[white]{option}[/white]")
            else:
                table.add_row("", f"[white]{option}[/white]")
        
        self.console.print(table)
        
        # 선택 입력 받기
        valid_choices = [str(i) for i in range(start_idx, len(options) + 1)]
        
        try:
            choice = Prompt.ask(
                "선택",
                choices=valid_choices,
                show_choices=False
            )
            
            if allow_cancel and choice == "0":
                return None
            
            return options[int(choice) - 1]
            
        except KeyboardInterrupt:
            self.console.print("\n[red]취소되었습니다.[/red]")
            raise
    
    def confirm(
        self, 
        message: str, 
        default: bool = False,
        show_default: bool = True
    ) -> bool:
        """Y/N 확인 프롬프트.
        
        Args:
            message: 확인 메시지
            default: 기본값 (Enter 키만 누를 때)
            show_default: 기본값 표시 여부
            
        Returns:
            사용자 확인 결과 (True/False)
        """
        return Confirm.ask(
            message,
            default=default,
            show_default=show_default,
            console=self.console
        )
    
    def text_input(
        self, 
        prompt: str, 
        default: Optional[str] = None,
        password: bool = False,
        show_default: bool = True,
        validator: Optional[callable] = None
    ) -> str:
        """텍스트 입력 프롬프트.
        
        Args:
            prompt: 입력 프롬프트 메시지
            default: 기본값
            password: 비밀번호 입력 모드 (입력 숨김)
            show_default: 기본값 표시 여부
            validator: 입력 검증 함수 (str -> bool)
            
        Returns:
            사용자 입력 문자열
            
        Raises:
            ValueError: 검증 실패 시
        """
        while True:
            result = Prompt.ask(
                prompt,
                default=default,
                password=password,
                show_default=show_default and default is not None,
                console=self.console
            )
            
            if validator:
                if validator(result):
                    return result
                else:
                    self.console.print("[red]올바르지 않은 입력입니다. 다시 시도해주세요.[/red]")
            else:
                return result
    
    def number_input(
        self,
        prompt: str,
        default: Optional[int] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> int:
        """숫자 입력 프롬프트.
        
        Args:
            prompt: 입력 프롬프트 메시지
            default: 기본값
            min_value: 최소값
            max_value: 최대값
            
        Returns:
            사용자 입력 숫자
        """
        while True:
            try:
                result = IntPrompt.ask(
                    prompt,
                    default=default,
                    show_default=default is not None,
                    console=self.console
                )
                
                if min_value is not None and result < min_value:
                    self.console.print(f"[red]최소값은 {min_value}입니다.[/red]")
                    continue
                    
                if max_value is not None and result > max_value:
                    self.console.print(f"[red]최대값은 {max_value}입니다.[/red]")
                    continue
                
                return result
                
            except ValueError:
                self.console.print("[red]올바른 숫자를 입력해주세요.[/red]")
    
    def show_table(
        self,
        title: str,
        headers: List[str],
        rows: List[List[str]],
        show_index: bool = False
    ) -> None:
        """테이블 형식으로 데이터 표시.
        
        Args:
            title: 테이블 제목
            headers: 컬럼 헤더 리스트
            rows: 데이터 행 리스트
            show_index: 인덱스 컬럼 표시 여부
        """
        table = Table(title=title, show_header=True, header_style="bold magenta")
        
        if show_index:
            table.add_column("#", style="dim", width=6)
        
        for header in headers:
            table.add_column(header)
        
        for i, row in enumerate(rows):
            if show_index:
                table.add_row(str(i + 1), *row)
            else:
                table.add_row(*row)
        
        self.console.print(table)
    
    def show_panel(
        self,
        content: str,
        title: Optional[str] = None,
        style: str = "cyan"
    ) -> None:
        """패널 형식으로 내용 표시.
        
        Args:
            content: 표시할 내용
            title: 패널 제목
            style: 패널 테두리 스타일
        """
        panel = Panel(content, title=title, border_style=style)
        self.console.print(panel)
    
    def show_progress(
        self,
        description: str,
        total: Optional[int] = None
    ) -> Progress:
        """진행 상황 표시기 생성.
        
        Args:
            description: 진행 상황 설명
            total: 전체 작업 수 (None이면 스피너만 표시)
            
        Returns:
            Progress 객체 (with 구문과 함께 사용)
        """
        if total is None:
            # 스피너만 표시
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            )
        else:
            # 진행률 바 표시
            return Progress(
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            )
    
    def show_success(self, message: str) -> None:
        """성공 메시지 표시.
        
        Args:
            message: 성공 메시지
        """
        self.console.print(f"✅ [bold green]{message}[/bold green]")
    
    def show_error(self, message: str) -> None:
        """에러 메시지 표시.
        
        Args:
            message: 에러 메시지
        """
        self.console.print(f"❌ [bold red]{message}[/bold red]")
    
    def show_warning(self, message: str) -> None:
        """경고 메시지 표시.
        
        Args:
            message: 경고 메시지
        """
        self.console.print(f"⚠️ [bold yellow]{message}[/bold yellow]")
    
    def show_info(self, message: str) -> None:
        """정보 메시지 표시.
        
        Args:
            message: 정보 메시지
        """
        self.console.print(f"ℹ️ [bold blue]{message}[/bold blue]")
    
    def clear_screen(self) -> None:
        """화면 지우기."""
        self.console.clear()
    
    def print_divider(self, style: str = "dim") -> None:
        """구분선 출력.
        
        Args:
            style: 구분선 스타일
        """
        self.console.print(f"[{style}]{'─' * 50}[/{style}]")