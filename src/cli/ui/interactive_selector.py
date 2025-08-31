"""
Interactive Selector UI Component
Phase 1: 대화형 선택 UI 컴포넌트

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import List, Tuple, Any, Optional
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table


class InteractiveSelector:
    """대화형 선택 UI 컴포넌트."""
    
    def __init__(self) -> None:
        """Initialize InteractiveSelector."""
        self.console = Console()
    
    def select(
        self, 
        prompt: str, 
        options: List[Tuple[str, Any]],
        default_index: int = 0
    ) -> Any:
        """
        사용자에게 옵션 목록을 표시하고 선택을 받음.
        
        Args:
            prompt: 표시할 프롬프트 메시지
            options: (표시 텍스트, 반환 값) 튜플 리스트
            default_index: 기본 선택 인덱스
            
        Returns:
            선택된 옵션의 값
            
        Raises:
            ValueError: 옵션이 비어있는 경우
        """
        if not options:
            raise ValueError("옵션 목록이 비어있습니다")
        
        # 옵션 표시
        for i, (display_text, _) in enumerate(options, 1):
            style = "bold cyan" if i == default_index + 1 else ""
            self.console.print(f"  {i}) {display_text}", style=style)
        
        # 선택 받기
        choices = [str(i) for i in range(1, len(options) + 1)]
        default_choice = str(default_index + 1)
        
        choice = Prompt.ask(
            f"  {prompt}",
            choices=choices,
            default=default_choice
        )
        
        # 선택된 값 반환
        selected_index = int(choice) - 1
        return options[selected_index][1]
    
    def multi_select(
        self,
        prompt: str,
        options: List[Tuple[str, Any]],
        default_selected: Optional[List[int]] = None
    ) -> List[Any]:
        """
        사용자에게 여러 옵션을 선택할 수 있게 함.
        
        Args:
            prompt: 표시할 프롬프트 메시지
            options: (표시 텍스트, 반환 값) 튜플 리스트
            default_selected: 기본 선택된 인덱스 리스트
            
        Returns:
            선택된 옵션들의 값 리스트
        """
        if not options:
            raise ValueError("옵션 목록이 비어있습니다")
        
        default_selected = default_selected or []
        
        # 옵션 표시
        self.console.print(f"\n{prompt}")
        self.console.print("  (쉼표로 구분하여 여러 개 선택 가능, 예: 1,3,5)")
        
        for i, (display_text, _) in enumerate(options, 1):
            prefix = "[✓]" if i - 1 in default_selected else "[ ]"
            style = "bold cyan" if i - 1 in default_selected else ""
            self.console.print(f"  {prefix} {i}) {display_text}", style=style)
        
        # 선택 받기
        default_choices = ",".join(str(i + 1) for i in default_selected) if default_selected else ""
        input_str = Prompt.ask(
            "  선택",
            default=default_choices
        )
        
        # 파싱 및 검증
        selected_indices = []
        for part in input_str.split(','):
            part = part.strip()
            if part:
                try:
                    index = int(part) - 1
                    if 0 <= index < len(options):
                        selected_indices.append(index)
                except ValueError:
                    continue
        
        # 선택된 값들 반환
        return [options[i][1] for i in selected_indices]
    
    def show_table(
        self,
        title: str,
        headers: List[str],
        rows: List[List[str]],
        highlight_row: Optional[int] = None
    ) -> None:
        """
        테이블 형식으로 데이터 표시.
        
        Args:
            title: 테이블 제목
            headers: 헤더 리스트
            rows: 행 데이터 리스트
            highlight_row: 강조할 행 인덱스
        """
        table = Table(title=title, show_header=True, header_style="bold magenta")
        
        # 헤더 추가
        for header in headers:
            table.add_column(header)
        
        # 행 추가
        for i, row in enumerate(rows):
            style = "bold cyan" if i == highlight_row else ""
            table.add_row(*row, style=style)
        
        self.console.print(table)
    
    def confirm(
        self,
        prompt: str,
        default: bool = True
    ) -> bool:
        """
        예/아니오 확인 프롬프트.
        
        Args:
            prompt: 표시할 프롬프트 메시지
            default: 기본값
            
        Returns:
            사용자 선택 (True/False)
        """
        from rich.prompt import Confirm
        return Confirm.ask(prompt, default=default)