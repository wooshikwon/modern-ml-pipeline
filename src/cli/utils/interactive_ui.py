"""
Interactive UI Components for Modern ML Pipeline CLI

Rich 라이브러리 기반 대화형 UI 컴포넌트를 제공합니다.
모든 대화형 CLI 명령어(get-config, get-recipe, init)에서 공유합니다.
"""

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table


class InteractiveUI:
    """Rich 라이브러리 기반 대화형 UI 컴포넌트.

    사용자와의 대화형 인터페이스를 위한 다양한 UI 컴포넌트 제공.
    """

    def __init__(self):
        """InteractiveUI 초기화."""
        self.console = Console()

    def select_from_list(
        self, title: str, options: List[str], show_numbers: bool = True, allow_cancel: bool = True
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

        if not options:
            # 선택지가 없을 때는 즉시 예외
            raise ValueError("선택 가능한 옵션이 없습니다")

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
            while True:
                choice = Prompt.ask(
                    "선택",
                    choices=None,  # 테스트 환경에서 side_effect를 허용하기 위해 수동 검증
                    show_choices=False,
                )

                if allow_cancel and choice == "0":
                    return None

                if choice in valid_choices and choice != "0":
                    return options[int(choice) - 1]

                # 잘못된 입력 시 재시도 안내
                self.console.print("[red]올바르지 않은 선택입니다. 다시 시도해주세요.[/red]")

        except KeyboardInterrupt:
            self.console.print("\n[red]취소되었습니다.[/red]")
            raise

    def confirm(self, message: str, default: bool = False, show_default: bool = True) -> bool:
        """Y/N 확인 프롬프트.

        Args:
            message: 확인 메시지
            default: 기본값 (Enter 키만 누를 때)
            show_default: 기본값 표시 여부

        Returns:
            사용자 확인 결과 (True/False)
        """
        return Confirm.ask(
            message, default=default, show_default=show_default, console=self.console
        )

    def text_input(
        self,
        prompt: str,
        default: Optional[str] = None,
        password: bool = False,
        show_default: bool = True,
        validator: Optional[callable] = None,
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
                console=self.console,
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
        max_value: Optional[int] = None,
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
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_value must be <= max_value")

        while True:
            try:
                result = IntPrompt.ask(
                    prompt, default=default, show_default=default is not None, console=self.console
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
        headers: List[str],
        rows: List[List[str]],
        title: Optional[str] = None,
        show_index: bool = False,
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

    def show_panel(self, content: str, title: Optional[str] = None, style: str = "cyan") -> None:
        """패널 형식으로 내용 표시.

        Args:
            content: 표시할 내용
            title: 패널 제목
            style: 패널 테두리 스타일 (무시됨)
        """
        panel = Panel(content, title=(title or "정보"), border_style=style)
        self.console.print(panel)

    def show_progress(self, description: str, total: Optional[int] = None) -> Progress:
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
                console=self.console,
            )
        else:
            # 진행률 바 표시
            return Progress(
                TextColumn("[progress.description]{task.description}"), console=self.console
            )

    def show_success(self, message: str) -> None:
        """성공 메시지 표시.

        Args:
            message: 성공 메시지
        """
        self.console.print(f"[green]{message}[/green]")

    def show_error(self, message: str) -> None:
        """에러 메시지 표시.

        Args:
            message: 에러 메시지
        """
        self.console.print(f"[red]{message}[/red]")

    def show_warning(self, message: str) -> None:
        """경고 메시지 표시.

        Args:
            message: 경고 메시지
        """
        self.console.print(f"[yellow]{message}[/yellow]")

    def show_info(self, message: str) -> None:
        """정보 메시지 표시.

        Args:
            message: 정보 메시지
        """
        self.console.print(f"[cyan]{message}[/cyan]")

    def clear_screen(self) -> None:
        """화면 지우기."""
        self.console.clear()

    def print_divider(self, style: str = "dim") -> None:
        """구분선 출력.

        Args:
            style: 구분선 스타일
        """
        self.console.print(f"[{style}]{'─' * 50}[/{style}]")

    # ===== Validation Helpers =====
    def non_empty_validator(self):
        """공백 제거 후 비어있지 않은지 검사하는 검증 함수 생성."""

        def _validator(value: str) -> bool:
            return bool(value and value.strip())

        return _validator

    @contextmanager
    def show_spinner(self, message: str):
        """스피너 표시 컨텍스트 매니저.

        Args:
            message: 스피너와 함께 표시할 메시지

        Yields:
            None (컨텍스트 매니저로 사용)
        """
        with self.console.status(message):
            yield

    # ===== Multi-select =====
    def multi_select(self, prompt: str, options: List[str]) -> List[str]:
        """쉼표로 구분된 번호 입력으로 다중 선택."""
        # 옵션 표시 (숫자)
        self.console.print(f"\n[bold blue]{prompt}[/bold blue]")
        table = Table(show_header=False, box=None)
        for i, option in enumerate(options, 1):
            table.add_row(f"[cyan]{i})[/cyan]", f"[white]{option}[/white]")
        self.console.print(table)

        raw = Prompt.ask("선택(쉼표로 구분, 비우면 없음)", default="")
        if not raw.strip():
            return []

        selected: List[str] = []
        for token in raw.split(","):
            token = token.strip()
            if token.isdigit():
                idx = int(token)
                if 1 <= idx <= len(options):
                    selected.append(options[idx - 1])
        # 중복 제거, 순서 유지
        seen = set()
        result: List[str] = []
        for x in selected:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result

    # ===== Step Progress =====
    def show_step(self, current: int, total: int, title: str) -> None:
        """단계 진행 표시.

        Args:
            current: 현재 단계 번호 (1부터 시작)
            total: 전체 단계 수
            title: 현재 단계 제목
        """
        # 진행 바 계산
        bar_width = 30
        filled = int((current / total) * bar_width)
        bar = f"[cyan]{'━' * filled}[/cyan][dim]{'━' * (bar_width - filled)}[/dim]"

        self.console.print(f"\n[bold]Step {current}/{total}[/bold] {bar} [bold]{title}[/bold]\n")

    # ===== Single Choice (key-description pairs) =====
    def single_choice(
        self, prompt: str, choices: List[Tuple[str, str]], default: Optional[str] = None
    ) -> str:
        """키-설명 쌍에서 단일 선택.

        Args:
            prompt: 선택 프롬프트 메시지
            choices: (키, 설명) 튜플 리스트
            default: 기본 선택 키 (Enter 시 적용)

        Returns:
            선택된 키 문자열
        """
        self.console.print(f"\n[bold blue]{prompt}[/bold blue]")

        table = Table(show_header=False, box=None)
        for key, desc in choices:
            marker = " [dim](기본값)[/dim]" if key == default else ""
            table.add_row(f"[cyan]{key})[/cyan]", f"[white]{desc}{marker}[/white]")
        self.console.print(table)

        valid_keys = [k for k, _ in choices]

        while True:
            result = Prompt.ask("선택", default=default or "", show_default=False)

            if not result and default:
                return default

            if result in valid_keys:
                return result

            self.console.print(f"[red]유효한 선택지를 입력해주세요: {valid_keys}[/red]")

    # ===== Summary and Confirm =====
    def show_summary_and_confirm(
        self,
        summary_data: Dict[str, str],
        title: str = "설정 요약",
        confirm_message: str = "이 설정으로 진행하시겠습니까?",
    ) -> bool:
        """요약 패널 표시 후 확인.

        Args:
            summary_data: 키-값 쌍의 요약 데이터
            title: 패널 제목
            confirm_message: 확인 메시지

        Returns:
            사용자 확인 결과 (True/False)
        """
        # 요약 내용 생성
        max_key_len = max(len(k) for k in summary_data.keys()) if summary_data else 10
        lines = []
        for key, value in summary_data.items():
            lines.append(f"  {key:<{max_key_len}} : {value}")
        content = "\n".join(lines)

        # 패널 표시
        panel = Panel(content, title=f"[bold]{title}[/bold]", border_style="cyan")
        self.console.print(panel)

        # 확인
        return self.confirm(confirm_message, default=True)
