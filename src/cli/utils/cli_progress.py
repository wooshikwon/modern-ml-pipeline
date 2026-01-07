"""
CLI 진행 상태 표시기
터미널에 깔끔한 인라인 진행 상태를 표시합니다.
"""

import sys
import os
from typing import Optional

# [중요] 로거와의 순환 참조를 피하기 위해 환경 변수로 플래그 공유
# MMP_CLI_LINE_ACTIVE: 로그 출력 시 들여쓰기 필요 여부
# MMP_CLI_NEEDS_NEWLINE: 로그 출력 전 줄바꿈 필요 여부 (인라인 모드에서 True)


def set_cli_line_active(active: bool, needs_newline: bool = False) -> None:
    """
    CLI 진행 상태 플래그 설정.

    Args:
        active: 들여쓰기 활성화 여부
        needs_newline: 로그 출력 전 줄바꿈 필요 여부 (기본 모드에서 True)
    """
    os.environ["MMP_CLI_LINE_ACTIVE"] = "1" if active else "0"
    os.environ["MMP_CLI_NEEDS_NEWLINE"] = "1" if needs_newline else "0"


def is_cli_line_active() -> bool:
    """현재 진행바 라인이 출력 중인지 확인"""
    return os.environ.get("MMP_CLI_LINE_ACTIVE") == "1"


def needs_newline_before_log() -> bool:
    """로그 출력 전 줄바꿈이 필요한지 확인"""
    return os.environ.get("MMP_CLI_NEEDS_NEWLINE") == "1"

class CLIProgress:
    """
    인라인 진행 상태 표시 클래스.

    기본 모드: [1/6] Loading config         done
    상세 모드 (-v):
        [1/6] Loading config
          [CONFIG] Recipe: test.yaml
          ✓ done
    """

    def __init__(self, total_steps: int, show_version: bool = True, verbose: bool = False):
        self.total_steps = total_steps
        self.current_step = 0
        self.show_version = show_version
        self.verbose = verbose
        self._step_width = 30

    def header(self, version: str) -> None:
        """버전 헤더 출력"""
        if self.show_version:
            sys.stdout.write(f"\nmmp v{version}\n\n")
            sys.stdout.flush()

    def step_start(self, name: str) -> None:
        """단계 시작 표시"""
        self.current_step += 1
        step_prefix = f"[{self.current_step}/{self.total_steps}]"
        step_text = f"{step_prefix} {name}"

        if is_cli_line_active():
            sys.stdout.write("\n")

        if self.verbose:
            # 상세 모드: 줄바꿈으로 종료, 상세 로그가 들여쓰기되어 출력됨
            sys.stdout.write(f"{step_text}\n")
            sys.stdout.flush()
            set_cli_line_active(True, needs_newline=False)  # 들여쓰기만, 줄바꿈 불필요
        else:
            # 기본 모드: 인라인 대기 (done이 같은 줄에 출력됨)
            sys.stdout.write(f"{step_text:<{self._step_width}}")
            sys.stdout.flush()
            set_cli_line_active(True, needs_newline=True)  # 줄바꿈 + 들여쓰기

    def step_done(self, stats: str = "") -> None:
        """단계 완료 표시"""
        if self.verbose:
            # 상세 모드: ✓ 마커와 함께 들여쓰기된 별도 라인
            if stats:
                sys.stdout.write(f"  ✓ {stats}\n")
            else:
                sys.stdout.write("  ✓ done\n")
        else:
            # 기본 모드: 인라인 완료
            if stats:
                sys.stdout.write(f"  done  {stats}\n")
            else:
                sys.stdout.write("  done\n")
        sys.stdout.flush()
        set_cli_line_active(False)

    def step_fail(self, error: Optional[str] = None) -> None:
        """단계 실패 표시"""
        sys.stdout.write("  fail\n")
        if error:
            sys.stdout.write(f"  [ERROR] {error}\n")
        sys.stdout.flush()
        set_cli_line_active(False)

    def step_skip(self, reason: str = "") -> None:
        """단계 건너뜀 표시"""
        if reason:
            sys.stdout.write(f"  skip  {reason}\n")
        else:
            sys.stdout.write("  skip\n")
        sys.stdout.flush()
        set_cli_line_active(False)

    def result(
        self,
        run_id: str,
        mlflow_url: Optional[str] = None,
        artifact_uri: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> None:
        """최종 결과 출력"""
        if is_cli_line_active():
            sys.stdout.write("\n")

        sys.stdout.write(f"\nRun ID: {run_id}\n")
        if artifact_uri:
            sys.stdout.write(f"Artifacts: {artifact_uri}\n")
        if log_file:
            sys.stdout.write(f"Log: {log_file}\n")
        if mlflow_url:
            sys.stdout.write(f"MLflow: {mlflow_url}\n")
        sys.stdout.flush()
        set_cli_line_active(False)
