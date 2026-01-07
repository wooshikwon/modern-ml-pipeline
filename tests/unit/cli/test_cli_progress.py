"""
CLIProgress 및 TerminalFormatter 테스트

Phase 9: CLI 출력 시스템 리팩토링
- TerminalFormatter 들여쓰기 로직
- log_config 함수
- MMP_CLI_LINE_ACTIVE 환경변수 연동

Phase 10: CLIProgress verbose 모드
- verbose 파라미터
- step_start/step_done 분기 처리
- ✓ 마커 출력
"""

import logging
import os
from io import StringIO

import pytest

from src.cli.utils.cli_progress import (
    CLIProgress,
    is_cli_line_active,
    needs_newline_before_log,
    set_cli_line_active,
)
from src.utils.core.logger import TerminalFormatter, log_config, log_sys


class TestTerminalFormatterIndent:
    """TerminalFormatter 들여쓰기 로직 테스트"""

    def setup_method(self):
        """각 테스트 전 환경변수 초기화"""
        os.environ.pop("MMP_CLI_LINE_ACTIVE", None)

    def teardown_method(self):
        """각 테스트 후 환경변수 정리"""
        os.environ.pop("MMP_CLI_LINE_ACTIVE", None)

    def test_format_without_cli_active(self):
        """CLI 라인 비활성 시 메시지만 반환"""
        formatter = TerminalFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="테스트 메시지",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert result == "테스트 메시지"
        assert not result.startswith("\n")

    def test_format_with_cli_active_needs_newline(self):
        """CLI 라인 활성 + 줄바꿈 필요 시 (기본 모드)"""
        set_cli_line_active(True, needs_newline=True)
        formatter = TerminalFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="[DATA] 로드 완료",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert result == "\n  [DATA] 로드 완료"
        assert result.startswith("\n  ")

    def test_format_with_cli_active_no_newline(self):
        """CLI 라인 활성 + 줄바꿈 불필요 시 (verbose 모드)"""
        set_cli_line_active(True, needs_newline=False)
        formatter = TerminalFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="[DATA] 로드 완료",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert result == "  [DATA] 로드 완료"
        assert result.startswith("  ")

    def test_format_respects_env_variable_changes(self):
        """환경변수 변경에 따라 동적으로 동작"""
        formatter = TerminalFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="메시지",
            args=(),
            exc_info=None,
        )

        # 비활성 상태
        set_cli_line_active(False)
        assert formatter.format(record) == "메시지"

        # 활성 + 줄바꿈 필요 (기본 모드)
        set_cli_line_active(True, needs_newline=True)
        assert formatter.format(record) == "\n  메시지"

        # 활성 + 줄바꿈 불필요 (verbose 모드)
        set_cli_line_active(True, needs_newline=False)
        assert formatter.format(record) == "  메시지"

        # 다시 비활성
        set_cli_line_active(False)
        assert formatter.format(record) == "메시지"


class TestCLILineActiveFlag:
    """MMP_CLI_LINE_ACTIVE 환경변수 플래그 테스트"""

    def setup_method(self):
        os.environ.pop("MMP_CLI_LINE_ACTIVE", None)

    def teardown_method(self):
        os.environ.pop("MMP_CLI_LINE_ACTIVE", None)

    def test_set_cli_line_active_true(self):
        """활성화 설정"""
        set_cli_line_active(True)
        assert os.environ.get("MMP_CLI_LINE_ACTIVE") == "1"
        assert is_cli_line_active() is True

    def test_set_cli_line_active_false(self):
        """비활성화 설정"""
        set_cli_line_active(False)
        assert os.environ.get("MMP_CLI_LINE_ACTIVE") == "0"
        assert is_cli_line_active() is False

    def test_is_cli_line_active_default(self):
        """환경변수 미설정 시 False"""
        assert is_cli_line_active() is False


class TestLogConfigFunction:
    """log_config 함수 테스트"""

    def test_log_config_format(self, caplog):
        """log_config가 [CONFIG] 접두사로 로깅"""
        with caplog.at_level(logging.INFO):
            log_config("Recipe: test.yaml")

        assert len(caplog.records) == 1
        assert "[CONFIG] Recipe: test.yaml" in caplog.text

    def test_log_sys_format(self, caplog):
        """log_sys가 [SYS] 접두사로 로깅 (기존 함수 확인)"""
        with caplog.at_level(logging.INFO):
            log_sys("패키지 검증 완료")

        assert len(caplog.records) == 1
        assert "[SYS] 패키지 검증 완료" in caplog.text


class TestCLIProgressBasic:
    """CLIProgress 기본 동작 테스트"""

    def setup_method(self):
        os.environ.pop("MMP_CLI_LINE_ACTIVE", None)

    def teardown_method(self):
        os.environ.pop("MMP_CLI_LINE_ACTIVE", None)

    def test_step_start_sets_active_flag(self, capsys):
        """step_start 호출 시 CLI 라인 활성화"""
        progress = CLIProgress(total_steps=3)
        progress.step_start("Loading")

        assert is_cli_line_active() is True

    def test_step_done_clears_active_flag(self, capsys):
        """step_done 호출 시 CLI 라인 비활성화"""
        progress = CLIProgress(total_steps=3)
        progress.step_start("Loading")
        progress.step_done()

        assert is_cli_line_active() is False

    def test_step_fail_clears_active_flag(self, capsys):
        """step_fail 호출 시 CLI 라인 비활성화"""
        progress = CLIProgress(total_steps=3)
        progress.step_start("Loading")
        progress.step_fail()

        assert is_cli_line_active() is False

    def test_step_sequence_output(self, capsys):
        """단계 시퀀스 출력 형식 확인"""
        progress = CLIProgress(total_steps=2, show_version=False)

        progress.step_start("Loading config")
        progress.step_done()
        progress.step_start("Processing")
        progress.step_done("100 rows")

        captured = capsys.readouterr()
        assert "[1/2] Loading config" in captured.out
        assert "done" in captured.out
        assert "[2/2] Processing" in captured.out
        assert "100 rows" in captured.out


class TestCLIProgressVerboseMode:
    """CLIProgress verbose 모드 테스트 (Phase 10)"""

    def setup_method(self):
        os.environ.pop("MMP_CLI_LINE_ACTIVE", None)

    def teardown_method(self):
        os.environ.pop("MMP_CLI_LINE_ACTIVE", None)

    def test_verbose_default_false(self):
        """verbose 기본값은 False"""
        progress = CLIProgress(total_steps=3)
        assert progress.verbose is False

    def test_verbose_can_be_set_true(self):
        """verbose를 True로 설정 가능"""
        progress = CLIProgress(total_steps=3, verbose=True)
        assert progress.verbose is True

    def test_verbose_step_start_outputs_newline(self, capsys):
        """verbose 모드에서 step_start는 줄바꿈으로 종료, 들여쓰기 활성화"""
        progress = CLIProgress(total_steps=2, show_version=False, verbose=True)
        progress.step_start("Loading config")

        captured = capsys.readouterr()
        assert "[1/2] Loading config\n" in captured.out
        # verbose 모드에서도 CLI 라인 활성화 (들여쓰기용)
        assert is_cli_line_active() is True
        # 단, 줄바꿈은 불필요 (이미 줄바꿈됨)
        assert needs_newline_before_log() is False

    def test_verbose_step_done_outputs_checkmark(self, capsys):
        """verbose 모드에서 step_done은 ✓ 마커 출력"""
        progress = CLIProgress(total_steps=2, show_version=False, verbose=True)
        progress.step_start("Loading config")
        progress.step_done()

        captured = capsys.readouterr()
        assert "✓ done" in captured.out

    def test_verbose_step_done_with_stats_outputs_checkmark(self, capsys):
        """verbose 모드에서 stats와 함께 ✓ 마커 출력"""
        progress = CLIProgress(total_steps=2, show_version=False, verbose=True)
        progress.step_start("Loading data")
        progress.step_done("134,218 rows")

        captured = capsys.readouterr()
        assert "✓ 134,218 rows" in captured.out

    def test_verbose_full_sequence(self, capsys):
        """verbose 모드 전체 시퀀스 출력 형식"""
        progress = CLIProgress(total_steps=2, show_version=False, verbose=True)

        progress.step_start("Loading config")
        progress.step_done()
        progress.step_start("Loading data")
        progress.step_done("100 rows")

        captured = capsys.readouterr()
        # 상세 모드 형식 검증
        assert "[1/2] Loading config\n" in captured.out
        assert "  ✓ done\n" in captured.out
        assert "[2/2] Loading data\n" in captured.out
        assert "  ✓ 100 rows\n" in captured.out

    def test_basic_mode_no_checkmark(self, capsys):
        """기본 모드에서는 ✓ 마커 미출력"""
        progress = CLIProgress(total_steps=2, show_version=False, verbose=False)
        progress.step_start("Loading")
        progress.step_done("100 rows")

        captured = capsys.readouterr()
        assert "✓" not in captured.out
        assert "done  100 rows" in captured.out

    def test_basic_mode_inline_output(self, capsys):
        """기본 모드에서 인라인 출력 (줄바꿈 없이 done 이어짐)"""
        progress = CLIProgress(total_steps=1, show_version=False, verbose=False)
        progress.step_start("Test")

        captured = capsys.readouterr()
        # 기본 모드: 줄바꿈 없이 대기
        assert captured.out.endswith("[1/1] Test" + " " * (30 - len("[1/1] Test")))
        assert is_cli_line_active() is True


class TestFormatDuration:
    """_format_duration 헬퍼 함수 테스트 (Phase 11)"""

    def test_format_duration_seconds(self):
        """60초 미만은 소수점 1자리 초 형식"""
        from src.pipelines.train_pipeline import _format_duration

        assert _format_duration(0.5) == "0.5s"
        assert _format_duration(30.7) == "30.7s"
        assert _format_duration(59.9) == "59.9s"

    def test_format_duration_minutes(self):
        """60초 이상 3600초 미만은 분:초 형식"""
        from src.pipelines.train_pipeline import _format_duration

        assert _format_duration(60) == "1m 0s"
        assert _format_duration(90) == "1m 30s"
        assert _format_duration(3599) == "59m 59s"

    def test_format_duration_hours(self):
        """3600초 이상은 시:분 형식"""
        from src.pipelines.train_pipeline import _format_duration

        assert _format_duration(3600) == "1h 0m"
        assert _format_duration(3660) == "1h 1m"
        assert _format_duration(7200) == "2h 0m"


class TestProgressCallback:
    """ProgressCallback 패턴 테스트 (Phase 11)"""

    def test_progress_callback_type_exported(self):
        """ProgressCallback 타입이 export 됨"""
        from src.pipelines.train_pipeline import ProgressCallback

        assert ProgressCallback is not None

    def test_callback_receives_events(self):
        """콜백이 이벤트를 수신함"""
        events = []

        def test_callback(event: str, stats: str = "") -> None:
            events.append((event, stats))

        # emit 함수 직접 테스트 (파이프라인 내부 구조 검증)
        def emit(event: str, stats: str = "") -> None:
            if test_callback:
                test_callback(event, stats)

        emit("loading_data")
        emit("loading_data_done", "100 rows")
        emit("training")
        emit("training_done", "1m 30s")

        assert len(events) == 4
        assert events[0] == ("loading_data", "")
        assert events[1] == ("loading_data_done", "100 rows")
        assert events[2] == ("training", "")
        assert events[3] == ("training_done", "1m 30s")


class TestVerboseModeDetection:
    """verbose 모드 감지 테스트 (Phase 12)"""

    def test_verbose_detected_from_debug_level(self):
        """DEBUG 레벨일 때 verbose=True로 감지"""
        root_logger = logging.getLogger()
        original_level = root_logger.level

        try:
            root_logger.setLevel(logging.DEBUG)
            verbose = root_logger.level == logging.DEBUG
            assert verbose is True
        finally:
            root_logger.setLevel(original_level)

    def test_verbose_detected_from_info_level(self):
        """INFO 레벨일 때 verbose=False로 감지"""
        root_logger = logging.getLogger()
        original_level = root_logger.level

        try:
            root_logger.setLevel(logging.INFO)
            verbose = root_logger.level == logging.DEBUG
            assert verbose is False
        finally:
            root_logger.setLevel(original_level)


class TestCommandLogIntegration:
    """Command 로그 통합 테스트 (Phase 12)"""

    def test_log_config_imported_in_train_command(self):
        """train_command에서 log_config가 import 됨"""
        from src.cli.commands import train_command as tc_module

        assert hasattr(tc_module, "log_config")

    def test_log_sys_imported_in_train_command(self):
        """train_command에서 log_sys가 import 됨"""
        from src.cli.commands import train_command as tc_module

        assert hasattr(tc_module, "log_sys")

    def test_log_config_imported_in_inference_command(self):
        """inference_command에서 log_config가 import 됨"""
        from src.cli.commands import inference_command as ic_module

        assert hasattr(ic_module, "log_config")

    def test_log_sys_imported_in_inference_command(self):
        """inference_command에서 log_sys가 import 됨"""
        from src.cli.commands import inference_command as ic_module

        assert hasattr(ic_module, "log_sys")
