"""
Console Manager 테스트 (Rich Console 통합 시스템 핵심 모듈)
tests/README.md 전략 준수: 컨텍스트 기반, 퍼블릭 API, 실제 객체, 결정론적

테스트 대상 핵심 기능:
- RichConsoleManager - 20+ methods with Rich formatting
- UnifiedConsole - Dual output system (Rich + Logger)  
- Context managers - pipeline_context, progress_tracker
- Specialized logging - component_init, data_operation, error handling
- Environment detection - CI, test mode, console mode
- Helper functions - CLI helpers, test helpers

핵심 Edge Cases:
- CI/CD 환경 감지 및 plain 모드 전환
- 테스트 환경에서의 console 동작
- Progress bar 생성/정리 및 메모리 관리
- 유니코드/이모지 출력 처리
- Context manager 중첩 및 예외 처리
- 대용량 출력 및 성능 최적화
"""
import pytest
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
from io import StringIO

from src.utils.core.console_manager import (
    RichConsoleManager,
    UnifiedConsole,
    console_manager,
    unified_console,
    get_console,
    get_rich_console,
    cli_print,
    cli_success,
    cli_error,
    cli_warning,
    cli_info,
    testing_print,
    phase_print,
    success_print,
    testing_info
)


class TestRichConsoleManager:
    """RichConsoleManager 클래스 핵심 테스트"""
    
    def test_console_manager_initialization(self):
        """케이스 A: RichConsoleManager 초기화"""
        # When: 새로운 console manager 생성
        manager = RichConsoleManager()
        
        # Then: 올바른 초기 상태
        assert manager.console is not None
        assert manager.current_pipeline is None
        assert manager.progress_bars == {}
        assert manager.iteration_counters == {}
        assert manager.active_progress is None
    
    def test_log_milestone_with_different_levels(self):
        """케이스 B: 다양한 레벨의 마일스톤 로깅"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When & Then: 각 레벨별 마일스톤 로깅 (에러 없이 실행)
        test_cases = [
            ("Test info message", "info"),
            ("Test success message", "success"), 
            ("Test warning message", "warning"),
            ("Test error message", "error"),
            ("Test start message", "start"),
            ("Test data message", "data"),
            ("Test model message", "model"),
            ("Test optimization message", "optimization"),
            ("Test mlflow message", "mlflow"),
            ("Test finish message", "finish"),
            ("Test unknown level", "unknown_level")  # 기본 이모지 사용
        ]
        
        for message, level in test_cases:
            # 에러 없이 로깅 실행되어야 함
            manager.log_milestone(message, level)
    
    def test_pipeline_context_manager_success(self):
        """케이스 C: Pipeline context manager 정상 동작"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: Pipeline context 사용
        with manager.pipeline_context("Test Pipeline", "Test Description"):
            assert manager.current_pipeline == "Test Pipeline"
            # Pipeline 내부에서 다른 작업 수행
            manager.log_milestone("Inside pipeline", "info")
        
        # Then: Pipeline 종료 후 상태 초기화
        assert manager.current_pipeline is None
    
    def test_pipeline_context_manager_with_exception(self):
        """케이스 D: Pipeline context manager 예외 처리"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: Pipeline 내부에서 예외 발생
        with pytest.raises(ValueError):
            with manager.pipeline_context("Test Pipeline", "Test Description"):
                assert manager.current_pipeline == "Test Pipeline"
                raise ValueError("Test exception")
        
        # Then: 예외 발생 후에도 상태 정리됨
        assert manager.current_pipeline is None
    
    def test_progress_tracker_with_progress_bar(self):
        """케이스 E: Progress tracker with progress bar 모드"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: Progress tracker 사용 (show_progress=True)
        with manager.progress_tracker("test_task", 3, "Processing items", show_progress=True) as update:
            # Progress update function이 제공됨
            assert callable(update)
            
            # Progress 업데이트 (에러 없이 실행)
            update(1)
            update(2)
            update(3)
        
        # Then: Progress tracker 정리됨
        assert "test_task" not in manager.progress_bars
    
    def test_progress_tracker_without_progress_bar(self):
        """케이스 F: Progress tracker without progress bar (hyperopt 모드)"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: Progress tracker 사용 (show_progress=False)
        with manager.progress_tracker("hyperopt_task", 100, "Hyperparameter optimization", show_progress=False) as update:
            # No-op function이 제공됨
            assert callable(update)
            
            # Update 호출해도 에러 없음
            update(50)
            update(100)
        
        # Then: Progress bars에 추가되지 않음
        assert "hyperopt_task" not in manager.progress_bars
    
    def test_log_periodic_optuna_trials(self):
        """케이스 G: Optuna trials 주기적 로깅"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: Optuna trials 로깅
        trial_data = {
            "trial": 5,
            "total_trials": 20,
            "score": 0.8545,
            "best_score": 0.9123,
            "params": {"n_estimators": 100, "max_depth": 5}
        }
        
        # 주기적 로깅 (에러 없이 실행)
        manager.log_periodic("optuna_trials", 4, trial_data, every_n=5)  # iteration=4, every_n=5 -> 출력됨
        manager.log_periodic("optuna_trials", 6, trial_data, every_n=5)  # iteration=6, every_n=5 -> 출력 안됨
        manager.log_periodic("optuna_trials", 9, trial_data, every_n=5)  # iteration=9, every_n=5 -> 출력됨
    
    def test_log_periodic_generic_process(self):
        """케이스 H: 일반적인 프로세스 주기적 로깅"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: 일반 프로세스 로깅
        generic_data = {"loss": 0.123, "accuracy": 0.876}
        
        # 주기적 로깅 (에러 없이 실행)
        manager.log_periodic("training", 0, generic_data, every_n=10)  # iteration=0 -> 항상 출력
        manager.log_periodic("training", 9, generic_data, every_n=10)  # iteration=9, every_n=10 -> 출력됨
        manager.log_periodic("training", 5, generic_data, every_n=10)  # iteration=5, every_n=10 -> 출력 안됨
    
    def test_display_metrics_table(self):
        """케이스 I: 메트릭 테이블 출력"""
        # Given: Console manager와 메트릭 데이터
        manager = RichConsoleManager()
        
        metrics = {
            "accuracy": 0.8547,
            "precision": 0.8123,
            "recall": 0.7956,
            "f1_score": 0.8038,
            "auc_score": 0.9234,
            "train_samples": 1000,  # 정수값
            "model_name": "RandomForest"  # 문자열
        }
        
        # When: 메트릭 테이블 출력 (에러 없이 실행)
        manager.display_metrics_table(metrics, title="Final Results")
        
        # Then: 에러 없이 완료 (실제 Rich 테이블 생성됨)
    
    def test_display_run_info(self):
        """케이스 J: MLflow run 정보 출력"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: 다양한 run 정보 출력 케이스들
        # 기본 run_id만
        manager.display_run_info("abc123def456")
        
        # run_id + model_uri
        manager.display_run_info(
            "abc123def456",
            model_uri="models:/MyModel/1"
        )
        
        # 모든 정보 포함
        manager.display_run_info(
            "abc123def456", 
            model_uri="models:/MyModel/1",
            tracking_uri="http://localhost:5000"
        )
        
        # 실험 링크 포함
        manager.display_run_info(
            "abc123def456",
            tracking_uri="http://localhost:5000/#/experiments/1/runs/abc123def456"
        )
        
        # Then: 에러 없이 모든 케이스 실행됨
    
    def test_log_artifacts_progress(self):
        """케이스 K: 아티팩트 업로드 진행률 로깅"""
        # Given: Console manager와 아티팩트 목록
        manager = RichConsoleManager()
        
        artifacts = [
            "model.pkl",
            "preprocessor.pkl", 
            "metrics.json",
            "feature_importance.png",
            "confusion_matrix.png"
        ]
        
        # When: 아티팩트 진행률 로깅 (에러 없이 실행, 시뮬레이션 업로드 시간 포함)
        manager.log_artifacts_progress(artifacts)
        
        # Then: 에러 없이 완료
    
    def test_cleanup_completed_tasks(self):
        """케이스 L: 완료된 작업 정리"""
        # Given: Console manager with mock progress bars
        manager = RichConsoleManager()
        
        # Mock progress bars (완료된 것과 진행중인 것)
        mock_completed_task = Mock()
        mock_completed_task.finished = True
        
        mock_ongoing_task = Mock() 
        mock_ongoing_task.finished = False
        
        manager.progress_bars = {
            "completed_task": (Mock(), mock_completed_task),
            "ongoing_task": (Mock(), mock_ongoing_task)
        }
        
        # When: 완료된 작업 정리
        manager.cleanup_completed_tasks()
        
        # Then: 완료된 작업만 제거됨
        assert "completed_task" not in manager.progress_bars
        assert "ongoing_task" in manager.progress_bars
    
    def test_environment_detection_methods(self):
        """케이스 M: 환경 감지 메소드들"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When & Then: CI 환경 감지 테스트
        with patch.dict(os.environ, {}, clear=True):
            # CI 환경이 아님
            assert manager.is_ci_environment() is False
            assert manager.get_console_mode() == "rich"
        
        with patch.dict(os.environ, {"CI": "true"}, clear=True):
            # CI 환경임
            assert manager.is_ci_environment() is True
            assert manager.get_console_mode() == "plain"
        
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True):
            # GitHub Actions 환경
            assert manager.is_ci_environment() is True
            assert manager.get_console_mode() == "plain"
        
        # TTY 감지 테스트
        with patch('sys.stdout.isatty', return_value=False):
            assert manager.get_console_mode() == "plain"  # 파이프/리디렉션
    
    def test_enhanced_logging_methods(self):
        """케이스 N: 향상된 로깅 메소드들"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: 각종 전문 로깅 메소드 실행 (에러 없이)
        
        # Component 초기화
        manager.log_component_init("DataAdapter", "success")
        manager.log_component_init("ModelTrainer", "error")
        
        # 처리 단계
        manager.log_processing_step("Data loading", "Loading 1000 records")
        manager.log_processing_step("Feature engineering")
        
        # 경고 with context
        manager.log_warning_with_context("Memory usage high", {"used": "8GB", "available": "2GB"})
        
        # 데이터베이스 작업
        manager.log_database_operation("Table creation", "Created users table")
        
        # 피처 엔지니어링
        manager.log_feature_engineering("Scaling", ["feature1", "feature2"], "Scaled to [0,1]")
        manager.log_feature_engineering("Encoding", ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6"], "One-hot encoded")
        
        # 데이터 작업
        manager.log_data_operation("Data loading", shape=(1000, 10), details="From PostgreSQL")
        
        # 모델 작업
        manager.log_model_operation("Model training", "RandomForest with 100 trees")
        
        # 파일 작업
        manager.log_file_operation("File save", "/path/to/model.pkl", "Binary format")
        
        # 에러 with context & suggestion
        manager.log_error_with_context(
            "Model training failed",
            context={"memory": "8GB", "dataset_size": "100K samples"},
            suggestion="Try reducing batch size or using a smaller model"
        )
        
        # 검증 결과
        manager.log_validation_result("Schema validation", "pass", "All columns present")
        manager.log_validation_result("Data quality", "fail", "Missing values found")
        manager.log_validation_result("Model performance", "warning", "Accuracy below threshold")
        
        # 연결 상태
        manager.log_connection_status("PostgreSQL", "connected", "Connection pool: 10/20")
        manager.log_connection_status("Redis", "failed", "Connection timeout")
        
        # Then: 모든 로깅 메소드가 에러 없이 실행됨


class TestUnifiedConsole:
    """UnifiedConsole 클래스 테스트 (Dual Output System)"""
    
    def test_unified_console_initialization(self):
        """케이스 A: UnifiedConsole 초기화"""
        # When: UnifiedConsole 생성
        console = UnifiedConsole()
        
        # Then: 올바른 초기화
        assert console.rich_console is not None
        assert console.logger is not None
        assert console.mode in ["rich", "plain", "test"]
    
    def test_unified_console_info_logging(self):
        """케이스 B: Info 로깅 (dual output)"""
        # Given: UnifiedConsole
        console = UnifiedConsole()
        
        # When: 다양한 info 로깅
        console.info("Basic info message")
        console.info("Structured info", rich_message="🔍 [bold]Structured info[/bold]")
        console.info("Info with kwargs", rich_message="Info message", style="cyan")
        
        # Then: 에러 없이 실행됨 (logger와 rich 모두 호출)
    
    def test_unified_console_error_logging(self):
        """케이스 C: Error 로깅 with context and suggestions"""
        # Given: UnifiedConsole  
        console = UnifiedConsole()
        
        # When: 다양한 error 로깅
        console.error("Basic error message")
        console.error("Structured error", rich_message="❌ [red]Critical Error[/red]")
        console.error(
            "Contextual error",
            context={"module": "data_loader", "line": 42},
            suggestion="Check file permissions"
        )
        
        # Then: 에러 없이 실행됨
    
    def test_unified_console_warning_logging(self):
        """케이스 D: Warning 로깅"""
        # Given: UnifiedConsole
        console = UnifiedConsole()
        
        # When: Warning 로깅
        console.warning("Basic warning")
        console.warning("Structured warning", rich_message="⚠️ [yellow]Warning[/yellow]")
        console.warning("Contextual warning", context={"threshold": 0.8, "current": 0.75})
        
        # Then: 에러 없이 실행됨
    
    def test_unified_console_debug_logging(self):
        """케이스 E: Debug 로깅"""
        # Given: UnifiedConsole
        console = UnifiedConsole()
        
        # When: Debug 로깅
        console.debug("Basic debug message")
        console.debug("Structured debug", rich_message="[dim]Debug info[/dim]")
        console.debug("Debug with details", rich_message="Debug", style="dim")
        
        # Then: 에러 없이 실행됨
    
    def test_unified_console_specialized_logging(self):
        """케이스 F: 전문 로깅 메소드들"""
        # Given: UnifiedConsole
        console = UnifiedConsole()
        
        # When: 전문 로깅 실행
        console.component_init("DataProcessor", "success")
        console.component_init("ModelTrainer", "error")
        
        console.data_operation("Data loading", shape=(1000, 50), details="From database")
        console.data_operation("Data preprocessing")
        
        # Then: 에러 없이 실행됨
    
    def test_unified_console_mode_detection(self):
        """케이스 G: 출력 모드 자동 감지"""
        # Given: 다양한 환경 시나리오
        
        # 테스트 환경
        with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_case"}, clear=True):
            console = UnifiedConsole()
            assert console.mode == "test"
        
        # CI 환경
        with patch.dict(os.environ, {"CI": "true"}, clear=True):
            console = UnifiedConsole()
            assert console.mode == "plain"
        
        # 일반 환경
        with patch.dict(os.environ, {}, clear=True):
            console = UnifiedConsole()
            # rich 또는 plain (TTY 여부에 따라)
            assert console.mode in ["rich", "plain"]
    
    def test_unified_console_plain_mode_output(self):
        """케이스 H: Plain 모드에서의 출력"""
        # Given: Plain 모드 console
        with patch.dict(os.environ, {"CI": "true"}, clear=True):
            console = UnifiedConsole()
            assert console.mode == "plain"
            
            # When: 다양한 로깅 (plain 모드 출력)
            with patch('builtins.print') as mock_print:
                console.info("Test info message")
                console.error("Test error message")
                console.warning("Test warning message")
                console.debug("Test debug message")
                console.component_init("TestComponent", "success")
                console.data_operation("Test operation", shape=(100, 10))
                
                # Then: print 함수가 호출됨 (Rich 대신)
                assert mock_print.call_count >= 6  # 각 로깅당 최소 1번 호출


class TestGlobalHelperFunctions:
    """전역 헬퍼 함수들 테스트"""
    
    def test_get_console_function(self):
        """케이스 A: get_console 함수"""
        # When: get_console 호출
        console = get_console()
        
        # Then: UnifiedConsole 인스턴스 반환
        assert isinstance(console, UnifiedConsole)
        
        # Settings와 함께 호출
        mock_settings = Mock()
        mock_settings.console_mode = "plain"
        console_with_settings = get_console(mock_settings)
        assert isinstance(console_with_settings, UnifiedConsole)
    
    def test_get_rich_console_function(self):
        """케이스 B: get_rich_console 함수"""
        # When: get_rich_console 호출
        rich_console = get_rich_console()
        
        # Then: RichConsoleManager 인스턴스 반환
        assert isinstance(rich_console, RichConsoleManager)
    
    def test_cli_helper_functions(self):
        """케이스 C: CLI 헬퍼 함수들"""
        # When: CLI 헬퍼 함수 호출 (에러 없이 실행)
        cli_print("Basic CLI message")
        cli_print("Styled message", style="bold green")
        cli_print("Message with emoji", emoji="🚀")
        cli_print("Full featured", style="cyan", emoji="💡")
        
        cli_success("Operation completed successfully")
        cli_error("Operation failed")
        cli_warning("Resource usage is high")
        cli_info("System information")
        
        # Then: 에러 없이 모든 함수 실행됨
    
    def test_test_helper_functions(self):
        """케이스 D: 테스트 헬퍼 함수들"""
        # When: 테스트 헬퍼 함수 호출 (에러 없이 실행)
        testing_print("Test message")
        testing_print("Custom emoji test", emoji="🧪")
        
        phase_print("Data Loading Phase")
        phase_print("Model Training Phase", emoji="🤖")
        
        success_print("Test passed successfully")
        testing_info("Test information message")
        
        # Then: 에러 없이 모든 함수 실행됨


class TestConsoleManagerEdgeCases:
    """Console Manager Edge Cases & Performance 테스트"""
    
    def test_unicode_emoji_handling(self):
        """케이스 A: 유니코드/이모지 처리"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: 다양한 유니코드/이모지 출력
        unicode_messages = [
            "한글 메시지 🚀",
            "日本語メッセージ 🎌",
            "Русский текст 🇷🇺", 
            "مرحبا بالعالم 🌍",
            "Mixed language 混合 언어 🔤",
            "Emojis: 🎯🔥💡🚨✅❌⚠️📊🤖🎨",
            "Special chars: ∑∫∆√π∞≈≠≤≥"
        ]
        
        for message in unicode_messages:
            # 에러 없이 유니코드 출력
            manager.log_milestone(message, "info")
            
        # Complex unicode in tables
        unicode_metrics = {
            "정확도": 0.8547,
            "Präzision": 0.8123,
            "再現率": 0.7956,
            "متوسط_الدقة": 0.8038
        }
        
        manager.display_metrics_table(unicode_metrics, title="多言語 메트릭 📊")
        
        # Then: 유니코드 처리 에러 없음
    
    def test_large_data_output_performance(self):
        """케이스 B: 대용량 데이터 출력 성능"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # Large metrics table
        large_metrics = {f"metric_{i}": float(i * 0.01) for i in range(100)}
        
        # When: 대용량 메트릭 테이블 출력
        start_time = time.time()
        manager.display_metrics_table(large_metrics, title="Large Metrics Table")
        end_time = time.time()
        
        # Then: 성능 기준 (1초 이내)
        assert end_time - start_time < 1.0
        
        # Large artifacts list
        large_artifacts = [f"artifact_{i}.pkl" for i in range(50)]
        
        start_time = time.time()
        manager.log_artifacts_progress(large_artifacts)
        end_time = time.time()
        
        # Progress with large list도 적절한 시간 내 완료
        assert end_time - start_time < 3.0  # 50개 아티팩트, 각각 0.1초 시뮬레이션
    
    def test_nested_context_managers(self):
        """케이스 C: Context manager 중첩"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: 중첩된 context manager 사용
        with manager.pipeline_context("Outer Pipeline", "Outer description"):
            assert manager.current_pipeline == "Outer Pipeline"
            
            with manager.progress_tracker("outer_progress", 3, "Outer progress", show_progress=False) as outer_update:
                outer_update(1)
                
                # Inner context (pipeline context는 중첩되지 않지만 progress는 가능)
                with manager.progress_tracker("inner_progress", 2, "Inner progress", show_progress=False) as inner_update:
                    inner_update(1)
                    inner_update(2)
                    
                outer_update(2)
                outer_update(3)
        
        # Then: 모든 context가 올바르게 정리됨
        assert manager.current_pipeline is None
        assert "outer_progress" not in manager.progress_bars
        assert "inner_progress" not in manager.progress_bars
    
    def test_console_mode_switching(self):
        """케이스 D: Console 모드 동적 전환"""
        # Given: 다양한 환경에서 UnifiedConsole 동작
        test_scenarios = [
            ({"CI": "true"}, "plain"),
            ({"GITHUB_ACTIONS": "true"}, "plain"),
            ({"PYTEST_CURRENT_TEST": "test"}, "test"),
            ({}, "rich")  # 기본 환경 (TTY 여부에 따라 rich/plain)
        ]
        
        for env_vars, expected_mode in test_scenarios:
            with patch.dict(os.environ, env_vars, clear=True):
                # When: 각 환경에서 UnifiedConsole 생성
                console = UnifiedConsole()
                
                # Then: 예상된 모드로 설정됨
                if expected_mode == "rich":
                    # TTY 여부에 따라 rich/plain 결정될 수 있음
                    assert console.mode in ["rich", "plain"]
                else:
                    assert console.mode == expected_mode
                
                # 각 모드에서 기본 로깅 동작 확인
                console.info("Mode test message")
                console.component_init("TestComponent")
    
    def test_memory_management_progress_bars(self):
        """케이스 E: Progress bar 메모리 관리"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: 다수의 progress tracker 생성/제거
        task_ids = []
        for i in range(5):
            task_id = f"task_{i}"
            task_ids.append(task_id)
            
            # Progress tracker 생성 후 즉시 완료
            with manager.progress_tracker(task_id, 1, f"Task {i}", show_progress=True) as update:
                update(1)  # 즉시 완료
        
        # Then: 완료된 작업들이 자동으로 정리되어야 함
        # (context manager 종료 시 progress_bars에서 제거됨)
        for task_id in task_ids:
            assert task_id not in manager.progress_bars
        
        # 수동 정리 테스트
        manager.progress_bars["test_task"] = (Mock(), Mock(finished=True))
        manager.cleanup_completed_tasks()
        assert "test_task" not in manager.progress_bars
    
    def test_log_periodic_edge_cases(self):
        """케이스 F: log_periodic Edge Cases"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: Edge cases for periodic logging
        
        # every_n=1 (매 iteration 출력)
        manager.log_periodic("always_log", 5, {"value": 1}, every_n=1)
        
        # every_n이 iteration보다 큰 경우
        manager.log_periodic("rare_log", 2, {"value": 2}, every_n=10)
        
        # iteration=0 (항상 출력)
        manager.log_periodic("first_iteration", 0, {"value": 3}, every_n=100)
        
        # 빈 data
        manager.log_periodic("empty_data", 10, {}, every_n=5)
        
        # 복잡한 data 구조
        complex_data = {
            "nested": {"deep": {"value": 42}},
            "list": [1, 2, 3, 4, 5],
            "mixed": {"str": "text", "num": 123, "bool": True}
        }
        manager.log_periodic("complex_log", 14, complex_data, every_n=5)
        
        # Then: 모든 edge case가 에러 없이 처리됨
    
    def test_error_resilience(self):
        """케이스 G: 에러 복원력 테스트"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: 잠재적 에러 상황들
        
        # None 값들
        manager.log_milestone(None, "info")  # None message
        manager.log_milestone("Valid message", None)  # None level
        
        # 빈 문자열들
        manager.log_milestone("", "info")  # 빈 메시지
        manager.log_phase("", "📝")  # 빈 phase name
        
        # 잘못된 타입들 (graceful handling 예상)
        try:
            manager.display_metrics_table(None, "Test")  # None metrics
        except:
            pass  # 에러가 발생해도 테스트 계속
        
        try:
            manager.display_run_info(None)  # None run_id  
        except:
            pass
        
        # Then: 시스템이 여전히 동작함 (다른 로깅 계속 가능)
        manager.log_milestone("Recovery test", "success")


class TestConsoleManagerIntegration:
    """Console Manager 통합 테스트"""
    
    def test_complete_ml_pipeline_logging_scenario(self):
        """케이스 A: 완전한 ML 파이프라인 로깅 시나리오"""
        # Given: Console manager
        manager = RichConsoleManager()
        
        # When: 전체 ML 파이프라인 시뮬레이션
        with manager.pipeline_context("ML Training Pipeline", "Training RandomForest model"):
            
            # 1. 컴포넌트 초기화
            manager.log_component_init("DataAdapter", "success")
            manager.log_component_init("Preprocessor", "success")
            manager.log_component_init("ModelTrainer", "success")
            
            # 2. 데이터 로딩
            manager.log_phase("Data Loading", "📊")
            manager.log_data_operation("Loading training data", shape=(10000, 50))
            manager.log_data_operation("Loading validation data", shape=(2000, 50))
            
            # 3. 전처리
            manager.log_phase("Data Preprocessing", "🔬")
            with manager.progress_tracker("preprocessing", 5, "Preprocessing steps", show_progress=False) as update:
                manager.log_feature_engineering("Scaling", ["numeric_1", "numeric_2", "numeric_3"], "StandardScaler applied")
                update(1)
                manager.log_feature_engineering("Encoding", ["cat_1", "cat_2"], "One-hot encoding")
                update(2)
                manager.log_processing_step("Feature selection", "Selected top 30 features")
                update(3)
                manager.log_processing_step("Train/test split", "80/20 split")
                update(4)
                update(5)
            
            # 4. 모델 학습
            manager.log_phase("Model Training", "🤖")
            with manager.progress_tracker("training", 10, "Training epochs", show_progress=False) as update:
                for epoch in range(10):
                    manager.log_periodic("training", epoch, 
                                       {"epoch": epoch+1, "loss": 0.5 - epoch*0.03, "accuracy": 0.7 + epoch*0.02}, 
                                       every_n=3)
                    update(epoch + 1)
            
            # 5. 하이퍼파라미터 최적화
            manager.log_phase("Hyperparameter Optimization", "🎯")
            for trial in range(5):
                trial_data = {
                    "trial": trial + 1,
                    "total_trials": 5,
                    "score": 0.85 + trial * 0.01,
                    "best_score": 0.89,
                    "params": {"n_estimators": 100 + trial * 50, "max_depth": 5 + trial}
                }
                manager.log_periodic("optuna_trials", trial, trial_data, every_n=1)
            
            # 6. 모델 평가
            manager.log_phase("Model Evaluation", "📏")
            final_metrics = {
                "accuracy": 0.8945,
                "precision": 0.8823,
                "recall": 0.8756,
                "f1_score": 0.8789,
                "auc_roc": 0.9234,
                "training_time": 145.67,
                "inference_time_ms": 2.34
            }
            manager.display_metrics_table(final_metrics, "Final Model Performance")
            
            # 7. MLflow 실험 추적  
            manager.log_phase("MLflow Experiment Tracking", "📤")
            artifacts = ["model.pkl", "preprocessor.pkl", "metrics.json", "feature_names.txt", "confusion_matrix.png"]
            manager.log_artifacts_progress(artifacts)
            
            manager.display_run_info(
                "abc123def456789",
                model_uri="models:/RandomForestModel/1", 
                tracking_uri="http://localhost:5000/#/experiments/1/runs/abc123def456789"
            )
            
            # 8. 최종 정리
            manager.log_milestone("Pipeline completed successfully", "success")
        
        # Then: 전체 파이프라인이 에러 없이 완료됨
        assert manager.current_pipeline is None
    
    def test_error_scenarios_in_pipeline(self):
        """케이스 B: 파이프라인 내 에러 시나리오"""
        # Given: Console manager
        manager = RichConsoleManager() 
        
        # When: 에러가 포함된 파이프라인 시뮬레이션
        with pytest.raises(RuntimeError):
            with manager.pipeline_context("Error Pipeline", "Testing error handling"):
                
                # 정상 동작
                manager.log_component_init("DataLoader", "success")
                manager.log_data_operation("Data loading", shape=(1000, 10))
                
                # 경고 발생
                manager.log_warning_with_context("Memory usage high", {"usage": "85%"})
                
                # 에러 발생
                manager.log_error_with_context(
                    "Critical error during processing",
                    context={"step": "preprocessing", "error_code": "E001"},
                    suggestion="Check data format and try again"
                )
                
                # 파이프라인 중단하는 예외
                raise RuntimeError("Pipeline failed")
        
        # Then: 예외 발생 후에도 console manager 상태가 정리됨
        assert manager.current_pipeline is None