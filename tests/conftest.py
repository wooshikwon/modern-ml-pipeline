# Enhanced conftest.py - Factory pattern integrated with independent test support
# Global fixture dependencies removed for test independence
# Factory pattern for standardized test data and mocking
# Phase 4 Performance Optimization: Session-scoped fixtures for immutable test data

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from tests.factories.test_data_factory import TestDataFactory
from tests.factories.settings_factory import SettingsFactory
from tests.mocks.component_registry import MockComponentRegistry


@pytest.fixture(scope="session")
def tests_root() -> Path:
    """Fixture to return the root path of the tests directory."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def scenario_path(tests_root: Path):
    """
    Scenario-based test data loader.
    Each test can independently load its required scenario data.
    
    Usage:
        def test_example(scenario_path):
            scenario_dir = scenario_path("basic_classification")
            recipe_path = scenario_dir / "recipe.yaml"
            config_path = scenario_dir / "config.yaml"
            data_path = scenario_dir / "sample_data.csv"
    """
    def _scenario(name: str) -> Path:
        return tests_root / "scenarios" / name
    return _scenario


@pytest.fixture
def fixture_recipes_path(tests_root: Path) -> Path:
    """Fixture to return the path to the test recipes in fixtures."""
    return tests_root / "fixtures" / "recipes"


# Service availability check for integration tests
@pytest.fixture(scope="session")
def check_dev_services():
    """
    Check if development services are available.
    Returns a function to test service availability.
    """
    import socket
    
    def _is_service_up(host: str, port: int, timeout: float = 1.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False
    
    def _check_services(required_services=None):
        if required_services is None:
            required_services = [
                ("localhost", 5432),  # PostgreSQL
                ("localhost", 6379),  # Redis  
                ("localhost", 5002),  # MLflow
            ]
        
        return all(_is_service_up(h, p) for h, p in required_services)
    
    return _check_services


# ========================================
# Factory Pattern Fixtures
# ========================================

@pytest.fixture(scope="session")
def test_factories():
    """테스트 팩토리들 제공 - 중앙화된 접근점"""
    return {
        'data': TestDataFactory,
        'settings': SettingsFactory,
        'mocks': MockComponentRegistry
    }


@pytest.fixture(autouse=True)
def reset_mocks():
    """각 테스트 후 Mock 초기화 - 테스트 독립성 보장"""
    yield
    MockComponentRegistry.reset_all()


@pytest.fixture(scope="session")
def shared_test_data():
    """공유 테스트 데이터 (세션 단위 캐싱) - 성능 최적화"""
    return {
        'classification_large': TestDataFactory.create_classification_data(n_samples=1000),
        'regression_large': TestDataFactory.create_regression_data(n_samples=1000),
        'comprehensive': TestDataFactory.create_comprehensive_training_data(n_samples=500)
    }


@pytest.fixture(scope="session")
def classification_test_data():
    """분류 작업용 표준 테스트 데이터 (50샘플 - 빠른 테스트) - 세션 캐싱"""
    return TestDataFactory.create_classification_data(n_samples=50)


@pytest.fixture(scope="session")
def regression_test_data():
    """회귀 작업용 표준 테스트 데이터 (50샘플 - 빠른 테스트) - 세션 캐싱"""  
    return TestDataFactory.create_regression_data(n_samples=50)


@pytest.fixture
def fast_classification_data(shared_test_data):
    """빠른 분류 데이터 (공유 데이터 슬라이싱) - 성능 최적화"""
    return shared_test_data['classification_large'].head(10).copy()


@pytest.fixture  
def fast_regression_data(shared_test_data):
    """빠른 회귀 데이터 (공유 데이터 슬라이싱) - 성능 최적화"""
    return shared_test_data['regression_large'].head(10).copy()


@pytest.fixture(scope="session")
def classification_settings():
    """분류 작업용 표준 설정 - 세션 캐싱"""
    return SettingsFactory.create_classification_settings()


@pytest.fixture(scope="session")
def regression_settings():
    """회귀 작업용 표준 설정 - 세션 캐싱"""
    return SettingsFactory.create_regression_settings()


@pytest.fixture(scope="session")
def minimal_settings():
    """최소한의 설정 - 빠른 테스트용 - 세션 캐싱"""
    return SettingsFactory.create_minimal_settings()


# ========================================
# Enhanced Markers and Configuration
# ========================================

def pytest_configure(config):
    """Register custom markers."""
    # 기존 마커 유지
    config.addinivalue_line(
        "markers", "requires_services: mark test as requiring external services"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    
    # 새로운 마커 추가
    config.addinivalue_line(
        "markers", "core: mark test as core functionality (highest priority)"
    )
    config.addinivalue_line(
        "markers", "extended: mark test as extended coverage (lower priority)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "blueprint_principle_1: Tests for Blueprint Principle 1 (설정-논리 분리)"
    )
    config.addinivalue_line(
        "markers", "blueprint_principle_2: Tests for Blueprint Principle 2 (환경별 역할 분담)"
    )
    config.addinivalue_line(
        "markers", "blueprint_principle_3: Tests for Blueprint Principle 3 (선언적 파이프라인)"
    )
    config.addinivalue_line(
        "markers", "blueprint_principle_4: Tests for Blueprint Principle 4 (모듈화/확장성)"
    )


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """테스트 환경 전역 설정"""
    # 재현 가능한 랜덤 시드 설정
    np.random.seed(42)
    
    # pandas 설정 최적화
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    
    yield
    
    # 테스트 후 정리
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')