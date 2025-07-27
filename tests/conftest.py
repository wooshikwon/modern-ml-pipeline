import pytest
import os
from src.settings import Settings
from src.settings.loaders import load_settings_by_file
from src.utils.system.logger import setup_logging

# 🆕 Phase 6: 테스트 환경 자동 감지 시스템
class TestConfig:
    """테스트 환경 기본 설정 클래스"""
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.recipe_file = None
        
    def get_env_name(self) -> str:
        return self.env_name

class LocalTestConfig(TestConfig):
    """LOCAL 환경 테스트 설정: 빠른 실험과 디버깅"""
    def __init__(self):
        super().__init__("local")
        self.recipe_file = "local_classification_test"
        self.mock_data_enabled = True
        self.external_services_required = False
        
class DevTestConfig(TestConfig):
    """DEV 환경 테스트 설정: 완전한 기능 검증"""
    def __init__(self):
        super().__init__("dev")
        self.recipe_file = "dev_classification_test"
        self.mock_data_enabled = False
        self.external_services_required = True
        
class MockTestConfig(TestConfig):
    """Mock 환경 테스트 설정: CI/CD 및 단위 테스트"""
    def __init__(self):
        super().__init__("mock")
        self.recipe_file = "e2e_classification_test"  # Mock 데이터가 포함된 E2E Recipe 사용
        self.mock_data_enabled = True
        self.external_services_required = False

@pytest.fixture(scope="session")  
def test_environment():
    """
    🆕 Phase 6: 테스트 환경 자동 감지 및 설정
    APP_ENV 환경 변수를 기반으로 적절한 TestConfig 반환
    """
    env = os.getenv("APP_ENV", "local")
    
    if env == "local":
        return LocalTestConfig()
    elif env == "dev":
        return DevTestConfig() 
    else:
        return MockTestConfig()

@pytest.fixture(scope="session")
def local_test_settings() -> Settings:
    """
    LOCAL 환경 테스트를 위한 표준 설정 객체(Settings)를 제공하는 Fixture.
    """
    os.environ['APP_ENV'] = 'local'
    # E2E Recipe 사용으로 변경 (Mock 데이터 포함)
    settings = load_settings_by_file('e2e_classification_test')
    setup_logging(settings)
    return settings

@pytest.fixture(scope="session")
def dev_test_settings() -> Settings:
    """
    DEV 환경 테스트를 위한 표준 설정 객체(Settings)를 제공하는 Fixture.
    """
    os.environ['APP_ENV'] = 'dev'
    # E2E Recipe 사용으로 변경
    settings = load_settings_by_file('e2e_classification_test')
    setup_logging(settings)
    return settings

@pytest.fixture(scope="session")
def e2e_test_settings() -> Settings:
    """
    🆕 Phase 6: E2E 통합 테스트를 위한 설정
    Phase 1-5 통합 기능 검증용 Fixture
    """
    # 환경에 따라 자동 설정
    env = os.getenv("APP_ENV", "local")
    os.environ['APP_ENV'] = env
    
    # E2E Recipe 사용 (Phase 1-5 통합 구조)
    settings = load_settings_by_file('e2e_classification_test')
    setup_logging(settings)
    return settings
