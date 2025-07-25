import pytest
import os
from src.settings import Settings
from src.settings.loaders import load_settings_by_file
from src.utils.system.logger import setup_logging

@pytest.fixture(scope="session")
def local_test_settings() -> Settings:
    """
    LOCAL 환경 테스트를 위한 표준 설정 객체(Settings)를 제공하는 Fixture.
    'local_classification_test.yaml' 레시피를 사용합니다.
    """
    os.environ['APP_ENV'] = 'local'
    settings = load_settings_by_file('local_classification_test.yaml')
    setup_logging(settings)
    return settings

@pytest.fixture(scope="session")
def dev_test_settings() -> Settings:
    """
    DEV 환경 테스트를 위한 표준 설정 객체(Settings)를 제공하는 Fixture.
    'dev_classification_test.yaml' 레시피를 사용합니다.
    """
    os.environ['APP_ENV'] = 'dev'
    settings = load_settings_by_file('dev_classification_test.yaml')
    setup_logging(settings)
    return settings
