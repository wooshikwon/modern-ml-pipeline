import pytest
from src.settings.settings import load_settings, Settings

@pytest.fixture(scope="session")
def xgboost_settings() -> Settings:
    """
    'xgboost_x_learner' 모델에 대한 테스트용 설정 객체를 제공하는 fixture.
    세션 단위로 한 번만 생성됩니다.
    """
    return load_settings("xgboost_x_learner")

@pytest.fixture(scope="session")
def causal_forest_settings() -> Settings:
    """
    'causal_forest' 모델에 대한 테스트용 설정 객체를 제공하는 fixture.
    세션 단위로 한 번만 생성됩니다.
    """
    return load_settings("causal_forest")

# 필요에 따라 다른 모델의 settings fixture를 추가할 수 있습니다.
