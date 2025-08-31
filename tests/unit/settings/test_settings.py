"""
Settings 테스트

설정 로딩, 검증, 환경별 설정 병합 테스트
"""

from src.settings import Settings
from src.settings._recipe_schema import HyperparametersSettings

def test_load_settings_by_file(test_factories):
    """
    SettingsFactory가 현대화된 Recipe 구조를 올바르게 생성하는지 테스트합니다.
    (Factory 패턴으로 파일 의존성 제거 - Phase 2)
    """
    # ENV_NAME 환경변수 설정 (v2.0: env_name property가 이를 읽음)
    import os
    os.environ['ENV_NAME'] = 'local'
    
    # SettingsFactory로 완전한 설정 딕셔너리 생성
    settings_dict = test_factories['settings'].create_classification_settings(
        "local",
        # 기존 테스트가 검증하던 특정 값들로 오버라이드
        **{
            "environment": {},  # v2.0: app_env 필드 제거됨
            "mlflow": {"experiment_name": "local-test-Campaign-Uplift-Modeling"},
            "recipe": {
                "model": {
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "hyperparameters": {"C": 1.0},  # 기존 테스트가 검증하던 값
                    "data_interface": {
                        "task_type": "classification",
                        "target_column": "outcome"  # 기존 테스트가 검증하던 값
                    },
                    "loader": {
                        "entity_schema": {
                            "entity_columns": ["user_id"],
                            "timestamp_column": "event_timestamp"
                        }
                    }
                }
            },
            "hyperparameter_tuning": {"enabled": False}
        }
    )
    
    # 실제 Settings 객체로 변환
    settings = Settings(**settings_dict)

    # 최상위 레벨 검증
    assert isinstance(settings, Settings)
    # v2.0: app_env 필드 제거됨, env_name property 사용
    assert settings.environment.env_name == "local"
    assert "Campaign-Uplift-Modeling" in settings.mlflow.experiment_name
    assert settings.hyperparameter_tuning.enabled is False

    # 레시피 내용 검증
    assert settings.recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"
    assert settings.recipe.model.data_interface.task_type == "classification"
    assert settings.recipe.model.data_interface.target_column == "outcome"
    assert settings.recipe.model.loader.entity_schema.entity_columns == ["user_id"]
    assert settings.recipe.model.loader.entity_schema.timestamp_column == "event_timestamp"
    
    # 하이퍼파라미터 검증 (HyperparametersSettings 스키마 준수)
    assert isinstance(settings.recipe.model.hyperparameters, HyperparametersSettings)
    assert isinstance(settings.recipe.model.hyperparameters.root, dict)
    assert settings.recipe.model.hyperparameters.root['C'] == 1.0 