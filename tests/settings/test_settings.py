"""
Settings 테스트

설정 로딩, 검증, 환경별 설정 병합 테스트
"""

import pytest
import os
from src.settings import Settings, load_settings_by_file

def test_load_settings_by_file(fixture_recipes_path):
    """
    settings.load_settings_by_file이 현대화된 Recipe 구조를 올바르게 로드하는지 테스트합니다.
    (Phase 1, 27개 Recipe 호환성 검증)
    """
    recipe_path = fixture_recipes_path / "local_classification_test.yaml"
    settings = load_settings_by_file(str(recipe_path))

    # 최상위 레벨 검증
    assert isinstance(settings, Settings)
    assert settings.environment.app_env == "local"
    assert "Campaign-Uplift-Modeling" in settings.mlflow.experiment_name
    assert settings.hyperparameter_tuning.enabled is False

    # 레시피 내용 검증
    assert settings.recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"
    assert settings.recipe.model.data_interface.task_type == "classification"
    assert settings.recipe.model.data_interface.target_column == "outcome"
    assert settings.recipe.model.loader.entity_schema.entity_columns == ["user_id"]
    assert settings.recipe.model.loader.entity_schema.timestamp_column == "event_timestamp"
    
    # 하이퍼파라미터 검증 (Dictionary 형태 유지)
    assert isinstance(settings.recipe.model.hyperparameters, dict)
    assert settings.recipe.model.hyperparameters['C'] == 1.0 