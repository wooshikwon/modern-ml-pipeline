"""
Settings 테스트

설정 로딩, 검증, 환경별 설정 병합 테스트
"""

import pytest
import os
from src.settings import Settings, load_settings_by_file

class TestSettingsLoading:
    """
    설정 로딩 및 환경별 병합 로직을 검증하는 테스트.
    Blueprint 원칙 1: "레시피는 논리, 설정은 인프라"
    """

    def test_load_local_settings_correctly(self, local_test_settings: Settings):
        """
        LOCAL 환경 설정이 `base.yaml`과 `local.yaml`을 기반으로
        올바르게 로드되는지 검증한다.
        """
        s = local_test_settings
        
        # 1. 환경 식별자 확인
        assert s.environment.app_env == "local"
        
        # 2. `base.yaml`의 기본값이 로드되었는지 확인
        assert "Campaign-Uplift-Modeling" in s.mlflow.experiment_name
        
        # 3. `local.yaml`에서 덮어쓴 값이 적용되었는지 확인
        assert s.hyperparameter_tuning.enabled is False, \
            "LOCAL 환경에서 HPO가 활성화되어 있습니다. local.yaml 설정을 확인하세요."
        
        # 4. `local_classification_test.yaml` 레시피 내용이 병합되었는지 확인
        assert s.recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"  # 🔄 수정: 일관성을 위해 recipe 구조 사용
        assert s.recipe.model.data_interface.task_type == "classification"  # 🔄 수정: task_type은 data_interface에 있음

    def test_load_dev_settings_correctly(self, dev_test_settings: Settings):
        """
        DEV 환경 설정이 `base.yaml`과 `dev.yaml`을 기반으로
        올바르게 로드되는지 검증한다.
        """
        s = dev_test_settings
        
        # 1. 환경 식별자 확인
        assert s.environment.app_env == "dev"
        
        # 2. `base.yaml`의 기본값이 로드되었는지 확인
        assert "Campaign-Uplift-Modeling" in s.mlflow.experiment_name
        
        # 3. `dev.yaml`에서 덮어쓴 값이 적용되었는지 확인
        assert s.hyperparameter_tuning.enabled is True, \
            "DEV 환경에서 HPO가 비활성화되어 있습니다. dev.yaml 설정을 확인하세요."
        assert "Dev" in s.mlflow.experiment_name, \
            "DEV 환경의 MLflow 실험 이름이 올바르지 않습니다."
            
        # 4. `dev_classification_test.yaml` 레시피 내용이 병합되었는지 확인
        assert s.recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"  # 🔄 수정: 일관성을 위해 recipe 구조 사용
        assert s.recipe.model.augmenter.type == "feature_store"

    def test_loading_non_existent_recipe_raises_error(self):
        """
        존재하지 않는 레시피 파일을 로드하려고 할 때 FileNotFoundError가
        발생하는지 검증한다.
        """
        with pytest.raises(FileNotFoundError):
            load_settings_by_file("non_existent_recipe.yaml") 