"""
Phase 2: Feature Store 완전 테스트

이 테스트는 `modern-ml-pipeline`이 `mmp-local-dev`의 실제 Feature Store와
완벽하게 연동되는지 검증하여, 데이터 흐름의 정합성을 보장합니다.
"""

import pytest
import pandas as pd
from datetime import datetime
from fastapi.testclient import TestClient
import mlflow
import shutil

from src.settings import load_settings_by_file
from src.pipelines.train_pipeline import run_training
from src.pipelines.inference_pipeline import run_batch_inference
from serving.api import app, setup_api_context
from src.core.factory import Factory
from src.core.augmenter import Augmenter
from src.utils.system.logger import setup_logging

# DEV 환경 통합 테스트임을 명시
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_dev_stack
]

@pytest.fixture(scope="module")
def trained_artifact_run_id(dev_test_settings):
    """
    테스트 전에 DEV 환경에서 학습을 실행하여 실제 run_id를 생성하고 반환하는 Fixture.
    테스트 격리를 위해 전용 MLflow 경로를 사용합니다.
    """
    test_tracking_uri = "./test_mlruns"
    mlflow.set_tracking_uri(test_tracking_uri)
    
    print(f"\n[SETUP] 테스트용 모델 학습 시작 (MLflow URI: {test_tracking_uri})...")
    result_artifact = run_training(settings=dev_test_settings)
    assert result_artifact is not None
    print(f"[SETUP] 테스트용 모델 학습 완료. Run ID: {result_artifact.run_id}")
    
    yield result_artifact.run_id
    
    # [TEARDOWN] 테스트 완료 후 임시 MLflow 디렉토리 정리
    print(f"\n[TEARDOWN] 테스트용 MLflow 디렉토리({test_tracking_uri})를 삭제합니다.")
    shutil.rmtree(test_tracking_uri, ignore_errors=True)


class TestFeatureStoreFlow:
    """
    DEV 환경에서 Feature Store의 오프라인/온라인 흐름을 완벽하게 검증하는 통합 테스트.
    """

    def test_offline_store_augmentation(self, dev_test_settings):
        """
        [1단계: 오프라인 저장소 검증]
        학습 파이프라인이 PostgreSQL 오프라인 저장소에서 Point-in-time으로
        피처를 정확하게 증강하는지 검증한다.
        """
        result_artifact = run_training(settings=dev_test_settings)
        assert result_artifact is not None, "학습 파이프라인이 아티팩트를 반환해야 합니다."
        
        augmented_df = result_artifact.get_pandas_dataframe("augmented_data")
        expected_features = [
            "user_total_purchase_amount_7d", "user_total_purchase_amount_30d",
            "product_price", "product_category", "product_brand"
        ]
        
        for feature in expected_features:
            assert feature in augmented_df.columns, f"오프라인 저장소 피처 '{feature}'가 누락되었습니다."
            
        user1001_first_event = augmented_df[augmented_df["user_id"] == "u1001"].sort_values("event_timestamp").iloc[0]
        assert user1001_first_event["user_total_purchase_amount_7d"] == 150.0, \
            "Point-in-time join이 과거 시점의 피처를 정확하게 가져오지 못했습니다."

    def test_online_store_serving(self, trained_artifact_run_id, dev_test_settings):
        """
        [2단계: 온라인 저장소 검증]
        API 서버가 Redis 온라인 저장소에서 실시간 피처를 정확하게 조회하여 서빙하는지 검증한다.
        """
        setup_api_context(run_id=trained_artifact_run_id, settings=dev_test_settings)
        client = TestClient(app)

        entity_key = {"user_id": "u1002", "product_id": "p2002"}
        response = client.post("/predict", json=entity_key)

        assert response.status_code == 200
        prediction_result = response.json()
        assert "prediction" in prediction_result
        assert "input_features" in prediction_result
        
        input_features = prediction_result["input_features"]
        assert input_features["product_price"] == 200.0, "Redis 온라인 스토어에서 'product_price'를 잘못 가져왔습니다."
        assert input_features["product_brand"] == "BrandB", "Redis 온라인 스토어에서 'product_brand'를 잘못 가져왔습니다."

    def test_data_consistency_between_offline_and_online(self, dev_test_settings):
        """
        [3단계: 데이터 일관성 검증]
        동일한 엔티티에 대해 Augmenter를 통해 조회한 오프라인과 온라인 저장소의 피처 값이 일치하는지 검증한다.
        """
        factory = Factory(dev_test_settings)
        augmenter: Augmenter = factory.create_augmenter()

        spine_df = pd.DataFrame([{
            "user_id": "u1003",
            "product_id": "p2003",
            "event_timestamp": datetime.now()
        }])
        
        # 1. 오프라인 저장소에서 피처 조회 (batch 모드)
        offline_augmented_df = augmenter.augment(spine_df, run_mode="batch")

        # 2. 온라인 저장소에서 피처 조회 (serving 모드)
        online_augmented_df = augmenter.augment(spine_df, run_mode="serving")
        
        # 3. 비교 검증
        assert not offline_augmented_df.empty, "오프라인 저장소에서 피처를 가져오지 못했습니다."
        assert not online_augmented_df.empty, "온라인 저장소에서 피처를 가져오지 못했습니다."
        
        offline_price = offline_augmented_df["product_price"].iloc[0]
        online_price = online_augmented_df["product_price"].iloc[0]
        assert offline_price == online_price, \
            f"온라인-오프라인 데이터 불일치 (product_price)! Offline: {offline_price}, Online: {online_price}"
            
        offline_brand = offline_augmented_df["product_brand"].iloc[0]
        online_brand = online_augmented_df["product_brand"].iloc[0]
        assert offline_brand == online_brand, \
            f"온라인-오프라인 데이터 불일치 (product_brand)! Offline: {offline_brand}, Online: {online_brand}"

        print(f"\n✅ [Success] Augmenter 일관성 검증 완료: u1003/p2003 -> price={online_price}, brand='{online_brand}'") 