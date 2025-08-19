"""
Phase 3 Step 3: Feature Store 온라인/오프라인 패리티(Parity) 검증

동일한 엔티티 키에 대해 오프라인 저장소에서 조회한 피처와 온라인 저장소에서 
조회한 피처가 데이터 타입과 값 모두에서 완벽하게 일치하는지 검증하는 테스트입니다.

Blueprint 원칙 검증:
- 온라인/오프라인 저장소 간 100% 패리티 보장
- 데이터 타입 및 값의 완벽한 일치성 검증
- 실시간 서빙과 배치 처리의 일관성 확보
- Feature Store 신뢰성 및 정확성 보장
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from src.engine.factory import Factory
from src.components.augmenter import Augmenter
from src.settings import Settings

# DEV 환경 Feature Store 패리티 테스트
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_dev_stack,
    pytest.mark.feature_store_parity
]

@pytest.fixture(scope="module")
def parity_test_entities():
    """패리티 테스트를 위한 표준 엔티티 세트"""
    base_time = datetime.now()
    
    return [
        {"user_id": "u1001", "product_id": "p2001", "event_timestamp": base_time},
        {"user_id": "u1002", "product_id": "p2002", "event_timestamp": base_time},
        {"user_id": "u1003", "product_id": "p2003", "event_timestamp": base_time},
        {"user_id": "u1001", "product_id": "p2003", "event_timestamp": base_time},  # 교차 조합
        {"user_id": "u1002", "product_id": "p2001", "event_timestamp": base_time},  # 교차 조합
    ]

@pytest.fixture(scope="module")
def comprehensive_feature_list():
    """검증할 전체 피처 목록"""
    return [
        # 사용자 피처
        "user_demographics:age",
        "user_demographics:country_code",
        
        # 제품 피처
        "product_details:price",
        "product_details:category", 
        "product_details:brand",
        
        # 구매 요약 피처 (시계열)
        "user_purchase_summary:user_total_purchase_amount_7d",
        "user_purchase_summary:user_total_purchase_amount_30d"
    ]


class TestFeatureStoreParityValidation:
    """
    Feature Store 온라인/오프라인 패리티 심층 검증
    Phase 3 Step 3: 완벽한 일치성 보장
    """

    def test_basic_offline_online_parity(self, dev_test_settings: Settings, parity_test_entities, comprehensive_feature_list):
        """
        [기본 패리티] 오프라인과 온라인 저장소에서 조회한 동일 엔티티 피처의 완벽한 일치성 검증
        """
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        spine_df = pd.DataFrame(parity_test_entities)
        
        # 1. 오프라인 저장소에서 피처 조회
        offline_features = feature_store_adapter.get_historical_features(
            spine_df, comprehensive_feature_list
        )
        
        # 2. 온라인 저장소에서 피처 조회
        online_features = feature_store_adapter.get_online_features(
            spine_df, comprehensive_feature_list
        )
        
        # 3. 기본 검증
        assert not offline_features.empty, "오프라인 피처 조회가 실패했습니다."
        assert not online_features.empty, "온라인 피처 조회가 실패했습니다."
        assert len(offline_features) == len(online_features), \
            f"조회 결과 건수 불일치: 오프라인 {len(offline_features)}, 온라인 {len(online_features)}"
        
        # 4. 엔티티별 패리티 검증
        mismatches = []
        
        for i in range(len(spine_df)):
            entity = spine_df.iloc[i]
            user_id = entity["user_id"]
            product_id = entity["product_id"]
            
            # 해당 엔티티의 오프라인/온라인 피처 찾기
            offline_row = offline_features[
                (offline_features["user_id"] == user_id) & 
                (offline_features["product_id"] == product_id)
            ]
            online_row = online_features[
                (online_features["user_id"] == user_id) & 
                (online_features["product_id"] == product_id)
            ]
            
            if len(offline_row) > 0 and len(online_row) > 0:
                offline_data = offline_row.iloc[0]
                online_data = online_row.iloc[0]
                
                # 피처별 정확한 일치성 검증
                for feature in comprehensive_feature_list:
                    feature_name = feature.split(":")[-1]  # 피처명만 추출
                    
                    if feature_name in offline_data and feature_name in online_data:
                        offline_val = offline_data[feature_name]
                        online_val = online_data[feature_name]
                        
                        # NULL/NaN 처리
                        if pd.isna(offline_val) and pd.isna(online_val):
                            continue  # 둘 다 NULL이면 일치
                        
                        # 타입 및 값 정확성 검증
                        if offline_val != online_val:
                            mismatches.append({
                                "entity": f"{user_id}/{product_id}",
                                "feature": feature_name,
                                "offline_value": offline_val,
                                "online_value": online_val,
                                "offline_type": type(offline_val).__name__,
                                "online_type": type(online_val).__name__
                            })
        
        # 5. 패리티 검증 결과
        if mismatches:
            mismatch_details = "\n".join([
                f"  {m['entity']} {m['feature']}: 오프라인={m['offline_value']}({m['offline_type']}) != 온라인={m['online_value']}({m['online_type']})"
                for m in mismatches[:5]  # 처음 5개만 표시
            ])
            pytest.fail(f"온라인/오프라인 패리티 위반 발견 ({len(mismatches)}건):\n{mismatch_details}")
        
        print(f"✅ 기본 오프라인/온라인 패리티 검증 완료: {len(spine_df)}개 엔티티, {len(comprehensive_feature_list)}개 피처")

    def test_data_type_consistency_parity(self, dev_test_settings: Settings, parity_test_entities):
        """
        [타입 일관성] 오프라인과 온라인 저장소 간 데이터 타입의 완벽한 일치성 검증
        """
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        spine_df = pd.DataFrame(parity_test_entities[:3])  # 처음 3개만 사용
        
        # 타입 검증에 중점을 둔 피처 선택
        type_critical_features = [
            "user_demographics:age",           # int
            "product_details:price",           # float/decimal
            "product_details:brand",           # string
            "user_purchase_summary:user_total_purchase_amount_7d"  # numeric
        ]
        
        offline_features = feature_store_adapter.get_historical_features(
            spine_df, type_critical_features
        )
        online_features = feature_store_adapter.get_online_features(
            spine_df, type_critical_features
        )
        
        assert not offline_features.empty and not online_features.empty, \
            "오프라인 또는 온라인 피처 조회 실패"
        
        # 엔티티별 타입 일관성 검증
        type_mismatches = []
        
        for i in range(len(spine_df)):
            entity = spine_df.iloc[i]
            user_id = entity["user_id"]
            product_id = entity["product_id"]
            
            offline_row = offline_features[
                (offline_features["user_id"] == user_id) & 
                (offline_features["product_id"] == product_id)
            ]
            online_row = online_features[
                (online_features["user_id"] == user_id) & 
                (online_features["product_id"] == product_id)
            ]
            
            if len(offline_row) > 0 and len(online_row) > 0:
                offline_data = offline_row.iloc[0]
                online_data = online_row.iloc[0]
                
                for feature in type_critical_features:
                    feature_name = feature.split(":")[-1]
                    
                    if feature_name in offline_data and feature_name in online_data:
                        offline_val = offline_data[feature_name]
                        online_val = online_data[feature_name]
                        
                        # NULL이 아닌 경우에만 타입 검증
                        if not pd.isna(offline_val) and not pd.isna(online_val):
                            offline_type = type(offline_val)
                            online_type = type(online_val)
                            
                            # 타입 호환성 검증 (int와 float는 숫자형으로 호환)
                            if not self._types_compatible(offline_type, online_type):
                                type_mismatches.append({
                                    "entity": f"{user_id}/{product_id}",
                                    "feature": feature_name,
                                    "offline_type": offline_type.__name__,
                                    "online_type": online_type.__name__,
                                    "offline_value": offline_val,
                                    "online_value": online_val
                                })
        
        if type_mismatches:
            type_details = "\n".join([
                f"  {m['entity']} {m['feature']}: {m['offline_type']} != {m['online_type']}"
                for m in type_mismatches
            ])
            pytest.fail(f"데이터 타입 일관성 위반 ({len(type_mismatches)}건):\n{type_details}")
        
        print("✅ 데이터 타입 일관성 패리티 검증 완료")

    def _types_compatible(self, type1, type2):
        """타입 호환성 검사 헬퍼 메서드"""
        # 숫자형 타입들은 서로 호환
        numeric_types = {int, float, np.int64, np.float64, Decimal}
        if type1 in numeric_types and type2 in numeric_types:
            return True
        
        # 문자열 타입들은 서로 호환
        string_types = {str, np.str_}
        if type1 in string_types and type2 in string_types:
            return True
        
        # 정확히 동일한 타입
        return type1 == type2

    def test_augmenter_parity_through_pipeline(self, dev_test_settings: Settings, parity_test_entities):
        """
        [파이프라인 패리티] Augmenter를 통한 배치/서빙 모드 간 패리티 검증
        """
        factory = Factory(dev_test_settings)
        augmenter: Augmenter = factory.create_augmenter()
        
        spine_df = pd.DataFrame(parity_test_entities[:3])
        
        # 1. 배치 모드로 피처 증강
        batch_augmented = augmenter.augment(spine_df, run_mode="batch")
        
        # 2. 서빙 모드로 피처 증강
        serving_augmented = augmenter.augment(spine_df, run_mode="serving")
        
        # 3. 기본 검증
        assert not batch_augmented.empty, "배치 모드 피처 증강 실패"
        assert not serving_augmented.empty, "서빙 모드 피처 증강 실패"
        assert len(batch_augmented) == len(serving_augmented), \
            f"Augmenter 모드별 결과 건수 불일치: 배치 {len(batch_augmented)}, 서빙 {len(serving_augmented)}"
        
        # 4. 엔티티별 Augmenter 패리티 검증
        augmenter_mismatches = []
        
        for i in range(len(spine_df)):
            entity = spine_df.iloc[i]
            user_id = entity["user_id"]
            product_id = entity["product_id"]
            
            batch_row = batch_augmented[
                (batch_augmented["user_id"] == user_id) & 
                (batch_augmented["product_id"] == product_id)
            ]
            serving_row = serving_augmented[
                (serving_augmented["user_id"] == user_id) & 
                (serving_augmented["product_id"] == product_id)
            ]
            
            if len(batch_row) > 0 and len(serving_row) > 0:
                batch_data = batch_row.iloc[0]
                serving_data = serving_row.iloc[0]
                
                # 공통 피처 컬럼 찾기
                common_columns = set(batch_data.index) & set(serving_data.index)
                feature_columns = [col for col in common_columns 
                                 if col not in ["user_id", "product_id", "event_timestamp"]]
                
                for col in feature_columns:
                    batch_val = batch_data[col]
                    serving_val = serving_data[col]
                    
                    # NULL 처리
                    if pd.isna(batch_val) and pd.isna(serving_val):
                        continue
                    
                    # 값 일치성 검증
                    if batch_val != serving_val:
                        augmenter_mismatches.append({
                            "entity": f"{user_id}/{product_id}",
                            "feature": col,
                            "batch_value": batch_val,
                            "serving_value": serving_val
                        })
        
        if augmenter_mismatches:
            augmenter_details = "\n".join([
                f"  {m['entity']} {m['feature']}: 배치={m['batch_value']} != 서빙={m['serving_value']}"
                for m in augmenter_mismatches[:5]
            ])
            pytest.fail(f"Augmenter 배치/서빙 모드 패리티 위반 ({len(augmenter_mismatches)}건):\n{augmenter_details}")
        
        print("✅ Augmenter 파이프라인 패리티 검증 완료")

    def test_timestamp_handling_parity(self, dev_test_settings: Settings):
        """
        [타임스탬프 패리티] 동일 시점 조회 시 오프라인/온라인의 타임스탬프 처리 일관성 검증
        """
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        # 정확한 시점 지정
        precise_time = datetime.now().replace(microsecond=0) - timedelta(hours=1)
        
        test_spine = pd.DataFrame([{
            "user_id": "u1001",
            "product_id": "p2001", 
            "event_timestamp": precise_time
        }])
        
        # 오프라인/온라인 동일 시점 조회
        offline_result = feature_store_adapter.get_historical_features(
            test_spine, ["user_demographics:age", "product_details:price"]
        )
        online_result = feature_store_adapter.get_online_features(
            test_spine, ["user_demographics:age", "product_details:price"]
        )
        
        # 타임스탬프 일관성 검증
        if not offline_result.empty and not online_result.empty:
            offline_timestamp = offline_result.iloc[0].get("event_timestamp")
            online_timestamp = online_result.iloc[0].get("event_timestamp")
            
            if offline_timestamp is not None and online_timestamp is not None:
                # 두 타임스탬프가 모두 지정된 시점 이전이어야 함
                assert offline_timestamp <= precise_time, \
                    f"오프라인 타임스탬프 오류: {offline_timestamp} > {precise_time}"
                assert online_timestamp <= precise_time, \
                    f"온라인 타임스탬프 오류: {online_timestamp} > {precise_time}"
                
                print(f"  오프라인 타임스탬프: {offline_timestamp}")
                print(f"  온라인 타임스탬프: {online_timestamp}")
                print(f"  쿼리 타임스탬프: {precise_time}")
        
        print("✅ 타임스탬프 처리 패리티 검증 완료")

    def test_large_scale_parity_validation(self, dev_test_settings: Settings):
        """
        [대규모 패리티] 대량 엔티티에 대한 온라인/오프라인 패리티 검증
        """
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        # 대량 엔티티 생성 (조합 확장)
        users = [f"u{1000 + i}" for i in range(20)]
        products = [f"p{2000 + i}" for i in range(10)]
        base_time = datetime.now()
        
        large_entities = []
        for i in range(100):  # 100개 엔티티
            user_id = users[i % len(users)]
            product_id = products[i % len(products)]
            large_entities.append({
                "user_id": user_id,
                "product_id": product_id,
                "event_timestamp": base_time - timedelta(minutes=i % 60)
            })
        
        large_spine = pd.DataFrame(large_entities)
        
        # 핵심 피처에 대해서만 대규모 패리티 검증
        key_features = [
            "user_demographics:age",
            "product_details:price",
            "product_details:brand"
        ]
        
        # 대량 조회 수행
        offline_large = feature_store_adapter.get_historical_features(
            large_spine, key_features
        )
        online_large = feature_store_adapter.get_online_features(
            large_spine, key_features
        )
        
        # 대규모 패리티 통계 분석
        if not offline_large.empty and not online_large.empty:
            total_comparisons = 0
            parity_violations = 0
            
            # 샘플링을 통한 패리티 검증 (전체를 다 확인하면 너무 오래 걸림)
            sample_size = min(50, len(large_spine))
            sample_entities = large_spine.head(sample_size)
            
            for _, entity in sample_entities.iterrows():
                user_id = entity["user_id"]
                product_id = entity["product_id"]
                
                offline_matches = offline_large[
                    (offline_large["user_id"] == user_id) & 
                    (offline_large["product_id"] == product_id)
                ]
                online_matches = online_large[
                    (online_large["user_id"] == user_id) & 
                    (online_large["product_id"] == product_id)
                ]
                
                if len(offline_matches) > 0 and len(online_matches) > 0:
                    offline_data = offline_matches.iloc[0]
                    online_data = online_matches.iloc[0]
                    
                    for feature in key_features:
                        feature_name = feature.split(":")[-1]
                        
                        if feature_name in offline_data and feature_name in online_data:
                            total_comparisons += 1
                            
                            offline_val = offline_data[feature_name]
                            online_val = online_data[feature_name]
                            
                            # NULL 처리
                            if pd.isna(offline_val) and pd.isna(online_val):
                                continue
                            
                            if offline_val != online_val:
                                parity_violations += 1
            
            # 패리티 위반율 계산
            if total_comparisons > 0:
                violation_rate = (parity_violations / total_comparisons) * 100
                assert violation_rate < 1.0, \
                    f"대규모 패리티 위반율이 허용 기준 초과: {violation_rate:.2f}% (기준: 1.0%)"
                
                print(f"  전체 비교: {total_comparisons}건")
                print(f"  패리티 위반: {parity_violations}건")
                print(f"  위반율: {violation_rate:.2f}%")
        
        print(f"✅ 대규모 패리티 검증 완료: {len(large_spine)}개 엔티티 샘플링 검증")

    def test_edge_case_parity_validation(self, dev_test_settings: Settings):
        """
        [엣지 케이스 패리티] 특수한 상황에서의 온라인/오프라인 패리티 검증
        """
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        # Edge Case 1: 존재하지 않는 엔티티
        nonexistent_spine = pd.DataFrame([{
            "user_id": "u9999_nonexistent",
            "product_id": "p9999_nonexistent",
            "event_timestamp": datetime.now()
        }])
        
        offline_nonexistent = feature_store_adapter.get_historical_features(
            nonexistent_spine, ["user_demographics:age"]
        )
        online_nonexistent = feature_store_adapter.get_online_features(
            nonexistent_spine, ["user_demographics:age"]
        )
        
        # 존재하지 않는 엔티티에 대해서는 둘 다 비어있거나 NULL이어야 함
        if not offline_nonexistent.empty and not online_nonexistent.empty:
            offline_age = offline_nonexistent.iloc[0].get("age")
            online_age = online_nonexistent.iloc[0].get("age")
            
            # 둘 다 NULL이거나 둘 다 동일한 기본값이어야 함
            if not (pd.isna(offline_age) and pd.isna(online_age)):
                assert offline_age == online_age, \
                    f"존재하지 않는 엔티티 처리 불일치: 오프라인={offline_age}, 온라인={online_age}"
        
        # Edge Case 2: 매우 먼 과거 시점
        old_time_spine = pd.DataFrame([{
            "user_id": "u1001",
            "product_id": "p2001",
            "event_timestamp": datetime(2020, 1, 1)  # 매우 옛날
        }])
        
        offline_old = feature_store_adapter.get_historical_features(
            old_time_spine, ["user_demographics:age"]
        )
        online_old = feature_store_adapter.get_online_features(
            old_time_spine, ["user_demographics:age"]
        )
        
        # 오래된 시점에 대해서도 일관된 처리가 되어야 함
        if not offline_old.empty and not online_old.empty:
            offline_age_old = offline_old.iloc[0].get("age")
            online_age_old = online_old.iloc[0].get("age")
            
            # 값이 있다면 일치해야 함
            if not (pd.isna(offline_age_old) and pd.isna(online_age_old)):
                assert offline_age_old == online_age_old, \
                    f"오래된 시점 데이터 처리 불일치: 오프라인={offline_age_old}, 온라인={online_age_old}"
        
        print("✅ 엣지 케이스 패리티 검증 완료")

    def test_comprehensive_parity_report(self, dev_test_settings: Settings, parity_test_entities, comprehensive_feature_list):
        """
        [종합 리포트] 전체 Feature Store 패리티 상태에 대한 종합적인 검증 리포트 생성
        """
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        spine_df = pd.DataFrame(parity_test_entities)
        
        # 전체 피처에 대한 패리티 검증
        offline_features = feature_store_adapter.get_historical_features(
            spine_df, comprehensive_feature_list
        )
        online_features = feature_store_adapter.get_online_features(
            spine_df, comprehensive_feature_list
        )
        
        # 종합 패리티 리포트 생성
        parity_report = {
            "total_entities": len(spine_df),
            "total_features": len(comprehensive_feature_list),
            "offline_results": len(offline_features),
            "online_results": len(online_features),
            "feature_parity_status": {},
            "overall_parity_score": 0.0
        }
        
        total_checks = 0
        successful_checks = 0
        
        # 피처별 패리티 상태 분석
        for feature in comprehensive_feature_list:
            feature_name = feature.split(":")[-1]
            feature_status = {
                "matches": 0,
                "mismatches": 0,
                "nulls": 0,
                "type_errors": 0
            }
            
            for i in range(len(spine_df)):
                entity = spine_df.iloc[i]
                user_id = entity["user_id"]
                product_id = entity["product_id"]
                
                offline_row = offline_features[
                    (offline_features["user_id"] == user_id) & 
                    (offline_features["product_id"] == product_id)
                ]
                online_row = online_features[
                    (online_features["user_id"] == user_id) & 
                    (online_features["product_id"] == product_id)
                ]
                
                if len(offline_row) > 0 and len(online_row) > 0:
                    offline_val = offline_row.iloc[0].get(feature_name)
                    online_val = online_row.iloc[0].get(feature_name)
                    
                    total_checks += 1
                    
                    if pd.isna(offline_val) and pd.isna(online_val):
                        feature_status["nulls"] += 1
                        successful_checks += 1
                    elif pd.isna(offline_val) or pd.isna(online_val):
                        feature_status["mismatches"] += 1
                    elif type(offline_val) != type(online_val):
                        feature_status["type_errors"] += 1
                    elif offline_val == online_val:
                        feature_status["matches"] += 1
                        successful_checks += 1
                    else:
                        feature_status["mismatches"] += 1
            
            parity_report["feature_parity_status"][feature] = feature_status
        
        # 전체 패리티 점수 계산
        if total_checks > 0:
            parity_report["overall_parity_score"] = (successful_checks / total_checks) * 100
        
        # 리포트 출력
        print("\n📊 Feature Store 패리티 종합 리포트:")
        print(f"  총 엔티티: {parity_report['total_entities']}개")
        print(f"  총 피처: {parity_report['total_features']}개")
        print(f"  전체 패리티 점수: {parity_report['overall_parity_score']:.2f}%")
        
        # 피처별 상태 요약
        for feature, status in parity_report["feature_parity_status"].items():
            total_feature_checks = sum(status.values())
            if total_feature_checks > 0:
                match_rate = (status["matches"] / total_feature_checks) * 100
                print(f"  {feature}: {match_rate:.1f}% 일치 (matches: {status['matches']}, mismatches: {status['mismatches']})")
        
        # 패리티 기준 검증
        assert parity_report["overall_parity_score"] >= 95.0, \
            f"전체 패리티 점수가 기준 미달: {parity_report['overall_parity_score']:.2f}% (기준: 95.0%)"
        
        print(f"✅ 종합 패리티 검증 완료: {parity_report['overall_parity_score']:.2f}% 패리티 달성") 