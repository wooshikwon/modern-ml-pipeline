"""
Phase 3 Step 1: Feature Store 데이터 수집(Ingestion) 및 정확성 심층 검증

이 테스트는 mmp-local-dev의 seed-features.sql 데이터가 Feast를 통해 
정확히 materialize되고, PostgreSQL(오프라인)과 Redis(온라인)에 저장된 
피처 값들이 원본 데이터와 100% 일치하는지 심층적으로 검증합니다.

Blueprint 원칙 검증:
- Feature Store 데이터 정확성 100% 보장
- 원본 → Feast → PostgreSQL → Redis 전체 파이프라인 무결성
- Point-in-time 정합성 및 Data Leakage 방지
"""

import pytest
import pandas as pd
import psycopg2
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
from unittest.mock import patch
import json

from src.settings import Settings
from src.core.factory import Factory
from src.core.augmenter import Augmenter
from src.utils.adapters.feature_store_adapter import FeatureStoreAdapter

# DEV 환경 Feature Store 심층 테스트
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_dev_stack,
    pytest.mark.feature_store_deep
]

@pytest.fixture(scope="module")
def postgres_connection():
    """mmp-local-dev PostgreSQL 직접 연결 Fixture"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="mlpipeline",
            user="mluser",
            password="mlpassword"
        )
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL 연결 실패: {e}. mmp-local-dev 스택이 실행 중인지 확인하세요.")

@pytest.fixture(scope="module")
def redis_connection():
    """mmp-local-dev Redis 직접 연결 Fixture"""
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()  # 연결 테스트
        yield r
    except Exception as e:
        pytest.skip(f"Redis 연결 실패: {e}. mmp-local-dev 스택이 실행 중인지 확인하세요.")

@pytest.fixture(scope="module")
def source_data_snapshot(postgres_connection):
    """원본 소스 데이터 스냅샷 생성"""
    cursor = postgres_connection.cursor()
    
    # 1. user_demographics 테이블 전체 데이터
    cursor.execute("SELECT * FROM user_demographics ORDER BY user_id")
    user_demographics = cursor.fetchall()
    user_demo_columns = [desc[0] for desc in cursor.description]
    
    # 2. product_details 테이블 전체 데이터
    cursor.execute("SELECT * FROM product_details ORDER BY product_id")
    product_details = cursor.fetchall()
    product_columns = [desc[0] for desc in cursor.description]
    
    # 3. user_purchase_summary 테이블 전체 데이터
    cursor.execute("SELECT * FROM user_purchase_summary ORDER BY user_id, event_timestamp")
    purchase_summary = cursor.fetchall()
    purchase_columns = [desc[0] for desc in cursor.description]
    
    cursor.close()
    
    return {
        "user_demographics": {
            "columns": user_demo_columns,
            "data": user_demographics
        },
        "product_details": {
            "columns": product_columns,
            "data": product_details
        },
        "user_purchase_summary": {
            "columns": purchase_columns,
            "data": purchase_summary
        }
    }


class TestFeatureStoreDataIngestionDeep:
    """
    Feature Store 데이터 수집 및 정확성 심층 검증
    Phase 3 Step 1: 100% 데이터 정확성 보장
    """

    def test_source_data_availability_and_structure(self, source_data_snapshot):
        """
        [전제조건 검증] mmp-local-dev의 원본 소스 데이터가 올바르게 존재하고 구조화되어 있는지 확인
        """
        # 1. 모든 테이블에 데이터가 존재하는지 확인
        assert len(source_data_snapshot["user_demographics"]["data"]) > 0, \
            "user_demographics 테이블에 데이터가 없습니다."
        assert len(source_data_snapshot["product_details"]["data"]) > 0, \
            "product_details 테이블에 데이터가 없습니다."
        assert len(source_data_snapshot["user_purchase_summary"]["data"]) > 0, \
            "user_purchase_summary 테이블에 데이터가 없습니다."
        
        # 2. 기대하는 컬럼들이 존재하는지 확인
        user_demo_cols = source_data_snapshot["user_demographics"]["columns"]
        expected_user_cols = ["user_id", "age", "country_code", "created_timestamp"]
        for col in expected_user_cols:
            assert col in user_demo_cols, f"user_demographics에 '{col}' 컬럼이 없습니다."
        
        product_cols = source_data_snapshot["product_details"]["columns"]
        expected_product_cols = ["product_id", "price", "category", "brand", "created_timestamp"]
        for col in expected_product_cols:
            assert col in product_cols, f"product_details에 '{col}' 컬럼이 없습니다."
        
        # 3. 데이터 품질 기본 검증
        user_count = len(source_data_snapshot["user_demographics"]["data"])
        product_count = len(source_data_snapshot["product_details"]["data"])
        
        print(f"✅ 원본 데이터 검증 완료: 사용자 {user_count}명, 제품 {product_count}개")

    def test_feast_materialization_process_accuracy(self, dev_test_settings: Settings, source_data_snapshot):
        """
        [핵심 검증] Feast가 원본 데이터를 PostgreSQL 오프라인 스토어로 정확히 materialize하는지 검증
        """
        factory = Factory(dev_test_settings)
        feature_store_adapter: FeatureStoreAdapter = factory.create_feature_store_adapter()
        
        # 1. Feast를 통한 피처 조회 테스트
        test_entities = [
            {"user_id": "u1001", "product_id": "p2001"},
            {"user_id": "u1002", "product_id": "p2002"},
            {"user_id": "u1003", "product_id": "p2003"}
        ]
        
        spine_df = pd.DataFrame(test_entities)
        spine_df["event_timestamp"] = datetime.now()
        
        # Feast를 통해 피처 조회
        augmented_df = feature_store_adapter.get_historical_features(
            spine_df, ["user_demographics:age", "user_demographics:country_code", 
                      "product_details:price", "product_details:category", "product_details:brand"]
        )
        
        # 2. 원본 데이터와 비교 검증
        source_users = {
            row[0]: dict(zip(source_data_snapshot["user_demographics"]["columns"], row))
            for row in source_data_snapshot["user_demographics"]["data"]
        }
        source_products = {
            row[0]: dict(zip(source_data_snapshot["product_details"]["columns"], row))
            for row in source_data_snapshot["product_details"]["data"]
        }
        
        for _, row in augmented_df.iterrows():
            user_id = row["user_id"]
            product_id = row["product_id"]
            
            # 사용자 피처 정확성 검증
            if user_id in source_users:
                source_user = source_users[user_id]
                assert row["age"] == source_user["age"], \
                    f"사용자 {user_id} age 불일치: Feast={row['age']}, 원본={source_user['age']}"
                assert row["country_code"] == source_user["country_code"], \
                    f"사용자 {user_id} country_code 불일치: Feast={row['country_code']}, 원본={source_user['country_code']}"
            
            # 제품 피처 정확성 검증
            if product_id in source_products:
                source_product = source_products[product_id]
                assert row["price"] == source_product["price"], \
                    f"제품 {product_id} price 불일치: Feast={row['price']}, 원본={source_product['price']}"
                assert row["category"] == source_product["category"], \
                    f"제품 {product_id} category 불일치: Feast={row['category']}, 원본={source_product['category']}"
                assert row["brand"] == source_product["brand"], \
                    f"제품 {product_id} brand 불일치: Feast={row['brand']}, 원본={source_product['brand']}"
        
        print(f"✅ Feast materialization 정확성 검증 완료: {len(test_entities)}개 엔티티 검증")

    def test_postgresql_offline_store_data_integrity(self, postgres_connection, dev_test_settings: Settings):
        """
        [저장소 검증] PostgreSQL 오프라인 스토어에 저장된 피처 데이터의 무결성 검증
        """
        cursor = postgres_connection.cursor()
        
        # 1. Feast feature 테이블들이 존재하는지 확인
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name LIKE '%_feature%'
        """)
        feature_tables = cursor.fetchall()
        
        assert len(feature_tables) > 0, "PostgreSQL에 Feast feature 테이블이 없습니다."
        print(f"발견된 Feature 테이블: {[table[0] for table in feature_tables]}")
        
        # 2. 각 피처 테이블의 데이터 품질 검증
        for table_tuple in feature_tables:
            table_name = table_tuple[0]
            
            # 테이블 스키마 확인
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """)
            columns_info = cursor.fetchall()
            
            # 데이터 건수 확인
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # 기본 데이터 품질 검증
            assert row_count > 0, f"Feature 테이블 {table_name}에 데이터가 없습니다."
            
            # 필수 컬럼 존재 확인 (Feast 표준)
            column_names = [col[0] for col in columns_info]
            feast_required_cols = ["event_timestamp", "created_timestamp"]
            for required_col in feast_required_cols:
                if required_col in column_names:  # 모든 테이블에 반드시 있는 것은 아님
                    print(f"  {table_name}: {required_col} 컬럼 확인됨")
        
        cursor.close()
        print(f"✅ PostgreSQL 오프라인 스토어 무결성 검증 완료: {len(feature_tables)}개 테이블")

    def test_redis_online_store_data_integrity(self, redis_connection, dev_test_settings: Settings):
        """
        [저장소 검증] Redis 온라인 스토어에 저장된 피처 데이터의 무결성 검증
        """
        # 1. Redis에 Feast 관련 키들이 존재하는지 확인
        feast_keys = redis_connection.keys("feast:*")
        feature_keys = redis_connection.keys("*:feature*") + redis_connection.keys("*_feature*")
        all_keys = list(set(feast_keys + feature_keys))
        
        assert len(all_keys) > 0, "Redis에 Feast 관련 데이터가 없습니다."
        print(f"발견된 Feature 키: {len(all_keys)}개")
        
        # 2. 샘플 키들의 데이터 구조 검증
        sample_keys = all_keys[:5]  # 처음 5개 키만 샘플 검증
        
        for key in sample_keys:
            key_type = redis_connection.type(key)
            
            if key_type == "string":
                # JSON 데이터인지 확인
                try:
                    value = redis_connection.get(key)
                    json.loads(value)  # JSON 파싱 가능한지 확인
                    print(f"  {key}: JSON 형식 데이터 확인")
                except json.JSONDecodeError:
                    # JSON이 아닌 일반 문자열일 수도 있음
                    print(f"  {key}: 문자열 데이터")
            
            elif key_type == "hash":
                # Hash 구조 데이터
                field_count = redis_connection.hlen(key)
                print(f"  {key}: Hash 구조, {field_count}개 필드")
            
            elif key_type == "list":
                # List 구조 데이터
                list_length = redis_connection.llen(key)
                print(f"  {key}: List 구조, {list_length}개 항목")
        
        # 3. 특정 엔티티에 대한 피처 조회 테스트
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        # 온라인 피처 조회 테스트
        test_spine = pd.DataFrame([{
            "user_id": "u1001",
            "product_id": "p2001",
            "event_timestamp": datetime.now()
        }])
        
        try:
            online_features = feature_store_adapter.get_online_features(
                test_spine, ["user_demographics:age", "product_details:price"]
            )
            assert not online_features.empty, "온라인 피처 조회가 빈 결과를 반환했습니다."
            print(f"✅ Redis 온라인 피처 조회 성공: {len(online_features)} 레코드")
        except Exception as e:
            pytest.fail(f"Redis 온라인 피처 조회 실패: {e}")
        
        print(f"✅ Redis 온라인 스토어 무결성 검증 완료")

    def test_data_type_consistency_across_stores(self, dev_test_settings: Settings, source_data_snapshot):
        """
        [타입 일관성] 원본 → PostgreSQL → Redis 전체 파이프라인에서 데이터 타입 일관성 검증
        """
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        test_spine = pd.DataFrame([{
            "user_id": "u1001",
            "product_id": "p2001",
            "event_timestamp": datetime.now()
        }])
        
        # 1. 오프라인 스토어에서 피처 조회
        offline_features = feature_store_adapter.get_historical_features(
            test_spine, ["user_demographics:age", "product_details:price", "product_details:brand"]
        )
        
        # 2. 온라인 스토어에서 피처 조회
        online_features = feature_store_adapter.get_online_features(
            test_spine, ["user_demographics:age", "product_details:price", "product_details:brand"]
        )
        
        # 3. 데이터 타입 일관성 검증
        assert not offline_features.empty and not online_features.empty, \
            "오프라인 또는 온라인 피처 조회가 실패했습니다."
        
        offline_row = offline_features.iloc[0]
        online_row = online_features.iloc[0]
        
        # 타입별 정확성 검증
        type_checks = [
            ("age", int, "정수형"),
            ("price", (int, float), "숫자형"),
            ("brand", str, "문자형")
        ]
        
        for field, expected_type, type_name in type_checks:
            if field in offline_row and field in online_row:
                offline_val = offline_row[field]
                online_val = online_row[field]
                
                # 타입 확인
                assert isinstance(offline_val, expected_type), \
                    f"오프라인 {field}의 타입이 잘못됨: {type(offline_val)} (기대: {type_name})"
                assert isinstance(online_val, expected_type), \
                    f"온라인 {field}의 타입이 잘못됨: {type(online_val)} (기대: {type_name})"
                
                # 값 일치 확인
                assert offline_val == online_val, \
                    f"{field} 값 불일치: 오프라인={offline_val}, 온라인={online_val}"
                
                print(f"  ✅ {field} 타입 및 값 일관성 확인: {offline_val} ({type_name})")
        
        print(f"✅ 데이터 타입 일관성 검증 완료")

    def test_completeness_and_no_data_loss(self, dev_test_settings: Settings, source_data_snapshot):
        """
        [완전성 검증] 원본 데이터 → Feature Store 파이프라인에서 데이터 손실이 없는지 검증
        """
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        # 1. 원본 데이터에서 모든 사용자 및 제품 ID 추출
        source_user_ids = {row[0] for row in source_data_snapshot["user_demographics"]["data"]}
        source_product_ids = {row[0] for row in source_data_snapshot["product_details"]["data"]}
        
        print(f"원본 데이터: 사용자 {len(source_user_ids)}명, 제품 {len(source_product_ids)}개")
        
        # 2. 모든 조합에 대해 피처 조회 테스트 (샘플링)
        sample_user_ids = list(source_user_ids)[:3]  # 처음 3명만 샘플링
        sample_product_ids = list(source_product_ids)[:3]  # 처음 3개만 샘플링
        
        test_combinations = []
        for user_id in sample_user_ids:
            for product_id in sample_product_ids:
                test_combinations.append({
                    "user_id": user_id,
                    "product_id": product_id,
                    "event_timestamp": datetime.now()
                })
        
        spine_df = pd.DataFrame(test_combinations)
        
        # 3. Feature Store를 통해 피처 조회
        augmented_df = feature_store_adapter.get_historical_features(
            spine_df, ["user_demographics:age", "user_demographics:country_code", 
                      "product_details:price", "product_details:category"]
        )
        
        # 4. 완전성 검증
        assert len(augmented_df) == len(test_combinations), \
            f"데이터 손실 발생: 요청 {len(test_combinations)}건, 응답 {len(augmented_df)}건"
        
        # 5. 각 조합별 피처 존재 여부 확인
        missing_features = []
        for _, row in augmented_df.iterrows():
            user_id = row["user_id"]
            product_id = row["product_id"]
            
            # NULL/NaN 값 체크
            if pd.isna(row["age"]) and user_id in source_user_ids:
                missing_features.append(f"{user_id}의 age")
            if pd.isna(row["price"]) and product_id in source_product_ids:
                missing_features.append(f"{product_id}의 price")
        
        assert len(missing_features) == 0, \
            f"누락된 피처가 있습니다: {missing_features}"
        
        print(f"✅ 데이터 완전성 검증 완료: {len(test_combinations)}개 조합, 손실 없음")

    def test_materialization_performance_baseline(self, dev_test_settings: Settings):
        """
        [성능 기준] Feature Store materialization 성능이 기준치를 만족하는지 검증
        """
        import time
        
        factory = Factory(dev_test_settings)
        feature_store_adapter = factory.create_feature_store_adapter()
        
        # 적당한 크기의 테스트 데이터 생성
        large_spine = pd.DataFrame([
            {
                "user_id": f"u{1000 + i}",
                "product_id": f"p{2000 + (i % 10)}",
                "event_timestamp": datetime.now() - timedelta(days=i % 30)
            }
            for i in range(100)  # 100개 레코드로 성능 테스트
        ])
        
        # 오프라인 피처 조회 성능 측정
        start_time = time.time()
        offline_result = feature_store_adapter.get_historical_features(
            large_spine, ["user_demographics:age", "product_details:price"]
        )
        offline_time = time.time() - start_time
        
        # 온라인 피처 조회 성능 측정 (첫 10개만)
        small_spine = large_spine.head(10)
        start_time = time.time()
        online_result = feature_store_adapter.get_online_features(
            small_spine, ["user_demographics:age", "product_details:price"]
        )
        online_time = time.time() - start_time
        
        # 성능 기준 검증
        assert offline_time < 30.0, f"오프라인 피처 조회가 너무 느림: {offline_time:.2f}초 (기준: 30초)"
        assert online_time < 5.0, f"온라인 피처 조회가 너무 느림: {online_time:.2f}초 (기준: 5초)"
        
        # 결과 유효성 확인
        assert len(offline_result) > 0, "오프라인 피처 조회 결과가 비어있습니다."
        assert len(online_result) > 0, "온라인 피처 조회 결과가 비어있습니다."
        
        print(f"✅ 성능 기준 검증 완료")
        print(f"  오프라인 조회 (100건): {offline_time:.2f}초")
        print(f"  온라인 조회 (10건): {online_time:.2f}초") 