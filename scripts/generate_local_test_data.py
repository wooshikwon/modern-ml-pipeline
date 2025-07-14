#!/usr/bin/env python3
"""
LOCAL 환경 테스트 데이터 생성 스크립트
Blueprint v17.0: LOCAL 환경의 완전 독립성을 위한 샘플 데이터 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta


def generate_classification_data():
    """분류 문제용 테스트 데이터 생성"""
    np.random.seed(42)
    n_samples = 1000
    
    # 기본 피처들
    data = {
        'user_id': [f'user_{i:04d}' for i in range(n_samples)],
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_level': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples),
        'region': np.random.choice(['north', 'south', 'east', 'west'], n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'num_products': np.random.poisson(2, n_samples),
        'event_timestamp': [
            datetime.now() - timedelta(days=np.random.randint(0, 365))
            for _ in range(n_samples)
        ]
    }
    
    # 타겟 변수 (신용카드 승인 여부)
    # 간단한 로직: 나이, 소득, 신용점수가 높을수록 승인 확률 높음
    approval_prob = (
        (data['age'] - 18) / 52 * 0.3 +
        np.clip((np.array(data['income']) - 30000) / 50000, 0, 1) * 0.4 +
        (np.array(data['credit_score']) - 300) / 550 * 0.3
    )
    data['approved'] = np.random.binomial(1, approval_prob, n_samples)
    
    return pd.DataFrame(data)


def generate_regression_data():
    """회귀 문제용 테스트 데이터 생성"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'property_id': [f'prop_{i:04d}' for i in range(n_samples)],
        'size_sqft': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age_years': np.random.randint(0, 50, n_samples),
        'location_score': np.random.uniform(1, 10, n_samples),
        'school_rating': np.random.uniform(1, 10, n_samples),
        'crime_rate': np.random.uniform(0, 100, n_samples),
        'event_timestamp': [
            datetime.now() - timedelta(days=np.random.randint(0, 365))
            for _ in range(n_samples)
        ]
    }
    
    # 타겟 변수 (주택 가격)
    price = (
        data['size_sqft'] * 150 +
        np.array(data['bedrooms']) * 5000 +
        np.array(data['bathrooms']) * 3000 +
        (50 - np.array(data['age_years'])) * 1000 +
        np.array(data['location_score']) * 10000 +
        np.array(data['school_rating']) * 8000 +
        (100 - np.array(data['crime_rate'])) * 500 +
        np.random.normal(0, 20000, n_samples)  # 노이즈
    )
    data['price'] = np.clip(price, 50000, 1000000)  # 현실적인 범위로 제한
    
    return pd.DataFrame(data)


def generate_causal_data():
    """인과추론/업리프트 문제용 테스트 데이터 생성"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'member_id': [f'member_{i:04d}' for i in range(n_samples)],
        'age': np.random.randint(18, 70, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'lifetime_value': np.random.lognormal(8, 1, n_samples),
        'days_since_last_purchase': np.random.exponential(30, n_samples),
        'avg_order_value': np.random.lognormal(4, 0.5, n_samples),
        'purchase_frequency': np.random.poisson(2, n_samples),
        'email_engagement': np.random.uniform(0, 1, n_samples),
        'event_timestamp': [
            datetime.now() - timedelta(days=np.random.randint(0, 365))
            for _ in range(n_samples)
        ]
    }
    
    # 트리트먼트 할당 (랜덤화)
    data['grp'] = np.random.choice(['control', 'treatment'], n_samples)
    
    # 결과 변수 (구매 여부)
    # 베이스라인 구매 확률
    base_prob = np.clip(
        np.log(data['lifetime_value']) / 15 +
        (1 / (np.array(data['days_since_last_purchase']) + 1)) * 0.3 +
        np.array(data['email_engagement']) * 0.2,
        0.05, 0.95
    )
    
    # 트리트먼트 효과 (일부 고객에게만 효과적)
    treatment_effect = np.where(
        (np.array(data['lifetime_value']) > np.median(data['lifetime_value'])) &
        (np.array(data['email_engagement']) > 0.5),
        0.15, 0.02  # 고가치 + 높은 이메일 참여도 고객에게 더 큰 효과
    )
    
    # 최종 구매 확률
    final_prob = np.where(
        np.array(data['grp']) == 'treatment',
        base_prob + treatment_effect,
        base_prob
    )
    
    # 확률 값 검증 및 정규화
    final_prob = np.clip(final_prob, 0.01, 0.99)  # 0-1 범위로 제한
    final_prob = np.nan_to_num(final_prob, nan=0.5)  # NaN 처리
    
    data['outcome'] = np.random.binomial(1, final_prob, n_samples)
    
    return pd.DataFrame(data)


def save_data():
    """테스트 데이터를 파일로 저장"""
    data_dir = Path("data")
    
    # 분류 데이터
    classification_df = generate_classification_data()
    classification_df.to_parquet(data_dir / "processed" / "classification_test.parquet", index=False)
    classification_df.to_csv(data_dir / "processed" / "classification_test.csv", index=False)
    
    # 회귀 데이터
    regression_df = generate_regression_data()
    regression_df.to_parquet(data_dir / "processed" / "regression_test.parquet", index=False)
    regression_df.to_csv(data_dir / "processed" / "regression_test.csv", index=False)
    
    # 인과추론 데이터
    causal_df = generate_causal_data()
    causal_df.to_parquet(data_dir / "processed" / "causal_test.parquet", index=False)
    causal_df.to_csv(data_dir / "processed" / "causal_test.csv", index=False)
    
    # 메타데이터 생성
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "datasets": {
            "classification": {
                "file": "classification_test.parquet",
                "task_type": "classification",
                "target_col": "approved",
                "n_samples": len(classification_df),
                "description": "신용카드 승인 예측 데이터"
            },
            "regression": {
                "file": "regression_test.parquet", 
                "task_type": "regression",
                "target_col": "price",
                "n_samples": len(regression_df),
                "description": "주택 가격 예측 데이터"
            },
            "causal": {
                "file": "causal_test.parquet",
                "task_type": "causal", 
                "target_col": "outcome",
                "treatment_col": "grp",
                "treatment_value": "treatment",
                "n_samples": len(causal_df),
                "description": "캠페인 업리프트 효과 데이터"
            }
        }
    }
    
    with open(data_dir / "processed" / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ LOCAL 환경 테스트 데이터 생성 완료!")
    print(f"📂 저장 위치: {data_dir.absolute() / 'processed'}")
    print(f"📊 분류 데이터: {len(classification_df):,} 행")
    print(f"📊 회귀 데이터: {len(regression_df):,} 행")  
    print(f"📊 인과추론 데이터: {len(causal_df):,} 행")
    print("")
    print("🚀 테스트 명령어:")
    print("  APP_ENV=local python main.py train --recipe-file local_classification_test")


if __name__ == "__main__":
    save_data() 