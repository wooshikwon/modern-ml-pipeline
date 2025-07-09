import pandas as pd
import pytest
from src.core.preprocessor import Preprocessor
from config.settings import Settings

def test_preprocessor_initialization(xgboost_settings: Settings):
    """
    Preprocessor가 올바른 설정으로 초기화되는지 테스트합니다.
    """
    preprocessor = Preprocessor(config=xgboost_settings.preprocessor, settings=xgboost_settings)
    assert preprocessor.settings.model.name == "xgboost_x_learner"
    assert "member_id" in preprocessor.exclude_cols
    assert "outcome" in preprocessor.exclude_cols
    assert "grp" in preprocessor.exclude_cols

def test_preprocessor_fit_transform(xgboost_settings: Settings):
    """
    Preprocessor의 fit과 transform 메서드가 정상적으로 동작하는지 테스트합니다.
    """
    # 테스트용 샘플 데이터 생성
    sample_data = {
        'member_id': ['a', 'b', 'c', 'd', 'e'],
        'member_gender': ['M', 'F', 'M', 'F', 'M'],
        'member_age': [25, 30, 25, 40, 35],
        'rsvn_30_count': [1, 5, 2, 8, 3],
        'grp': ['A', 'B', 'A', 'B', 'A'],
        'outcome': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(sample_data)

    preprocessor = Preprocessor(config=xgboost_settings.preprocessor, settings=xgboost_settings)
    
    # fit
    preprocessor.fit(df)
    
    # fit 결과 확인
    assert "member_gender" in preprocessor.categorical_cols_
    assert "member_age" in preprocessor.numerical_cols_
    assert "rsvn_30_count" in preprocessor.numerical_cols_
    assert preprocessor._is_fitted()

    # transform
    transformed_df = preprocessor.transform(df)

    # transform 결과 확인
    assert isinstance(transformed_df, pd.DataFrame)
    assert all(col in transformed_df.columns for col in preprocessor.feature_names_out_)
    assert "member_id" not in transformed_df.columns
    
    # 수치형 데이터가 스케일링 되었는지 확인 (평균이 0에 가까운지)
    assert abs(transformed_df['member_age'].mean()) < 1e-9
