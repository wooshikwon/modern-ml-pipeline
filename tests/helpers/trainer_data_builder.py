"""
TrainerDataBuilder - 데이터 핸들러 테스트용 특수 데이터 생성
data_handler.py의 엣지 케이스와 예외 상황을 테스트하기 위한 데이터 생성
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from .dataframe_builder import DataFrameBuilder


class TrainerDataBuilder(DataFrameBuilder):
    """
    Trainer 컴포넌트 테스트에 특화된 데이터 생성기
    DataFrameBuilder를 상속하여 엣지 케이스들을 위한 메서드 추가
    """
    
    @staticmethod
    def build_edge_case_small_dataset(task_choice: str = "classification", n_samples: int = 3) -> pd.DataFrame:
        """
        작은 데이터셋 생성 (stratify 불가능한 경우 테스트용)
        
        Args:
            task_type: classification, regression, clustering, causal
            n_samples: 샘플 수 (기본 3개)
        """
        if task_choice == "classification":
            return DataFrameBuilder.build_classification_data(
                n_samples=n_samples, 
                n_features=2, 
                n_classes=2,
                add_entity_column=True,
                random_state=42
            )
        elif task_choice == "regression":
            return DataFrameBuilder.build_regression_data(
                n_samples=n_samples,
                n_features=2,
                add_entity_column=True, 
                random_state=42
            )
        elif task_choice == "clustering":
            return DataFrameBuilder.build_clustering_data(
                n_samples=n_samples,
                n_features=2,
                n_clusters=2,
                add_entity_column=True,
                random_state=42
            )
        elif task_choice == "causal":
            return DataFrameBuilder.build_causal_data(
                n_samples=n_samples,
                n_features=2,
                add_entity_column=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
    
    @staticmethod 
    def build_edge_case_single_class(n_samples: int = 10) -> pd.DataFrame:
        """
        단일 클래스만 있는 classification 데이터 (stratify 불가능)
        모든 target 값이 0으로 동일
        """
        data = {
            'user_id': list(range(n_samples)),
            'feature_0': [i * 0.1 for i in range(n_samples)],
            'feature_1': [i * 0.2 for i in range(n_samples)],
            'target': [0] * n_samples  # 모든 값이 동일
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def build_edge_case_stratify_impossible() -> pd.DataFrame:
        """
        Stratify 불가능한 케이스: 각 클래스당 1개씩만 존재
        """
        data = {
            'user_id': [1, 2, 3],
            'feature_0': [1.0, 2.0, 3.0], 
            'feature_1': [0.5, 1.0, 1.5],
            'target': [0, 1, 2]  # 각 클래스당 1개씩
        }
        return pd.DataFrame(data)
        
    @staticmethod
    def build_edge_case_missing_columns(missing_column: str = "target") -> pd.DataFrame:
        """
        필수 컬럼이 누락된 데이터
        
        Args:
            missing_column: 누락시킬 컬럼명 (target, treatment 등)
        """
        base_data = DataFrameBuilder.build_classification_data(n_samples=10, random_state=42)
        
        if missing_column == "target":
            del base_data['target']
        elif missing_column == "treatment":
            # causal 데이터로 변경하고 treatment 제거
            base_data = DataFrameBuilder.build_causal_data(n_samples=10, random_state=42)
            del base_data['treatment']
        elif missing_column == "user_id":
            del base_data['user_id']
            
        return base_data
    
    @staticmethod
    def build_edge_case_no_numeric_features(n_samples: int = 10) -> pd.DataFrame:
        """
        숫자형 피처가 없는 데이터 (모든 피처가 문자열)
        """
        data = {
            'user_id': list(range(n_samples)),
            'category_1': [f'cat_{i % 3}' for i in range(n_samples)],
            'category_2': [f'type_{i % 2}' for i in range(n_samples)],
            'text_field': [f'text_{i}' for i in range(n_samples)],
            'target': [i % 2 for i in range(n_samples)]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def build_edge_case_all_features_excluded(n_samples: int = 10) -> pd.DataFrame:
        """
        모든 컬럼이 exclude되어 feature가 남지 않는 케이스
        user_id(entity) + timestamp + target만 있는 데이터
        """
        data = {
            'user_id': list(range(n_samples)),
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
            'target': [i % 2 for i in range(n_samples)]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def build_edge_case_nan_values(
        nan_in_target: bool = False, 
        nan_in_treatment: bool = False,
        nan_in_features: bool = False,
        n_samples: int = 10
    ) -> pd.DataFrame:
        """
        NaN 값이 포함된 데이터
        
        Args:
            nan_in_target: target 컬럼에 NaN 포함
            nan_in_treatment: treatment 컬럼에 NaN 포함 
            nan_in_features: feature 컬럼들에 NaN 포함
        """
        if nan_in_treatment:
            base_data = DataFrameBuilder.build_causal_data(n_samples=n_samples, random_state=42)
            if nan_in_treatment:
                base_data.loc[0, 'treatment'] = np.nan
        else:
            base_data = DataFrameBuilder.build_classification_data(n_samples=n_samples, random_state=42)
            
        if nan_in_target:
            base_data.loc[0, 'target'] = np.nan
            
        if nan_in_features:
            base_data.loc[1, 'feature_0'] = np.nan
            base_data.loc[2, 'feature_1'] = np.nan
            
        return base_data
    
    @staticmethod
    def build_causal_data_with_custom_columns(
        target_column: str = "outcome",
        treatment_column: str = "treatment", 
        entity_columns: Optional[List[str]] = None,
        timestamp_column: Optional[str] = None,
        n_samples: int = 20
    ) -> pd.DataFrame:
        """
        커스텀 컬럼명을 가진 causal 데이터
        data_interface 설정 테스트용
        """
        base_data = DataFrameBuilder.build_causal_data(n_samples=n_samples, random_state=42)
        
        # 컬럼명 변경
        if target_column != "outcome":
            if "outcome" in base_data.columns:
                base_data = base_data.rename(columns={"outcome": target_column})
            elif "target" in base_data.columns:
                base_data = base_data.rename(columns={"target": target_column})
                
        if treatment_column != "treatment":
            base_data = base_data.rename(columns={"treatment": treatment_column})
            
        # Entity 컬럼 추가/변경
        if entity_columns:
            # 기본 user_id 제거
            if 'user_id' in base_data.columns:
                del base_data['user_id']
            # 새 entity 컬럼들 추가
            for i, col_name in enumerate(entity_columns):
                base_data[col_name] = [f"{col_name}_{j % 5}" for j in range(n_samples)]
                
        # Timestamp 컬럼 추가
        if timestamp_column:
            base_data[timestamp_column] = pd.date_range('2024-01-01', periods=n_samples, freq='h')
            
        return base_data
    
    @staticmethod
    def build_mixed_data_types() -> pd.DataFrame:
        """
        다양한 데이터 타입이 섞인 복잡한 데이터
        숫자형/범주형/날짜형/불린형 등이 모두 포함
        """
        n_samples = 15
        data = {
            'user_id': list(range(n_samples)),
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
            # 숫자형 피처들
            'feature_numeric_int': list(range(n_samples)),
            'feature_numeric_float': [i * 0.1 for i in range(n_samples)],
            # 범주형 피처들  
            'feature_category': [f'cat_{i % 3}' for i in range(n_samples)],
            'feature_boolean': [i % 2 == 0 for i in range(n_samples)],
            # 문자열 피처
            'feature_text': [f'text_{i}' for i in range(n_samples)],
            # 타겟
            'target': [i % 2 for i in range(n_samples)]
        }
        return pd.DataFrame(data)
        
    @staticmethod  
    def build_data_for_task_type(
        task_choice: str,
        n_samples: int = 20,
        add_entity_columns: Optional[List[str]] = None,
        add_timestamp: bool = False,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        특정 Task 타입에 맞는 테스트 데이터 생성
        data_interface 설정과 매칭되는 구조
        """
        # 기본 데이터 생성
        if task_choice == "classification":
            base_data = DataFrameBuilder.build_classification_data(
                n_samples=n_samples, 
                add_entity_column=False,
                add_timestamp=add_timestamp,
                random_state=42
            )
        elif task_choice == "regression":
            base_data = DataFrameBuilder.build_regression_data(
                n_samples=n_samples,
                add_entity_column=False, 
                add_timestamp=add_timestamp,
                random_state=42
            )
        elif task_choice == "clustering":
            base_data = DataFrameBuilder.build_clustering_data(
                n_samples=n_samples,
                add_entity_column=False,
                add_timestamp=add_timestamp,
                random_state=42
            )
        elif task_choice == "causal":
            base_data = DataFrameBuilder.build_causal_data(
                n_samples=n_samples,
                add_entity_column=False,
                add_timestamp=add_timestamp, 
                random_state=42
            )
            # causal의 경우 target을 outcome으로 변경
            if 'outcome' not in base_data.columns and 'target' in base_data.columns:
                base_data = base_data.rename(columns={'target': 'outcome'})
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
            
        # Entity 컬럼 추가
        if add_entity_columns:
            for entity_col in add_entity_columns:
                base_data[entity_col] = [f"{entity_col}_{i}" for i in range(n_samples)]
        
        # Feature 컬럼 제한 (지정된 것만 남기기)
        if feature_columns:
            # 타겟/엔티티/타임스탬프 컬럼 보존
            preserve_columns = ['target', 'outcome', 'treatment', 'timestamp']
            if add_entity_columns:
                preserve_columns.extend(add_entity_columns)
            
            # feature_columns + preserve_columns만 남기기
            all_keep_columns = list(set(feature_columns + preserve_columns))
            available_columns = [col for col in all_keep_columns if col in base_data.columns]
            base_data = base_data[available_columns]
            
        return base_data