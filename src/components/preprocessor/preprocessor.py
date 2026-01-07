from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.components.preprocessor.base import BasePreprocessor
from src.utils.core.logger import log_prep, logger

from .registry import PreprocessorStepRegistry

if TYPE_CHECKING:
    from src.settings import Settings


class Preprocessor(BasePreprocessor):
    """
    Recipe에 정의된 여러 전처리 단계를 동적으로 조립하고 실행하는
    Pipeline Builder 클래스입니다.

    Global vs Targeted 전처리 정책:
    - Global: 모든 적합한 컬럼에 자동 적용 (컬럼명 보존)
    - Targeted: 특정 컬럼에 적용 (컬럼명 변경 가능)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.config = settings.recipe.preprocessor  # Recipe 루트의 preprocessor 참조

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Preprocessor":
        self._fitted_transformers = []
        self._columns_to_delete = set()  # 지연 삭제할 원본 컬럼들 추적

        if self.config and self.config.steps:
            current_data = X.copy()

            # 각 step을 순차적으로 처리 (ColumnTransformer/Pipeline 없이 직접 실행)
            for step in self.config.steps:
                # 파라미터 추출 (type과 columns 제외)
                step_params = step.model_dump(exclude={"type", "columns"})
                step_params = {k: v for k, v in step_params.items() if v is not None}

                # transformer 생성
                transformer = PreprocessorStepRegistry.create(step.type, **step_params)

                # Global vs Targeted 처리 분기
                application_type = transformer.get_application_type()

                if application_type == "global":
                    # Global 타입: 적용 가능한 모든 컬럼에 자동 적용
                    target_columns = transformer.get_applicable_columns(current_data)
                elif step.columns:
                    # Targeted 타입 + 컬럼 지정: 지정된 컬럼 찾기
                    target_columns = self._find_matching_columns(step.columns, current_data.columns)
                else:
                    # Targeted 타입 + 컬럼 미지정: get_applicable_columns로 자동 탐지 시도
                    if hasattr(transformer, "get_applicable_columns"):
                        target_columns = transformer.get_applicable_columns(current_data)
                        if not target_columns:
                            continue
                    else:
                        continue

                # 실제로 존재하는 컬럼만 필터링 (graceful error handling)
                existing_columns = []
                if target_columns:
                    existing_columns = [
                        col for col in target_columns if col in current_data.columns
                    ]
                else:
                    # target_columns가 None인 경우 건너뜀
                    continue

                if not existing_columns:
                    continue

                # 존재하는 컬럼만으로 데이터 추출
                target_data = current_data[existing_columns]

                # transformer 학습 및 변환
                transformed_data = transformer.fit_transform(target_data, y)

                # 결과를 현재 데이터에 병합
                if transformer.preserves_column_names():
                    # 컬럼명이 보존되는 경우 (Scaler 등): 기존 컬럼 업데이트
                    # 메모리 조각화 방지를 위해 한꺼번에 업데이트하는 방식 권장되나,
                    # 순차 적용을 위해 할당 후 copy()로 조각화 해결
                    for col in transformed_data.columns:
                        current_data[col] = transformed_data[col]
                    current_data = current_data.copy() 
                else:
                    # 컬럼명이 변경되는 경우 (Encoder 등):
                    if transformer.get_application_type() == "targeted":
                        # 원본 컬럼 제거 후 새로운 컬럼 병합
                        remaining_columns = [c for c in current_data.columns if c not in existing_columns]
                        current_data = pd.concat([current_data[remaining_columns], transformed_data], axis=1)
                    else:
                        # Global 변환의 경우 (새 컬럼 생성 시)
                        current_data = pd.concat([current_data, transformed_data], axis=1)

                self._fitted_transformers.append(
                    {
                        "transformer": transformer,
                        "target_columns": existing_columns,
                        "step_type": step.type,
                    }
                )

            # 모든 전처리 완료 후: 지연 삭제할 원본 컬럼들 일괄 제거
            if self._columns_to_delete:
                columns_to_remove = [
                    col for col in self._columns_to_delete if col in current_data.columns
                ]
                if columns_to_remove:
                    current_data = current_data.drop(columns=columns_to_remove)

            # 최종 데이터 저장 (transform 시 사용)
            self._final_fit_data = current_data

        log_prep(
            f"파이프라인 학습 완료 - {X.shape} → {self._final_fit_data.shape if hasattr(self, '_final_fit_data') else X.shape}"
        )
        return self

    def get_output_columns(self) -> list[str]:
        """
        전처리 후 최종 출력 컬럼 목록 반환.
        fit() 호출 후에만 유효한 값을 반환합니다.
        """
        if hasattr(self, "_final_fit_data"):
            return list(self._final_fit_data.columns)
        return []

    def transform(self, X: pd.DataFrame, dataset_name: str = "") -> pd.DataFrame:
        if not hasattr(self, "_fitted_transformers"):
            raise RuntimeError("Preprocessor가 아직 학습되지 않았습니다. 'fit'을 먼저 호출하세요.")

        current_data = X.copy()
        dataset_label = f"[{dataset_name}] " if dataset_name else ""

        # 각 단계를 순차적으로 적용
        for step_info in self._fitted_transformers:
            transformer = step_info["transformer"]
            original_target_columns = step_info["target_columns"]

            # Global vs Targeted 처리 분기 (transform 시에도 동일 로직)
            if transformer.get_application_type() == "global":
                # Global 타입: 다시 적용 가능한 컬럼 확인
                target_columns = transformer.get_applicable_columns(current_data)
            else:
                # Targeted 타입: 매핑된 컬럼 재확인
                target_columns = self._find_matching_columns(
                    original_target_columns, current_data.columns
                )

            # 대상 컬럼이 존재하지 않는 경우 기본값으로 생성 (Targeted 타입만)
            if transformer.get_application_type() == "targeted":
                for col in target_columns:
                    if col not in current_data.columns:
                        logger.warning(f"[PREP] 컬럼 '{col}' 누락 - 기본값 0으로 생성")
                        current_data[col] = 0

            if not target_columns:
                continue

            # 대상 컬럼 데이터 추출
            target_data = current_data[target_columns]

            # transformer 적용
            transformed_data = transformer.transform(target_data)

            # 결과를 현재 데이터에 병합
            if transformer.preserves_column_names():
                # 컬럼명이 보존되는 경우: 기존 컬럼 업데이트 및 조각화 방지
                for col in transformed_data.columns:
                    current_data[col] = transformed_data[col]
                current_data = current_data.copy()
            else:
                # 컬럼명이 변경되는 경우: 일괄 병합
                if transformer.get_application_type() == "targeted":
                    # 원본 컬럼 제거 후 새로운 컬럼 병합
                    remaining_columns = [c for c in current_data.columns if c not in target_columns]
                    current_data = pd.concat([current_data[remaining_columns], transformed_data], axis=1)
                else:
                    # Global 변환의 경우 (새 컬럼 생성 시)
                    current_data = pd.concat([current_data, transformed_data], axis=1)

        # 모든 변환 완료 후: 지연 삭제할 원본 컬럼들 일괄 제거
        if hasattr(self, "_columns_to_delete") and self._columns_to_delete:
            columns_to_remove = [
                col for col in self._columns_to_delete if col in current_data.columns
            ]
            if columns_to_remove:
                current_data = current_data.drop(columns=columns_to_remove)

        log_prep(f"{dataset_label}전처리 완료 - {X.shape} → {current_data.shape}")

        return current_data

    def save(self, file_path: str):
        pass

    @classmethod
    def load(cls, file_path: str) -> "Preprocessor":
        pass

    def _find_matching_columns(self, target_columns, available_columns):
        """
        단순화된 컬럼 매핑: DataFrame-First 정책 하에서 각 전처리기가 컬럼명을 직접 관리

        매칭 우선순위:
        1. 정확한 일치 (Exact match) - 대부분의 경우
        2. 컬럼 존재하지 않으면 그대로 전달 (전처리기에서 처리)
        """
        if not target_columns:
            return target_columns

        return list(target_columns)
