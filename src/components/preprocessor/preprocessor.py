from __future__ import annotations
import pandas as pd
from typing import Optional, TYPE_CHECKING

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.interface import BasePreprocessor
from src.utils.core.console import Console
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
        self.console = Console(settings)
        self.config = settings.recipe.preprocessor  # Recipe 루트의 preprocessor 참조
        self.pipeline: Optional[Pipeline] = None


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Preprocessor':
        self.console.info("DataFrame-First 순차적 전처리 파이프라인 빌드를 시작합니다...",
                         rich_message="🔧 Building preprocessing pipeline")
        self.console.data_operation("Initial data loaded", X.shape)
        
        self._fitted_transformers = []
        self._columns_to_delete = set()  # 지연 삭제할 원본 컬럼들 추적
        
        if self.config and self.config.steps:
            current_data = X.copy()
            
            # 각 step을 순차적으로 처리 (ColumnTransformer/Pipeline 없이 직접 실행)
            for i, step in enumerate(self.config.steps):
                self.console.info(f"Step {i+1}: {step.type}, 대상 컬럼: {step.columns}",
                                rich_message=f"🔍 Step {i+1}: [cyan]{step.type}[/cyan] on [dim]{step.columns}[/dim]")
                
                # 파라미터 추출 (type과 columns 제외)
                step_params = step.model_dump(exclude={'type', 'columns'})
                step_params = {k: v for k, v in step_params.items() if v is not None}
                
                # transformer 생성
                transformer = PreprocessorStepRegistry.create(step.type, **step_params)
                
                # Global vs Targeted 처리 분기
                if transformer.get_application_type() == 'global':
                    # Global 타입: 적용 가능한 모든 컬럼에 자동 적용
                    target_columns = transformer.get_applicable_columns(current_data)
                    self.console.info(f"Global 적용 - 대상 컬럼: {target_columns}",
                                    rich_message=f"   🌐 Global application: [green]{len(target_columns)}[/green] columns")
                else:
                    # Targeted 타입: 지정된 컬럼 찾기
                    target_columns = self._find_matching_columns(step.columns, current_data.columns)
                    self.console.info(f"Targeted 적용 - 매핑된 컬럼: {step.columns} -> {target_columns}",
                                    rich_message=f"   🎯 Targeted mapping: [yellow]{step.columns}[/yellow] → [green]{target_columns}[/green]")
                
                
                # 실제로 존재하는 컬럼만 필터링 (graceful error handling)
                existing_columns = [col for col in target_columns if col in current_data.columns]
                
                if not existing_columns:
                    self.console.warning(f"Step {i+1} ({step.type}): 적용할 컬럼이 없습니다. 지정된 컬럼: {target_columns}",
                                       rich_message=f"   ⚠️  No applicable columns for [red]{step.type}[/red] - specified: [yellow]{target_columns}[/yellow]")
                    continue
                
                # 존재하지 않는 컬럼들 경고 로그
                missing_columns = [col for col in target_columns if col not in current_data.columns]
                if missing_columns:
                    self.console.warning(f"Step {i+1} ({step.type}): 존재하지 않는 컬럼들이 스킵됩니다: {missing_columns}",
                                       rich_message=f"   🚫 Missing columns skipped: [red]{missing_columns}[/red]")
                
                # 존재하는 컬럼만으로 데이터 추출
                target_data = current_data[existing_columns]
                
                # transformer 학습 및 변환
                transformed_data = transformer.fit_transform(target_data, y)
                
                # 결과를 현재 데이터에 병합
                if transformer.preserves_column_names():
                    # 컬럼명이 보존되는 경우: 기존 컬럼 업데이트
                    for col in transformed_data.columns:
                        current_data[col] = transformed_data[col]
                else:
                    # 컬럼명이 변경되는 경우: 지연 삭제를 위해 원본 컬럼 추적
                    if transformer.get_application_type() == 'targeted':
                        # Targeted 전처리기의 실제 처리된 컬럼은 지연 삭제 목록에 추가
                        self._columns_to_delete.update(existing_columns)
                        self.console.info(f"지연 삭제 목록에 추가: {existing_columns}",
                                        rich_message=f"   🗑️  Marked for delayed deletion: [dim]{existing_columns}[/dim]")
                    
                    # 새로운 컬럼들을 현재 데이터에 추가 (원본 컬럼은 유지)
                    for col in transformed_data.columns:
                        current_data[col] = transformed_data[col]
                
                self._fitted_transformers.append({
                    'transformer': transformer,
                    'target_columns': existing_columns,  # 실제 존재하는 컬럼만 저장
                    'step_type': step.type
                })
                
                self.console.data_operation(f"Step {i+1} transformation completed", 
                                           current_data.shape, 
                                           f"Columns: {len(current_data.columns)}")
        
            # 모든 전처리 완료 후: 지연 삭제할 원본 컬럼들 일괄 제거
            if self._columns_to_delete:
                # 실제로 존재하는 컬럼만 삭제 (이미 다른 단계에서 제거된 컬럼 제외)
                columns_to_remove = [col for col in self._columns_to_delete if col in current_data.columns]
                if columns_to_remove:
                    self.console.info(f"지연 삭제 실행: {columns_to_remove}",
                                    rich_message=f"🗑️  Executing delayed column deletion: [red]{len(columns_to_remove)}[/red] columns")
                    current_data = current_data.drop(columns=columns_to_remove)
                    self.console.data_operation("Final preprocessing result", 
                                               current_data.shape,
                                               f"Final columns: {len(current_data.columns)}")
            
            # 최종 데이터 저장 (transform 시 사용)
            self._final_fit_data = current_data
        
        # 단순한 identity pipeline으로 설정 (실제 변환은 transform에서 수행)
        identity = FunctionTransformer(validate=False)
        self.pipeline = Pipeline([('identity', identity)])
        
        self.console.info("DataFrame-First 전처리 파이프라인 빌드 및 학습 완료.",
                         rich_message="✅ Preprocessing pipeline built and fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, '_fitted_transformers'):
            raise RuntimeError("Preprocessor가 아직 학습되지 않았습니다. 'fit'을 먼저 호출하세요.")
        
        current_data = X.copy()
        self.console.info(f"Transform 시작 - 입력 데이터 shape: {current_data.shape}, 컬럼: {list(current_data.columns)}",
                         rich_message="🔄 Starting data transformation")
        self.console.data_operation("Transform input", current_data.shape)
        
        # 각 단계를 순차적으로 적용
        for i, step_info in enumerate(self._fitted_transformers):
            step_type = step_info['step_type']
            self.console.info(f"=== Step {i+1} 적용 중 ===",
                            rich_message=f"🔧 Applying Step {i+1}: [cyan]{step_type}[/cyan]")
            self.console.data_operation(f"Step {i+1} input", current_data.shape, f"Processing {step_type}")
            transformer = step_info['transformer']
            original_target_columns = step_info['target_columns']
            
            # Global vs Targeted 처리 분기 (transform 시에도 동일 로직)
            if transformer.get_application_type() == 'global':
                # Global 타입: 다시 적용 가능한 컬럼 확인
                target_columns = transformer.get_applicable_columns(current_data)
            else:
                # Targeted 타입: 매핑된 컬럼 재확인
                target_columns = self._find_matching_columns(original_target_columns, current_data.columns)
            
            
            # 대상 컬럼이 존재하지 않는 경우 기본값으로 생성 (Targeted 타입만)
            if transformer.get_application_type() == 'targeted':
                for col in target_columns:
                    if col not in current_data.columns:
                        self.console.warning(f"컬럼 '{col}'이 존재하지 않아 기본값 0으로 생성합니다.",
                                           rich_message=f"   ⚠️  Missing column [yellow]{col}[/yellow], creating with default value 0")
                        current_data[col] = 0
            
            if not target_columns:
                self.console.warning(f"Transform 시 적용할 컬럼이 없습니다: {transformer.__class__.__name__}",
                                   rich_message=f"   ⚠️  No columns to apply for [red]{transformer.__class__.__name__}[/red]")
                continue
            
            # 대상 컬럼 데이터 추출
            target_data = current_data[target_columns]
            
            # transformer 적용
            transformed_data = transformer.transform(target_data)
            
            # 결과를 현재 데이터에 병합
            preserves_names = transformer.preserves_column_names()
            self.console.info(f"변환된 데이터 shape: {transformed_data.shape}, 컬럼: {list(transformed_data.columns)}",
                            rich_message=f"   🔄 Transformed: [green]{transformed_data.shape}[/green], preserves names: [cyan]{preserves_names}[/cyan]")
            
            if transformer.preserves_column_names():
                # 컬럼명이 보존되는 경우: 기존 컬럼 업데이트
                for col in transformed_data.columns:
                    current_data[col] = transformed_data[col]
                self.console.info(f"보존된 컬럼들을 업데이트함: {list(transformed_data.columns)}",
                                rich_message=f"   ✅ Updated preserved columns: [green]{len(transformed_data.columns)}[/green]")
            else:
                # 컬럼명이 변경되는 경우: 원본 컬럼은 유지하고 새 컬럼만 추가 (지연 삭제)
                # 새로운 컬럼들을 현재 데이터에 추가
                for col in transformed_data.columns:
                    current_data[col] = transformed_data[col]
                self.console.info(f"새로운 컬럼들 추가: {list(transformed_data.columns)}",
                                rich_message=f"   ➕ Added new columns: [green]{len(transformed_data.columns)}[/green]")
            
            self.console.data_operation(f"Step {i+1} completed", 
                                       current_data.shape, 
                                       f"Total columns: {len(current_data.columns)}")
        
        # 모든 변환 완료 후: 지연 삭제할 원본 컬럼들 일괄 제거
        if hasattr(self, '_columns_to_delete') and self._columns_to_delete:
            columns_to_remove = [col for col in self._columns_to_delete if col in current_data.columns]
            if columns_to_remove:
                self.console.info(f"Transform 지연 삭제 실행: {columns_to_remove}",
                                rich_message=f"🗑️  Executing delayed column deletion: [red]{len(columns_to_remove)}[/red] columns")
                current_data = current_data.drop(columns=columns_to_remove)
                self.console.data_operation("Transform final result", 
                                           current_data.shape,
                                           f"Final columns: {len(current_data.columns)}")
        
        return current_data

    def save(self, file_path: str):
        pass

    @classmethod
    def load(cls, file_path: str) -> 'Preprocessor':
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
            
        mapped_columns = []
        available_set = set(available_columns)
        
        for col in target_columns:
            if col in available_set:
                # 컬럼이 존재하는 경우 - 정상 케이스
                mapped_columns.append(col)
            else:
                # 컬럼이 존재하지 않는 경우 - 전처리기에서 처리하도록 그대로 전달
                # Use console for internal column mapping information
                self.console.info(f"컬럼 '{col}'이 현재 데이터에 없습니다. 전처리기에서 처리될 예정입니다.",
                                 rich_message=f"🔍 Column [yellow]'{col}'[/yellow] not found → will be processed by preprocessor")
                mapped_columns.append(col)
            
        return mapped_columns
        
