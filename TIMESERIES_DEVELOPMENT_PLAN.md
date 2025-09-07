# 🔮 Timeseries Task + DataHandler 아키텍처 개발 계획

**목표**: 확장 가능한 DataHandler 아키텍처를 통해 timeseries를 추가하고, 향후 딥러닝까지 지원하는 강건한 시스템 구축

## 📊 현재 시스템 분석 및 문제점

### ❌ 현재 구조의 한계
```
src/components/trainer/modules/data_handler.py  # 모든 task 처리가 한 파일에 집중
├── split_data()          # task별 분기 처리 
├── prepare_training_data()  # task별 분기 처리
└── _get_exclude_columns()   # 공통 로직
```

**문제점**:
1. **단일 파일 집중**: 모든 task 로직이 한 파일에 혼재
2. **확장성 부족**: 새로운 task 추가 시 기존 코드 수정 필요
3. **책임 분산 부족**: tabular ML과 timeseries의 특성이 다름에도 동일 처리
4. **딥러닝 미준비**: 배치 생성, 시퀀스 처리 등 딥러닝 요구사항 미고려

### 🎯 새로운 아키텍처 목표
1. **Task별 전용 핸들러**: tabular vs timeseries vs deeplearning 분리
2. **확장성**: Registry 패턴으로 새로운 handler 쉽게 추가
3. **파이프라인 일관성**: Data Adapter → Fetcher → **DataHandler** → Preprocessor → Trainer
4. **미래 준비**: 딥러닝 배치 생성, 시퀀스 처리까지 대비

---

## 🏗️ 새로운 DataHandler 아키텍처

### **전체 아키텍처 개요**
```
src/components/datahandler/
├── registry.py                    # DataHandler Registry
├── __init__.py
├── base.py                        # BaseDataHandler 인터페이스
└── modules/
    ├── tabular_handler.py         # 기존 4개 task (classification, regression, clustering, causal)
    ├── timeseries_handler.py      # timeseries 전용 
    └── deeplearning_handler.py    # 향후 딥러닝용 (batch 생성, sequence 처리)
```

### **Registry 패턴**
```python
# src/components/datahandler/registry.py
class DataHandlerRegistry:
    handlers: Dict[str, Type[BaseDataHandler]] = {}
    
    @classmethod
    def get_handler_for_task(cls, task_type: str) -> BaseDataHandler:
        # task_type에 따른 자동 매핑
        handler_mapping = {
            "classification": "tabular",
            "regression": "tabular", 
            "clustering": "tabular",
            "causal": "tabular",
            "timeseries": "timeseries",
            "deeplearning": "deeplearning"  # 향후
        }
        handler_type = handler_mapping.get(task_type, "tabular")
        return cls.create(handler_type)
```

### **파이프라인 흐름**
```
Data Adapter (CSV/SQL/API) 
    ↓ 
Fetcher (Feature Store 연동)
    ↓
🆕 DataHandler (task별 데이터 처리)  ← 새로 추가되는 계층
    ├─ TabularHandler: feature selection, validation, basic split
    ├─ TimeseriesHandler: time features, time-based split, lag/rolling
    └─ DeepLearningHandler: batch generation, sequence padding, tokenization
    ↓
Preprocessor (scaling, encoding 등 범용 처리)
    ↓
Trainer (model fitting)
```

---

## 🚀 Phase별 개발 계획

---

## **Phase 1: DataHandler 아키텍처 구축** (핵심 리팩토링)

### 1.1 DataHandler 컴포넌트 신설

#### A. 기본 구조 생성
```bash
mkdir -p src/components/datahandler/modules/
touch src/components/datahandler/{__init__.py,registry.py,base.py}
touch src/components/datahandler/modules/{tabular_handler.py,timeseries_handler.py}
```

#### B. BaseDataHandler 인터페이스 (`src/components/datahandler/base.py`)
```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd
from src.settings import Settings

class BaseDataHandler(ABC):
    """데이터 핸들러 기본 인터페이스"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_interface = settings.recipe.data.data_interface
        
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """데이터 준비 (X, y, additional_data 반환)"""
        pass
        
    @abstractmethod 
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Test 분할"""
        pass
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """데이터 검증 (각 handler별 구체화)"""
        return True
        
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 메타 정보 반환"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_ratio": (df.isnull().sum() / len(df)).to_dict()
        }
```

#### C. DataHandler Registry (`src/components/datahandler/registry.py`)
```python
from typing import Dict, Type, Optional
from .base import BaseDataHandler
from src.utils.system.logger import logger

class DataHandlerRegistry:
    """DataHandler 중앙 레지스트리"""
    handlers: Dict[str, Type[BaseDataHandler]] = {}
    
    @classmethod
    def register(cls, handler_type: str, handler_class: Type[BaseDataHandler]):
        if not issubclass(handler_class, BaseDataHandler):
            raise TypeError(f"{handler_class.__name__} must be a subclass of BaseDataHandler")
        cls.handlers[handler_type] = handler_class
        logger.debug(f"DataHandler registered: {handler_type} -> {handler_class.__name__}")
    
    @classmethod
    def create(cls, handler_type: str, *args, **kwargs) -> BaseDataHandler:
        handler_class = cls.handlers.get(handler_type)
        if not handler_class:
            available = list(cls.handlers.keys())
            raise ValueError(f"Unknown handler type: '{handler_type}'. Available: {available}")
        return handler_class(*args, **kwargs)
    
    @classmethod
    def get_handler_for_task(cls, task_type: str, settings) -> BaseDataHandler:
        """task_type에 따른 자동 handler 매핑"""
        handler_mapping = {
            "classification": "tabular",
            "regression": "tabular",
            "clustering": "tabular", 
            "causal": "tabular",
            "timeseries": "timeseries"
        }
        handler_type = handler_mapping.get(task_type, "tabular")
        return cls.create(handler_type, settings)
```

### 1.2 기존 data_handler.py → tabular_handler.py 리팩토링

#### A. 파일 이동 및 클래스화
```python
# src/components/datahandler/modules/tabular_handler.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

from ..base import BaseDataHandler
from ..registry import DataHandlerRegistry
from src.utils.system.logger import logger

class TabularDataHandler(BaseDataHandler):
    """전통적인 테이블 형태 ML을 위한 데이터 핸들러 (classification, regression, clustering, causal)"""
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Test 분할 (조건부 stratify)"""
        test_size = 0.2
        stratify_series = self._get_stratify_column_data(df)
        
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=stratify_series
        )
        return train_df, test_df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """테이블 형태 데이터 준비 (기존 prepare_training_data 로직)"""
        task_type = self.data_interface.task_type
        exclude_cols = self._get_exclude_columns(df)
        
        if task_type in ["classification", "regression"]:
            return self._prepare_supervised_data(df, exclude_cols)
        elif task_type == "clustering":
            return self._prepare_clustering_data(df, exclude_cols)  
        elif task_type == "causal":
            return self._prepare_causal_data(df, exclude_cols)
        else:
            raise ValueError(f"지원하지 않는 task_type: {task_type}")
    
    # 기존 로직들을 private 메서드로 이동
    def _prepare_supervised_data(self, df, exclude_cols): ...
    def _prepare_clustering_data(self, df, exclude_cols): ... 
    def _prepare_causal_data(self, df, exclude_cols): ...
    def _get_stratify_column_data(self, df): ...
    def _get_exclude_columns(self, df): ...
    def _check_missing_values_warning(self, X): ...  # 기존 함수들 모두 포함

# Self-registration
DataHandlerRegistry.register("tabular", TabularDataHandler)
```

### 1.3 Recipe Schema 확장 (`src/settings/recipe.py`)

#### A. TaskType 확장
```python
# Line 127 근처 수정
task_type: Literal["classification", "regression", "clustering", "causal", "timeseries"] = Field(
    ..., 
    description="ML 태스크 타입"
)
```

#### B. DataInterface에 timeseries 필드 추가
```python
class DataInterface(BaseModel):
    """데이터 인터페이스 설정"""
    task_type: Literal["classification", "regression", "clustering", "causal", "timeseries"] = Field(...)
    target_column: str = Field(..., description="타겟 컬럼 이름")
    
    # 기존 필드들...
    feature_columns: Optional[List[str]] = Field(None, ...)
    treatment_column: Optional[str] = Field(None, ...)
    entity_columns: List[str] = Field(..., ...)
    
    # ✅ Timeseries 전용 필드들 추가
    timestamp_column: Optional[str] = Field(None, description="시계열 타임스탬프 컬럼 (timeseries task에서 필수)")
    forecast_horizon: Optional[int] = Field(None, ge=1, le=365, description="예측 기간 (timeseries task에서 필수)")
    frequency: Optional[str] = Field(None, description="시계열 주기 ('D', 'H', 'M', 'W', 'Y')")
    
    @model_validator(mode='after')
    def validate_timeseries_fields(self):
        """timeseries task일 때 필수 필드 검증"""
        if self.task_type == "timeseries":
            if not self.timestamp_column:
                raise ValueError("timeseries task에서는 timestamp_column이 필수입니다")
            if not self.forecast_horizon:
                raise ValueError("timeseries task에서는 forecast_horizon이 필수입니다")
        return self
```

### 1.4 TimeseriesDataHandler 구현 (`src/components/datahandler/modules/timeseries_handler.py`)

```python
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta

from ..base import BaseDataHandler  
from ..registry import DataHandlerRegistry
from src.utils.system.logger import logger

class TimeseriesDataHandler(BaseDataHandler):
    """시계열 데이터 전용 핸들러"""
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """시계열 데이터 검증"""
        timestamp_col = self.data_interface.timestamp_column
        target_col = self.data_interface.target_column
        
        # 필수 컬럼 존재 검증
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp 컬럼 '{timestamp_col}'을 찾을 수 없습니다")
        if target_col not in df.columns:
            raise ValueError(f"Target 컬럼 '{target_col}'을 찾을 수 없습니다")
            
        # 타임스탬프 데이터 타입 검증
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                logger.info(f"Timestamp 컬럼을 datetime으로 변환했습니다: {timestamp_col}")
            except:
                raise ValueError(f"Timestamp 컬럼 '{timestamp_col}'을 datetime으로 변환할 수 없습니다")
        
        return True
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """시간 기준 분할 (시간 순서 유지)"""
        timestamp_col = self.data_interface.timestamp_column
        test_size = 0.2
        
        # 시간 순서로 정렬 (필수)
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # 시간 기준 분할점 계산
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        logger.info(f"시계열 시간 기준 분할: Train({len(train_df)}) / Test({len(test_df)})")
        logger.info(f"Train 기간: {train_df[timestamp_col].min()} ~ {train_df[timestamp_col].max()}")
        logger.info(f"Test 기간: {test_df[timestamp_col].min()} ~ {test_df[timestamp_col].max()}")
        
        return train_df, test_df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """시계열 데이터 준비"""
        # 데이터 검증
        self.validate_data(df)
        
        timestamp_col = self.data_interface.timestamp_column
        target_col = self.data_interface.target_column
        
        # 시간 순서 정렬 (필수)
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # 시계열 특성 생성
        df_with_features = self._generate_time_features(df)
        
        # Feature/Target 분리
        exclude_cols = self._get_exclude_columns(df_with_features)
        
        if self.data_interface.feature_columns is None:
            # 자동 선택: timestamp, target, entity 제외
            auto_exclude = [timestamp_col, target_col] + exclude_cols
            X = df_with_features.drop(columns=[c for c in auto_exclude if c in df_with_features.columns])
            logger.info(f"Timeseries feature columns 자동 선택: {list(X.columns)}")
        else:
            # 명시적 선택
            X = df_with_features[self.data_interface.feature_columns]
        
        # 숫자형 컬럼만 선택
        X = X.select_dtypes(include=[np.number])
        
        # 결측치 경고
        self._check_missing_values_warning(X)
        
        y = df[target_col]
        
        additional_data = {
            'timestamp': df[timestamp_col],
            'forecast_horizon': self.data_interface.forecast_horizon,
            'frequency': self.data_interface.frequency
        }
        
        return X, y, additional_data
    
    def _generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시계열 시간 기반 특성 자동 생성"""
        timestamp_col = self.data_interface.timestamp_column
        df_copy = df.copy()
        
        # 기본 시간 특성
        df_copy['year'] = df_copy[timestamp_col].dt.year
        df_copy['month'] = df_copy[timestamp_col].dt.month
        df_copy['day'] = df_copy[timestamp_col].dt.day
        df_copy['dayofweek'] = df_copy[timestamp_col].dt.dayofweek
        df_copy['quarter'] = df_copy[timestamp_col].dt.quarter
        df_copy['is_weekend'] = df_copy['dayofweek'].isin([5, 6]).astype(int)
        
        # Lag features (1, 2, 3, 7, 14일 전)
        target_col = self.data_interface.target_column
        for lag in [1, 2, 3, 7, 14]:
            df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
            
        # Rolling features (3, 7, 14일 평균)
        for window in [3, 7, 14]:
            df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window=window).mean()
            df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window=window).std()
        
        logger.info(f"시계열 특성 생성 완료: {len(df_copy.columns) - len(df)}개 특성 추가")
        return df_copy
    
    def _get_exclude_columns(self, df: pd.DataFrame) -> list:
        """시계열에서 제외할 컬럼"""
        exclude_columns = []
        
        # Entity columns
        if self.data_interface.entity_columns:
            exclude_columns.extend(self.data_interface.entity_columns)
            
        return [col for col in exclude_columns if col in df.columns]
    
    def _check_missing_values_warning(self, X: pd.DataFrame, threshold: float = 0.05):
        """5% 이상 결측치 경고 (tabular_handler와 동일)"""
        if X.empty:
            return
            
        missing_info = []
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            missing_ratio = missing_count / len(X)
            
            if missing_ratio >= threshold:
                missing_info.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_ratio': missing_ratio,
                    'total_rows': len(X)
                })
        
        if missing_info:
            logger.warning("⚠️  결측치가 많은 컬럼이 발견되었습니다:")
            for info in missing_info:
                logger.warning(
                    f"   - {info['column']}: {info['missing_count']:,}개 ({info['missing_ratio']:.1%}) "
                    f"/ 전체 {info['total_rows']:,}개 행"
                )
            logger.warning("   💡 전처리 단계에서 결측치 처리를 고려해보세요 (Imputation, 컬럼 제거 등)")
        else:
            logger.info(f"✅ 모든 특성 컬럼의 결측치 비율이 {threshold:.0%} 미만입니다.")

# Self-registration
DataHandlerRegistry.register("timeseries", TimeseriesDataHandler)
```

### 1.5 Trainer 통합 (`src/components/trainer/trainer.py`)

#### A. DataHandler 통합
```python
# imports 수정
from src.components.datahandler import DataHandlerRegistry

class Trainer(BaseTrainer):
    def train(self, df, model, fetcher, preprocessor, evaluator, context_params):
        # 기존 fetcher 로직...
        
        # ✅ DataHandler 사용 (기존 data_handler 함수 대체)
        data_handler = DataHandlerRegistry.get_handler_for_task(
            self.settings.recipe.data.data_interface.task_type,
            self.settings
        )
        
        # 데이터 준비
        X, y, additional_data = data_handler.prepare_data(df)
        
        # 데이터 분할
        train_df, test_df = data_handler.split_data(df)
        
        # 나머지 로직은 기존과 동일...
```

#### B. 기존 import 제거
```python
# 제거: from .modules.data_handler import split_data, prepare_training_data
```

---

## **Phase 2: Timeseries Models Catalog 구축**

### 2.1 Models Catalog 구축 (`src/models/catalog/Timeseries/`)

#### A. `LinearTrend.yaml`
```yaml
class_path: "sklearn.linear_model.LinearRegression"
description: "Linear trend model with automated time-based features"
library: "scikit-learn"
hyperparameters:
  fixed:
    fit_intercept: true
  tunable:
    normalize:
      type: "categorical"
      range: [true, false]
      default: false
  environment_defaults:
    local:
      normalize: false
    dev: 
      normalize: false
    prod:
      normalize: false
supported_tasks: ["timeseries"]
feature_requirements:
  numerical: true
  categorical: false
  text: false
  time_features_auto: true
preprocessing_notes: "TimeseriesDataHandler에서 시간 기반 특성을 자동 생성합니다"
```

#### B. `ARIMA.yaml` (sklearn wrapper 버전)
```yaml
class_path: "src.models.custom.timeseries_wrappers.ARIMAWrapper"  # 커스텀 wrapper
description: "ARIMA model wrapped for sklearn compatibility"
library: "statsmodels"
hyperparameters:
  tunable:
    order_p:
      type: "int"
      range: [0, 5] 
      default: 1
    order_d:
      type: "int"
      range: [0, 2]
      default: 1
    order_q:
      type: "int" 
      range: [0, 5]
      default: 1
  environment_defaults:
    local:
      order_p: 1
      order_d: 1
      order_q: 1
supported_tasks: ["timeseries"]
requires_custom_wrapper: true
feature_requirements:
  numerical: true
  univariate_preferred: true
```

#### C. `ExponentialSmoothing.yaml`
```yaml
class_path: "src.models.custom.timeseries_wrappers.ExponentialSmoothingWrapper"
description: "Exponential Smoothing with sklearn interface"
library: "statsmodels"
hyperparameters:
  tunable:
    trend:
      type: "categorical"
      range: [null, "add", "mul"]
      default: "add"
    seasonal:
      type: "categorical" 
      range: [null, "add", "mul"]
      default: null
    seasonal_periods:
      type: "int"
      range: [2, 52]
      default: 12
supported_tasks: ["timeseries"]
requires_custom_wrapper: true
```

### 2.2 Custom Wrappers 구현 (`src/models/custom/timeseries_wrappers.py`)

```python
"""Timeseries 모델들을 sklearn 인터페이스로 감싸는 wrapper들"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class ARIMAWrapper(BaseEstimator, RegressorMixin):
    """ARIMA 모델을 sklearn 인터페이스로 감싸는 wrapper"""
    
    def __init__(self, order_p=1, order_d=1, order_q=1):
        self.order_p = order_p
        self.order_d = order_d  
        self.order_q = order_q
        self.model_ = None
        self.fitted_model_ = None
    
    def fit(self, X, y):
        from statsmodels.tsa.arima.model import ARIMA
        
        # ARIMA는 univariate이므로 y만 사용
        self.model_ = ARIMA(y, order=(self.order_p, self.order_d, self.order_q))
        self.fitted_model_ = self.model_.fit()
        return self
    
    def predict(self, X):
        if self.fitted_model_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # X의 길이만큼 예측
        forecast_steps = len(X)
        forecast = self.fitted_model_.forecast(steps=forecast_steps)
        
        return np.array(forecast)

class ExponentialSmoothingWrapper(BaseEstimator, RegressorMixin):
    """Exponential Smoothing sklearn wrapper"""
    
    def __init__(self, trend="add", seasonal=None, seasonal_periods=12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model_ = None
    
    def fit(self, X, y):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        self.fitted_model_ = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods if self.seasonal else None
        ).fit()
        return self
    
    def predict(self, X):
        if self.fitted_model_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecast_steps = len(X)
        forecast = self.fitted_model_.forecast(steps=forecast_steps)
        return np.array(forecast)
```

---

## **Phase 3: Factory 통합 및 고급 기능**

### 3.1 Factory DataHandler 통합 (`src/factory/factory.py`)

```python
def create_data_handler(self, task_type: Optional[str] = None) -> Any:
    """DataHandler 생성 (일관된 접근 패턴)"""
    from src.components.datahandler import DataHandlerRegistry
    
    task_type = task_type or self._recipe.data.data_interface.task_type
    
    try:
        handler = DataHandlerRegistry.get_handler_for_task(task_type, self._settings)
        logger.info(f"✅ Created data handler: {task_type}")
        return handler
    except Exception as e:
        logger.error(f"Failed to create data handler for '{task_type}': {e}")
        raise
```

### 3.2 Timeseries Evaluator (`src/components/evaluator/timeseries.py`)

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.interface import BaseEvaluator
from .registry import EvaluatorRegistry

def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE 계산"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """SMAPE 계산"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

class TimeSeriesEvaluator(BaseEvaluator):
    """시계열 전용 평가기"""
    
    def evaluate(self, y_true, y_pred, data_interface=None) -> dict:
        """시계열 평가 지표 계산"""
        
        # 기본 regression 지표
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # 시계열 특화 지표
        mape = mean_absolute_percentage_error(y_true, y_pred)
        smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        
        return {
            # Regression 지표
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'r2': r2,
            
            # Timeseries 특화 지표
            'mape': mape,
            'smape': smape,
            
            # 메타 정보
            'n_predictions': len(y_true),
            'evaluation_type': 'timeseries'
        }

# Self-registration
EvaluatorRegistry.register_evaluator("timeseries", TimeSeriesEvaluator)
```

---

## **Phase 4: Recipe Builder 및 UI**

### 4.1 Recipe Builder 확장 (`src/cli/utils/recipe_builder.py`)

```python
# TASK_METRICS 확장 (Line 28 근처)
TASK_METRICS = {
    "Classification": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "Regression": ["mae", "mse", "rmse", "r2", "mape"],
    "Clustering": ["silhouette_score", "davies_bouldin", "calinski_harabasz"],
    "Causal": ["ate", "att", "confidence_intervals"],
    "Timeseries": ["mae", "mse", "rmse", "r2", "mape", "smape"]  # ✅ 추가
}

# _configure_data_interface 메서드에 timeseries 케이스 추가
def _configure_data_interface(self, selections: Dict[str, Any]):
    """데이터 인터페이스 설정"""
    
    # 기존 로직...
    
    # ✅ Timeseries 전용 설정 추가
    if selections["task"] == "Timeseries":
        self.ui.show_info("🕐 시계열 데이터 설정")
        
        # 타임스탬프 컬럼 선택
        timestamp_col = self.ui.select_from_list(
            "타임스탬프 컬럼을 선택하세요",
            available_columns
        )
        selections["timestamp_column"] = timestamp_col
        
        # Forecast horizon 설정
        horizon = self.ui.number_input(
            "예측 기간을 입력하세요 (예: 30일)",
            default=30,
            min_value=1,
            max_value=365
        )
        selections["forecast_horizon"] = horizon
        
        # Frequency 설정
        freq = self.ui.select_from_list(
            "시계열 주기를 선택하세요",
            ["auto", "D (일별)", "H (시간별)", "M (월별)", "W (주별)", "Y (연별)"]
        )
        freq_code = freq.split()[0] if freq != "auto" else None
        selections["frequency"] = freq_code
        
        self.ui.show_success("✅ 시계열 설정 완료")
        self.ui.print_divider()
```

### 4.2 Timeseries Template (`src/cli/templates/recipes/timeseries_template.yaml.j2`)

```yaml
name: "{{ recipe_name }}"
description: "Time Series Forecasting Recipe"

data:
  loader:
    source_uri: "{{ source_uri }}"
  fetcher:
    mode: "offline"
    feature_store:
      enabled: false
  data_interface:
    task_type: "timeseries"
    target_column: "{{ target_column }}"
    timestamp_column: "{{ timestamp_column }}"
    forecast_horizon: {{ forecast_horizon }}
    {% if frequency %}frequency: "{{ frequency }}"{% endif %}
    entity_columns: {{ entity_columns }}
    {% if feature_columns %}feature_columns: {{ feature_columns }}{% endif %}

preprocessing:
  enabled: {{ preprocessing_enabled }}
  steps: {{ preprocessing_steps }}

model:
  algorithm: "{{ algorithm }}"
  hyperparameters:
    tuning_enabled: {{ tuning_enabled }}
    {% if tuning_enabled %}
    optimization_metric: "{{ optimization_metric }}"
    n_trials: {{ n_trials }}
    timeout: {{ timeout }}
    fixed: {{ fixed_params }}
    tunable: {{ tunable_params }}
    {% else %}
    values: {{ hyperparameter_values }}
    {% endif %}

validation:
  method: "train_test_split"  # 시계열은 항상 time-based split
  test_size: {{ test_size }}
  random_state: {{ random_state }}

evaluation:
  metrics: {{ evaluation_metrics }}
```

---

## **Phase 5: API/배포 지원**

### 5.1 PyfuncWrapper 확장 (`src/factory/artifact.py`)

```python
def predict(self, context, model_input):
    """예측 수행 (timeseries 지원 추가)"""
    
    if self._task_type == "timeseries":
        return self._predict_timeseries(context, model_input)
    else:
        # 기존 로직
        return self._predict_tabular(context, model_input)

def _predict_timeseries(self, context, model_input):
    """시계열 예측 수행"""
    df_input = self._convert_input_to_dataframe(model_input)
    
    # DataHandler로 시계열 특성 생성
    from src.components.datahandler import DataHandlerRegistry
    data_handler = DataHandlerRegistry.get_handler_for_task("timeseries", self._settings)
    
    # 특성 생성 (timestamp 기반)
    df_with_features = data_handler._generate_time_features(df_input)
    
    # 예측 수행
    predictions = self._model.predict(processed_features)
    
    # 미래 타임스탬프 생성
    forecast_horizon = self._additional_data.get('forecast_horizon', len(predictions))
    last_timestamp = df_input[self._timestamp_column].max()
    future_timestamps = self._generate_future_timestamps(
        last_timestamp, 
        forecast_horizon,
        self._frequency
    )
    
    return {
        'predictions': predictions.tolist(),
        'timestamps': [ts.isoformat() for ts in future_timestamps],
        'forecast_horizon': forecast_horizon,
        'model_type': 'timeseries',
        'frequency': self._frequency
    }
```

---

## **Phase 6: 딥러닝 확장 준비** (미래)

### 6.1 DeepLearning DataHandler 설계

```python
# src/components/datahandler/modules/deeplearning_handler.py
class DeepLearningDataHandler(BaseDataHandler):
    """딥러닝을 위한 데이터 핸들러"""
    
    def prepare_data(self, df):
        """배치 생성, 시퀀스 패딩, 토크나이제이션 등"""
        pass
        
    def create_dataloaders(self, train_df, val_df, batch_size=32):
        """PyTorch/TensorFlow DataLoader 생성"""
        pass
        
    def prepare_sequences(self, df, sequence_length=50):
        """시퀀스 데이터 준비 (RNN/LSTM용)"""
        pass
```

---

## 🎯 구현 우선순위 및 마일스톤

### **🚨 Phase 1 (필수): 아키텍처 리팩토링** 
**목표**: DataHandler 아키텍처 구축 및 기존 코드 마이그레이션
1. BaseDataHandler + Registry 구현
2. TabularDataHandler 구현 (기존 data_handler.py 이관)
3. Recipe Schema에 timeseries 필드 추가
4. TimeseriesDataHandler 기본 구현
5. Trainer에서 DataHandler 통합

**완료 기준**: ✅ 기존 4개 task가 새 아키텍처에서 정상 작동

### **⚡ Phase 2 (핵심): Timeseries 완성**
**목표**: Timeseries 완전 지원
1. LinearTrend, ARIMA, ExponentialSmoothing 모델 구현
2. Custom Wrapper 구현
3. TimeseriesEvaluator 구현

**완료 기준**: ✅ CLI로 timeseries 학습 및 평가 가능

### **🎨 Phase 3 (향상): 사용성**
**목표**: 사용자 친화적 환경
1. Recipe Builder timeseries 설정 UI
2. Timeseries template 구현

**완료 기준**: ✅ CLI로 timeseries 프로젝트 쉽게 생성

### **🚀 Phase 4 (완성): 배포**
**목표**: 운영 환경 준비
1. PyfuncWrapper timeseries 지원
2. API endpoints 확장

**완료 기준**: ✅ Timeseries API 서비스 가능

### **🔮 Phase 5 (미래): 딥러닝 확장**
**목표**: 딥러닝 생태계 준비
1. DeepLearningDataHandler 설계
2. 배치 생성, 시퀀스 처리 로직

---

## 📋 핵심 설계 철학

### ✅ **관심사 분리 (Separation of Concerns)**
- **DataHandler**: 데이터 형태별 특화 처리 (tabular vs timeseries vs deeplearning)
- **Preprocessor**: 범용 변환 (scaling, encoding 등)
- **Trainer**: 모델 학습 로직

### ✅ **확장성 (Extensibility)**
- Registry 패턴으로 새로운 Handler 쉽게 추가
- 각 Handler는 독립적으로 발전 가능

### ✅ **하위 호환성 (Backward Compatibility)**
- 기존 4개 task는 TabularDataHandler로 완전 호환
- 기존 Recipe/설정 구조 유지

### ✅ **미래 대응성 (Future-Proofing)**
- 딥러닝, 멀티모달 등 미래 확장 대비
- 배치 처리, GPU 가속 등 고급 기능 수용 가능

---

## 🎉 **결과적으로 달성하는 것**

1. **🔄 깔끔한 아키텍처**: 각 컴포넌트가 명확한 책임을 가짐
2. **🚀 Timeseries 완전 지원**: 학습부터 API 서빙까지
3. **💎 확장성**: 새로운 데이터 타입/모델 쉽게 추가
4. **🛡️ 견고함**: 각 Handler가 독립적으로 테스트/발전 가능
5. **🌟 미래 준비**: 딥러닝, 멀티모달까지 수용 가능한 구조

이 계획은 **현재를 존중하면서 미래를 준비**하는 점진적 진화 전략입니다! 🚀