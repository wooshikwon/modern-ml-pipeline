# 🧩 컴포넌트 가이드 (Component Guide)

Modern ML Pipeline의 기능을 확장하고 싶으신가요?  
새로운 모델, 전처리기, 데이터 소스를 추가하는 방법을 안내합니다.

---

## 1. 컴포넌트 확장 원칙

모든 컴포넌트는 다음 3단계를 통해 추가됩니다.

1.  **상속**: 해당 컴포넌트의 `Base` 클래스를 상속받습니다.
2.  **구현**: 필수 메서드(추상 메서드)를 구현합니다.
3.  **등록**: `Registry.register`를 호출하여 시스템에 알립니다.

---

## 2. 확장 가이드: 새 모델 추가하기

기존 라이브러리(Scikit-learn 등) 호환 모델이 아니라, 커스텀 로직이 필요한 모델을 추가하는 예시입니다.

**파일 생성**: `src/models/custom/my_model.py`

```python
from ..base import BaseModel
# 1. Base 클래스 상속
class MyCustomModel(BaseModel):
    def __init__(self, param1=10):
        self.param1 = param1
        self.model = None

    # 2. 필수 메서드 구현 (fit, predict)
    def fit(self, X, y):
        # 학습 로직 구현
        print(f"Training with {self.param1}")
        return self

    def predict(self, X):
        # 예측 로직 구현
        return [0] * len(X)

# (참고) 모델은 Registry 등록 없이 class_path로 직접 로드됩니다.
```

**사용법 (Recipe YAML)**:
```yaml
model:
  class_path: src.models.custom.my_model.MyCustomModel
  hyperparameters:
    values:
      param1: 50
```

---

## 2-1. 외부 라이브러리 모델 래핑하기

PyTorch-TabNet, FTTransformer 등 외부 라이브러리 모델을 사용할 때는 **BaseModel을 상속한 wrapper**를 만드는 것을 권장합니다.

**이유:**
- BaseModel 인터페이스 일관성 유지 (DataFrame 입력 지원)
- Trainer, Evaluator 등 컴포넌트에서 모델별 분기 로직 제거
- 모델 특성(numpy 변환 등)을 모델 레이어에 캡슐화

**예시: TabNet Wrapper** (`src/models/custom/tabnet_wrapper.py`)

```python
from src.models.base import BaseModel
import pandas as pd
import numpy as np

class TabNetClassifierWrapper(BaseModel):
    def __init__(self, n_d=8, n_a=8, **kwargs):
        self.n_d = n_d
        self.n_a = n_a
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs):
        from pytorch_tabnet.tab_model import TabNetClassifier

        # DataFrame -> numpy 변환 (모델 내부에서 처리)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        self.model = TabNetClassifier(n_d=self.n_d, n_a=self.n_a)
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_np)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_np)
```

**Catalog 등록** (`src/models/catalog/Classification/TabNetClassifier.yaml`)

```yaml
class_path: src.models.custom.tabnet_wrapper.TabNetClassifierWrapper
description: TabNet Classifier (BaseModel wrapper)
library: pytorch-tabnet
hyperparameters:
  fixed:
    seed: 42
  tunable:
    n_d:
      type: int
      range: [8, 64]
      default: 8
```

대화형 CLI (`mmp get-recipe`)에서 모델 선택 시 Catalog의 class_path가 자동으로 Recipe에 반영됩니다.

**현재 제공되는 Wrapper 모델:**
| 모델 | Wrapper 경로 |
|------|-------------|
| TabNetClassifier | `src.models.custom.tabnet_wrapper.TabNetClassifierWrapper` |
| TabNetRegressor | `src.models.custom.tabnet_wrapper.TabNetRegressorWrapper` |
| FTTransformerClassifier | `src.models.custom.ft_transformer.FTTransformerClassifier` |
| FTTransformerRegressor | `src.models.custom.ft_transformer.FTTransformerRegressor` |
| LSTMTimeSeries | `src.models.custom.lstm_timeseries.LSTMTimeSeries` |

---

## 3. 확장 가이드: 새 전처리기 추가하기

특정 컬럼의 값을 변환하는 새로운 전처리 로직을 추가해봅니다.

**파일 생성**: `src/components/preprocessor/modules/my_scaler.py`

```python
from src.components.preprocessor.base import BasePreprocessor
from src.components.preprocessor.registry import PreprocessorStepRegistry

# 1. 상속
class MyScaler(BasePreprocessor):
    def __init__(self, factor=2):
        self.factor = factor

    # 2. 구현
    def fit(self, X, y=None):
        return self  # 학습할 게 없으면 self 반환

    def transform(self, X):
        return X * self.factor

    def get_application_type(self):
        return 'global'  # 또는 'targeted' (특정 컬럼만)

# 3. 등록
PreprocessorStepRegistry.register("my_scaler", MyScaler)
```

**사용법 (Recipe YAML)**:
```yaml
preprocessor:
  steps:
    - type: "my_scaler"
      factor: 10
```

---

## 4. 확장 가이드: 새 데이터 어댑터 추가하기

새로운 데이터 소스(예: MongoDB, Kafka)를 연결하고 싶을 때 사용합니다.

**파일 생성**: `src/components/adapter/modules/mongo_adapter.py`

```python
from src.components.adapter.base import BaseAdapter
from src.components.adapter.registry import AdapterRegistry

class MongoAdapter(BaseAdapter):
    def read(self, source, **kwargs):
        # MongoDB 읽기 로직
        return pd.DataFrame(...)

    def write(self, df, target, **kwargs):
        # MongoDB 쓰기 로직
        pass

# 등록 키: 'mongo'
AdapterRegistry.register("mongo", MongoAdapter)
```

**사용법 (Config YAML)**:
```yaml
data_source:
  adapter_type: "mongo"
  config:
    uri: "mongodb://localhost:27017"
```

---

## 5. 전체 컴포넌트 목록

확장 가능한 주요 컴포넌트들입니다.

| 컴포넌트 | 역할 | Base 클래스 위치 | Registry 위치 |
|----------|------|-----------------|---------------|
| **Adapter** | 데이터 I/O | `src.components.adapter.base` | `src.components.adapter.registry` |
| **Fetcher** | 피처 추가 조회 | `src.components.fetcher.base` | `src.components.fetcher.registry` |
| **Evaluator** | 성능 평가 | `src.components.evaluator.base` | `src.components.evaluator.registry` |
| **Calibrator** | 확률 보정 | `src.components.calibration.base` | `src.components.calibration.registry` |
