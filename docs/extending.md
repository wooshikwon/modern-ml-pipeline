# 컴포넌트 확장 가이드

## 1. 확장 원칙: 상속 → 구현 → 등록

모든 컴포넌트는 3단계로 추가된다.

1. **상속** — 해당 컴포넌트의 `Base` 클래스를 상속
2. **구현** — 추상 메서드 구현
3. **등록** — `Registry.register(key, Class)`로 시스템에 등록 (모델은 `class_path`로 직접 로드하므로 등록 불필요)

### 3-Tier 컴포넌트 아키텍처

| Tier | 이름 | 특징 | 예시 |
|------|------|------|------|
| **1** | Atomic | 독립적 단위, 다른 컴포넌트에 의존하지 않음 | Adapter, Evaluator, Fetcher |
| **2** | Composite | Tier 1을 조합하여 동작 | Trainer, DataHandler |
| **3** | Orchestrator | 여러 하위 단계를 순차 실행 | Preprocessor |

실용적 의미: Tier 1부터 확장하는 것이 가장 간단하다. Tier가 올라갈수록 의존성이 많아지므로 기존 코드를 충분히 이해한 뒤 확장한다.

### import-linter 제약

`pyproject.toml`에 정의된 규칙:

```
mmp.components → mmp.factory  (금지)
```

**컴포넌트는 factory를 import할 수 없다.** 컴포넌트는 자신의 Base와 Registry만 참조하고, factory가 컴포넌트를 조립한다.

---

## 2. 새 모델 추가

모델은 Registry 등록 없이 `class_path`로 직접 로드된다.

### 파일 생성: `mmp/models/custom/my_model.py`

```python
from mmp.models.base import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, param1=10):
        self.param1 = param1

    def fit(self, X, y, **kwargs):
        # 학습 로직
        return self

    def predict(self, X):
        # 예측 로직
        return [0] * len(X)
```

### Recipe YAML

```yaml
model:
  class_path: mmp.models.custom.my_model.MyCustomModel
  hyperparameters:
    values:
      param1: 50
```

### 실제 예시: QuantileRegressorEnsemble

`mmp/models/custom/quantile_ensemble.py` -- 여러 분위수별 GBM 모델을 앙상블하는 커스텀 모델.

```python
from mmp.models.base import BaseModel

class QuantileRegressorEnsemble(BaseModel):
    def __init__(self, base_class_path: str, quantiles: list[float], **kwargs):
        self.base_class_path = base_class_path
        self.quantiles = quantiles
        self.base_params = dict(kwargs)
        self.models: dict[float, Any] = {}

    def fit(self, X, y=None, **kwargs):
        model_class = self._load_class()
        library = self._detect_library()
        obj_fn = _LIBRARY_OBJECTIVE_MAP[library]

        for q in self.quantiles:
            q_params = {**self.base_params, **obj_fn(q)}
            model = model_class(**q_params)
            model.fit(X.values, y.values)
            self.models[q] = model
        return self

    def predict(self, X) -> pd.DataFrame:
        # 각 분위수별 예측 결과를 DataFrame 컬럼으로 반환
        # pred_p10, pred_p50, pred_p90 등
        ...
```

핵심 패턴:
- `BaseModel`을 상속하되, `predict`가 DataFrame을 반환 (다중 분위수 출력)
- `base_class_path`로 LightGBM/XGBoost/CatBoost를 동적 로드
- 라이브러리별 quantile objective를 자동 매핑

Recipe:

```yaml
model:
  class_path: mmp.models.custom.quantile_ensemble.QuantileRegressorEnsemble
  hyperparameters:
    values:
      base_class_path: lightgbm.LGBMRegressor
      quantiles: [0.1, 0.5, 0.9]
      n_estimators: 100
```

---

## 3. 새 전처리기 추가

전처리기는 Registry에 등록해야 한다.

### 파일 생성: `mmp/components/preprocessor/modules/my_scaler.py`

```python
from mmp.components.preprocessor.base import BasePreprocessor
from mmp.components.preprocessor.registry import PreprocessorStepRegistry

# 1. 상속
class MyScaler(BasePreprocessor):
    def __init__(self, factor=2):
        self.factor = factor

    # 2. 구현
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.factor

    def get_application_type(self):
        return 'global'  # 또는 'targeted' (특정 컬럼만)

# 3. 등록
PreprocessorStepRegistry.register("my_scaler", MyScaler)
```

### Recipe YAML

```yaml
preprocessor:
  steps:
    - type: "my_scaler"
      factor: 10
```

### Self-Registration 패턴

등록 코드(`Registry.register(...)`)는 **모듈 레벨**에 위치한다. 모듈이 import되는 시점에 자동 등록된다. Registry의 내부 구조:

```python
# mmp/components/base_registry.py
class BaseRegistry(Generic[T]):
    _registry: Dict[str, Type[T]]
    _base_class: Type[T] = None

    @classmethod
    def register(cls, key: str, klass: Type[T]) -> None:
        if cls._base_class is not None:
            if not issubclass(klass, cls._base_class):
                raise TypeError(...)
        cls._registry[key] = klass

    @classmethod
    def get_class(cls, key: str) -> Type[T]:
        return cls._registry[key]

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> T:
        return cls.get_class(key)(*args, **kwargs)
```

각 컴포넌트 Registry는 이를 상속:

```python
class PreprocessorStepRegistry(BaseRegistry[BasePreprocessor]):
    _registry: Dict[str, Type[BasePreprocessor]] = {}
    _base_class = BasePreprocessor
```

---

## 4. 새 어댑터 추가

### 파일 생성: `mmp/components/adapter/modules/mongo_adapter.py`

```python
from mmp.components.adapter.base import BaseAdapter
from mmp.components.adapter.registry import AdapterRegistry

class MongoAdapter(BaseAdapter):
    def read(self, source, **kwargs):
        return pd.DataFrame(...)

    def write(self, df, target, **kwargs):
        pass

AdapterRegistry.register("mongo", MongoAdapter)
```

### Config YAML

```yaml
data_source:
  adapter_type: "mongo"
  config:
    uri: "mongodb://localhost:27017"
```

---

## 5. 컴포넌트 목록

| 컴포넌트 | 역할 | Base 클래스 | Registry | 등록 필요 |
|----------|------|------------|----------|----------|
| **Model** | 학습/추론 | `mmp.models.base.BaseModel` | - | X (`class_path` 직접 로드) |
| **Adapter** | 데이터 I/O | `mmp.components.adapter.base` | `AdapterRegistry` | O |
| **Preprocessor** | 전처리 스텝 | `mmp.components.preprocessor.base` | `PreprocessorStepRegistry` | O |
| **Fetcher** | 피처 추가 조회 | `mmp.components.fetcher.base` | `FetcherRegistry` | O |
| **Evaluator** | 성능 평가 | `mmp.components.evaluator.base` | `EvaluatorRegistry` | O |
| **Calibrator** | 확률 보정 | `mmp.components.calibration.base` | `CalibrationRegistry` | O |

---

프로젝트 전체 구조와 스키마 레퍼런스는 `AGENT.md` 참조.
