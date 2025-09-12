# 📊 Probability Calibration Module Implementation Plan

## 🎯 목표
분류 태스크에서 예측 확률의 보정(Calibration)을 지원하는 모듈을 추가하여, 더 신뢰할 수 있는 확률 예측값을 제공합니다.

## 🏗️ 아키텍처 설계

### 1. 디렉토리 구조
```
src/interface/
├── __init__.py
└── base_calibrator.py       # 베이스 인터페이스 (새로 추가)

src/components/calibration/
├── __init__.py
├── registry.py              # Calibration 레지스트리
└── modules/
    ├── __init__.py
    ├── platt_scaling.py     # Platt Scaling (Binary)
    ├── beta_calibration.py  # Beta Calibration (Binary) 
    ├── isotonic_regression.py # Isotonic Regression (Binary/Multi)
    └── temperature_scaling.py # Temperature Scaling (Multi-class)
```

### 2. 데이터 분할 전략 (Data Leakage 방지)
```yaml
# Recipe 구조
data:
  split:
    train: 0.7      # 모델 학습용
    test: 0.1       # 최종 평가용
    validation: 0.1 # 하이퍼파라미터 튜닝용
    calibration: 0.1 # 확률 보정용 (classification only)
```

**Data Leakage 방지 원칙:**
- Calibration 데이터는 학습/검증에 사용되지 않음
- 모델 학습 → Calibration fitting → Test 평가 순서 엄격 준수
- Preprocessor는 train 데이터로만 fit

## 📝 구현 상세

### 1. **Base Calibrator Interface** (`src/interface/base_calibrator.py`)
```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple

class BaseCalibrator(ABC):
    """확률 보정 베이스 인터페이스"""
    
    @abstractmethod
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'BaseCalibrator':
        """보정 모델 학습"""
        pass
    
    @abstractmethod
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """확률값 보정"""
        pass
    
    @abstractmethod
    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """학습 후 변환"""
        pass
    
    @property
    @abstractmethod
    def supports_multiclass(self) -> bool:
        """다중 클래스 지원 여부"""
        pass
```

### 2. **Calibration Registry** (`src/components/calibration/registry.py`)
```python
from typing import Dict, Type
from src.interface import BaseCalibrator

class CalibrationRegistry:
    """Calibration 메서드 레지스트리"""
    calibrators: Dict[str, Type[BaseCalibrator]] = {}
    
    @classmethod
    def register(cls, name: str, calibrator_class: Type[BaseCalibrator]):
        """보정 메서드 등록"""
        if not issubclass(calibrator_class, BaseCalibrator):
            raise TypeError(f"{calibrator_class.__name__} must inherit BaseCalibrator")
        cls.calibrators[name] = calibrator_class
    
    @classmethod
    def create(cls, method: str, **kwargs) -> BaseCalibrator:
        """보정 메서드 인스턴스 생성"""
        if method not in cls.calibrators:
            available = list(cls.calibrators.keys())
            raise ValueError(f"Unknown calibration method: {method}. Available: {available}")
        return cls.calibrators[method](**kwargs)
```

### 3. **Calibration Modules**

#### Platt Scaling (`modules/platt_scaling.py`)
```python
from sklearn.linear_model import LogisticRegression
from src.interface import BaseCalibrator
from ..registry import CalibrationRegistry

class PlattScaling(BaseCalibrator):
    """Sigmoid 함수를 이용한 확률 보정 (Binary Classification)"""
    
    def __init__(self):
        self.calibrator = LogisticRegression()
        self._supports_multiclass = False
    
    def fit(self, y_prob, y_true):
        # Binary classification만 지원
        if len(np.unique(y_true)) > 2:
            raise ValueError("Platt Scaling은 이진 분류만 지원합니다")
        self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        return self
    
    def transform(self, y_prob):
        return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    
    @property
    def supports_multiclass(self):
        return self._supports_multiclass

# 자동 등록
CalibrationRegistry.register("platt", PlattScaling)
```

#### Isotonic Regression (`modules/isotonic_regression.py`)
```python
from sklearn.isotonic import IsotonicRegression
from src.interface import BaseCalibrator
from ..registry import CalibrationRegistry

class IsotonicCalibration(BaseCalibrator):
    """단조 회귀를 이용한 확률 보정"""
    
    def __init__(self):
        self.calibrator = None
        self._supports_multiclass = True
    
    def fit(self, y_prob, y_true):
        if len(y_prob.shape) == 1:  # Binary
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
        else:  # Multi-class (one-vs-rest)
            self.calibrator = []
            for i in range(y_prob.shape[1]):
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(y_prob[:, i], (y_true == i).astype(int))
                self.calibrator.append(iso)
        return self
    
    def transform(self, y_prob):
        if isinstance(self.calibrator, list):  # Multi-class
            calibrated = np.zeros_like(y_prob)
            for i, iso in enumerate(self.calibrator):
                calibrated[:, i] = iso.transform(y_prob[:, i])
            # Normalize to sum to 1
            return calibrated / calibrated.sum(axis=1, keepdims=True)
        else:  # Binary
            return self.calibrator.transform(y_prob)
    
    @property
    def supports_multiclass(self):
        return self._supports_multiclass

# 자동 등록
CalibrationRegistry.register("isotonic", IsotonicCalibration)
```

### 4. **Recipe 수정** (`src/cli/templates/recipes/recipe.yaml.j2`)
```yaml
# Data split configuration
data:
  split:
    train: {{ train_ratio }}
    test: {{ test_ratio }}
    validation: {{ validation_ratio }}
    {% if task|lower == 'classification' and calibration_enabled %}
    calibration: {{ calibration_ratio }}
    {% endif %}

# Model configuration  
model:
  class_path: {{ model_class }}
  library: {{ model_library }}
  
  {% if task|lower == 'classification' %}
  calibration:
    enabled: {{ calibration_enabled | lower }}
    {% if calibration_enabled %}
    method: {{ calibration_method }}  # platt, isotonic, beta, temperature
    {% endif %}
  {% endif %}
```

### 5. **Recipe Builder 수정** (`src/cli/utils/recipe_builder.py`)
```python
def _configure_data_split(self, task: str) -> Dict[str, float]:
    """데이터 분할 비율 설정"""
    
    if task.lower() == 'classification':
        use_calibration = self.ui.confirm(
            "확률 보정(Calibration)을 사용하시겠습니까?",
            default=True
        )
        
        if use_calibration:
            # Calibration 포함 분할 (기본값: 0.7/0.1/0.1/0.1)
            splits = self._get_split_ratios_with_calibration()
        else:
            # 일반 분할 (기본값: 0.8/0.1/0.1)
            splits = self._get_standard_split_ratios()
    else:
        # Classification이 아닌 경우 표준 분할
        splits = self._get_standard_split_ratios()
    
    # 합이 1.0인지 검증
    total = sum(splits.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"분할 비율의 합이 1.0이 아닙니다: {total}")
    
    return splits

def _configure_calibration(self) -> Dict[str, Any]:
    """Calibration 설정"""
    methods = {
        'platt': 'Platt Scaling (Binary only)',
        'isotonic': 'Isotonic Regression',
        'beta': 'Beta Calibration (Binary)',
        'temperature': 'Temperature Scaling (Multi-class)'
    }
    
    self.ui.show_table(
        "Calibration Methods",
        ["Method", "Description", "Support"],
        [
            ["platt", "Sigmoid-based", "Binary"],
            ["isotonic", "Non-parametric", "Binary/Multi"],
            ["beta", "Beta distribution", "Binary"],
            ["temperature", "Single parameter", "Multi-class"]
        ]
    )
    
    method = self.ui.select_from_list(
        "Calibration 방법을 선택하세요",
        list(methods.keys())
    )
    
    return {
        'enabled': True,
        'method': method
    }
```

### 6. **Settings Validator 수정** (`src/settings/validator.py`)
```python
def validate_split_configuration(recipe: Dict[str, Any]) -> None:
    """데이터 분할 설정 검증"""
    
    split_config = recipe.get('data', {}).get('split', {})
    task = recipe.get('task_choice', '')
    calibration = recipe.get('model', {}).get('calibration', {})
    
    # 기본값 설정
    if not split_config:
        if task == 'classification' and calibration.get('enabled'):
            split_config = {'train': 0.7, 'test': 0.1, 'validation': 0.1, 'calibration': 0.1}
        else:
            split_config = {'train': 0.8, 'test': 0.1, 'validation': 0.1}
    
    # 합이 1인지 검증
    total = sum(split_config.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"분할 비율의 합이 1.0이 아닙니다: {total}")
    
    # Calibration 검증
    if task == 'classification':
        if calibration.get('enabled'):
            if 'calibration' not in split_config or split_config['calibration'] <= 0:
                raise ValueError("Calibration이 활성화되었지만 calibration 분할이 설정되지 않았습니다")
    else:
        if calibration.get('enabled'):
            raise ValueError(f"{task} 태스크는 calibration을 지원하지 않습니다")
        if 'calibration' in split_config and split_config['calibration'] > 0:
            raise ValueError(f"{task} 태스크에는 calibration 분할을 사용할 수 없습니다")
```

### 7. **DataHandler 수정** (`src/components/datahandler/modules/tabular_handler.py`)
```python
def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """데이터를 train/test/validation/calibration으로 분할"""
    
    split_config = self.settings.recipe.data.split
    task = self.settings.recipe.task_choice
    
    # 분할 비율 가져오기
    train_ratio = split_config.get('train', 0.8)
    test_ratio = split_config.get('test', 0.1)
    val_ratio = split_config.get('validation', 0.1)
    calib_ratio = split_config.get('calibration', 0)
    
    # Stratify 설정
    stratify = None
    if task == 'classification':
        target_col = self.data_interface.target_column
        if self._can_stratify(df[target_col]):
            stratify = df[target_col]
    
    # 순차적 분할
    # 1. Train+Val+Calib vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        df, test_size=test_ratio, random_state=42, stratify=stratify
    )
    
    # 2. Train+Val vs Calib (if needed)
    if calib_ratio > 0:
        calib_size = calib_ratio / (1 - test_ratio)
        X_temp2, X_calib, y_temp2, y_calib = train_test_split(
            X_temp, test_size=calib_size, random_state=42, 
            stratify=y_temp if stratify is not None else None
        )
    else:
        X_temp2, X_calib = X_temp, None
    
    # 3. Train vs Val
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp2, test_size=val_size, random_state=42,
        stratify=y_temp2 if stratify is not None else None
    )
    
    return {
        'train': X_train,
        'validation': X_val,
        'test': X_test,
        'calibration': X_calib  # None if not classification
    }
```

### 8. **Training Pipeline 수정** (`src/pipelines/train_pipeline.py`)
```python
def run_training_pipeline(settings: Settings):
    """학습 파이프라인 with Calibration"""
    
    # ... 기존 코드 ...
    
    # 데이터 분할
    splits = datahandler.split_data(df)
    
    # 모델 학습
    trained_model, results = trainer.train(
        X_train=splits['train']['X'],
        y_train=splits['train']['y'],
        X_val=splits['validation']['X'],
        y_val=splits['validation']['y'],
        model=model
    )
    
    # Calibration (Classification only)
    calibrator = None
    if (settings.recipe.task_choice == 'classification' and 
        settings.recipe.model.calibration.enabled and
        splits['calibration'] is not None):
        
        # Calibration 데이터로 예측 (Data Leakage 방지)
        X_calib = splits['calibration']['X'] 
        y_calib = splits['calibration']['y']
        y_prob_calib = trained_model.predict_proba(X_calib)
        
        # Calibrator 학습 및 직렬화를 위한 상태 저장
        method = settings.recipe.model.calibration.method
        calibrator = CalibrationRegistry.create(method)
        
        if y_prob_calib.shape[1] == 2:  # Binary
            calibrator.fit(y_prob_calib[:, 1], y_calib)
        else:  # Multi-class
            if not calibrator.supports_multiclass:
                raise ValueError(f"{method}는 다중 클래스를 지원하지 않습니다")
            calibrator.fit(y_prob_calib, y_calib)
    
    # PyfuncWrapper에 calibrator 포함 (MLflow 저장을 위해 직렬화 가능해야 함)
    pyfunc_model = PyfuncWrapper(
        settings=settings,
        trained_model=trained_model,
        trained_calibrator=calibrator,  # 재현성을 위해 calibrator 자체 저장
        # ... 기타 파라미터
    )
```

### 9. **PyfuncWrapper 수정** (`src/utils/integrations/pyfunc_wrapper.py`)
```python
class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, ..., trained_calibrator=None):
        # ... 기존 코드 ...
        self.trained_calibrator = trained_calibrator  # 재현성을 위해 calibrator 객체 저장
        self.calibration_enabled = (
            self._task_type == 'classification' and
            trained_calibrator is not None  # calibrator가 실제로 학습되었는지 확인
        )
    
    def predict(self, context, model_input, params=None):
        # ... 기존 예측 코드 ...
        
        # Classification with probability
        if self._task_type == "classification":
            predictions = self.trained_model.predict(X)
            
            if hasattr(self.trained_model, 'predict_proba'):
                probabilities = self.trained_model.predict_proba(X)
                
                # Calibration 적용
                apply_calibration = self._should_apply_calibration(params)
                if apply_calibration and self.trained_calibrator:
                    if probabilities.shape[1] == 2:  # Binary
                        probabilities[:, 1] = self.trained_calibrator.transform(
                            probabilities[:, 1]
                        )
                        probabilities[:, 0] = 1 - probabilities[:, 1]
                    else:  # Multi-class
                        probabilities = self.trained_calibrator.transform(probabilities)
                
                # DataFrame 생성
                if probabilities.shape[1] == 2:
                    result_df = pd.DataFrame({
                        'prediction': predictions,
                        'prediction_proba': probabilities[:, 1]
                    })
                else:
                    result_df = pd.DataFrame({
                        'prediction': predictions,
                        'prediction_proba': probabilities.tolist()
                    })
                
                return result_df
    
    def _should_apply_calibration(self, params):
        """Calibration 적용 여부 결정 (재현성 보장)"""
        if params and 'calibration' in params:
            # 추론 시 명시적 설정이 있으면 우선 (추론 시 오버라이드)
            explicit_setting = params['calibration']
            if explicit_setting and not self.trained_calibrator:
                # Calibrator가 없는데 적용 요청한 경우 경고
                self.console.warning("Calibrator가 학습되지 않았습니다. Calibration을 건너뜁니다.")
                return False
            return explicit_setting
        else:
            # 기본값: 학습된 calibrator가 있으면 적용
            return self.trained_calibrator is not None
```

### 10. **Inference Command 수정** (`src/cli/commands/inference_command.py`)
```python
def batch_inference_command(
    run_id: str,
    config_path: str,
    data_path: str,
    context_params: Optional[str] = None,
    calibration: Optional[bool] = typer.Option(
        None,
        "--calibration/--no-calibration",
        help="확률 보정 적용 여부 (기본: 학습 시 설정 따름)"
    )
) -> None:
    """배치 추론 with Calibration 옵션"""
    
    # ... 기존 코드 ...
    
    # Context params에 calibration 설정 추가
    if calibration is not None:
        params['calibration'] = calibration
    
    run_inference_pipeline(
        settings=settings,
        run_id=run_id,
        data_path=data_path,
        context_params=params
    )
```

### 11. **format_predictions 수정** (`src/utils/data/data_io.py`)
```python
def format_predictions(predictions_result, original_df, data_interface=None):
    """예측 결과 포맷팅 (확률값 포함)"""
    
    # DataFrame 변환
    if isinstance(predictions_result, pd.DataFrame):
        pred_df = predictions_result
        # prediction과 prediction_proba 컬럼 모두 유지
    else:
        # 기존 로직 (단순 예측값만)
        pred_df = pd.DataFrame({'prediction': predictions_result})
    
    # Entity columns 추가
    # ... 기존 코드 ...
    
    return pred_df
```

## 🧪 테스트 계획

### 1. **Unit Tests**

#### `test_calibration_registry.py`
```python
def test_calibration_registry():
    """레지스트리 등록 및 생성 테스트"""
    # Given: Calibrator 클래스
    # When: 레지스트리에 등록
    # Then: 정상 생성 확인

def test_invalid_calibrator_registration():
    """잘못된 Calibrator 등록 시 오류"""
    # Given: BaseCalibrator를 상속받지 않은 클래스
    # When: 등록 시도
    # Then: TypeError 발생
```

#### `test_platt_scaling.py`
```python
def test_platt_scaling_binary():
    """이진 분류 Platt Scaling 테스트"""
    # Given: 이진 분류 확률값
    # When: Calibration 적용
    # Then: 보정된 확률값 확인

def test_platt_scaling_multiclass_error():
    """다중 클래스에서 오류 발생 테스트"""
    # Given: 다중 클래스 확률값
    # When: Platt Scaling 적용 시도
    # Then: ValueError 발생
```

#### `test_isotonic_regression.py`
```python
def test_isotonic_binary():
    """이진 분류 Isotonic Regression 테스트"""
    
def test_isotonic_multiclass():
    """다중 클래스 Isotonic Regression 테스트"""
    # Given: 다중 클래스 확률값
    # When: Calibration 적용
    # Then: 각 클래스별 보정 및 정규화 확인
```

### 2. **Integration Tests**

#### `test_calibration_in_training.py`
```python
def test_training_with_calibration():
    """Calibration을 포함한 학습 파이프라인 테스트"""
    # Given: Classification 설정 with calibration
    # When: 학습 파이프라인 실행
    # Then: Calibrator가 학습되고 저장됨

def test_data_leakage_prevention():
    """Data Leakage 방지 테스트"""
    # Given: 4-way split 데이터
    # When: 각 단계별 데이터 사용
    # Then: Calibration 데이터가 학습에 사용되지 않음 확인
```

#### `test_inference_with_calibration.py`
```python
def test_inference_default_calibration():
    """기본 calibration 설정으로 추론"""
    # Given: Calibration으로 학습된 모델
    # When: 추론 실행 (옵션 없음)
    # Then: 학습 시 설정대로 calibration 적용

def test_inference_override_calibration():
    """Calibration 설정 오버라이드 테스트"""
    # Given: Calibration으로 학습된 모델
    # When: --no-calibration 옵션으로 추론
    # Then: Calibration 미적용 확인
```

### 3. **End-to-End Tests**

#### `test_classification_with_probability.py`
```python
def test_full_pipeline_with_probabilities():
    """분류 태스크 전체 파이프라인 테스트"""
    # Given: Classification recipe with calibration
    # When: Train → Inference → Save
    # Then: prediction과 prediction_proba 모두 저장됨
```

## 🚀 구현 우선순위

### Phase 1: Core Infrastructure (1-2일)
1. ✅ Base Calibrator Interface 생성 (`src/interface/base_calibrator.py`)
2. ✅ Calibration Registry 구현 (`src/components/calibration/registry.py`)
3. ✅ Recipe 스키마 수정 (data.split + model.calibration 섹션)
4. ✅ Settings Validator 수정 (분할 검증 로직)

### Phase 2: Calibration Methods (2-3일)
1. ✅ Platt Scaling 구현
2. ✅ Isotonic Regression 구현
3. ✅ Beta Calibration 구현
4. ✅ Temperature Scaling 구현

### Phase 3: Pipeline Integration (2-3일)
1. ✅ DataHandler split_data 수정
2. ✅ Training Pipeline 수정
3. ✅ PyfuncWrapper 수정
4. ✅ format_predictions 수정

### Phase 4: CLI Integration (1-2일)
1. ✅ Recipe Builder 수정
2. ✅ Inference Command 수정
3. ✅ 대화형 UI 업데이트

### Phase 5: Testing & Documentation (2일)
1. ✅ Unit Tests 작성
2. ✅ Integration Tests 작성
3. ✅ 문서화 및 예제 작성

## 📋 체크리스트

### 필수 요구사항
- [ ] **Data Leakage 방지** (calibration 데이터 분리)
- [ ] **재현성 보장** (PyfuncWrapper에 calibrator 객체 저장)
- [ ] **분할 비율 합 = 1.0 검증**
- [ ] **Classification 태스크에서만 calibration 활성화**
- [ ] **학습 시 설정과 추론 시 오버라이드 모두 지원**
- [ ] **Binary/Multi-class 모두 지원**
- [ ] **Base Calibrator를 src/interface/에 배치**

### 품질 기준
- [ ] 모든 calibration 메서드에 대한 테스트
- [ ] Registry 패턴 일관성
- [ ] 에러 메시지 명확성
- [ ] 문서화 완성도

## 🔍 예상 이슈 및 해결방안

### 1. Multi-class Calibration 복잡도
**문제**: 다중 클래스에서 각 클래스별 calibration 필요
**해결**: One-vs-Rest 방식으로 각 클래스별 calibrator 학습

### 2. 작은 데이터셋에서 4-way split
**문제**: 데이터가 너무 작게 분할됨
**해결**: 최소 샘플 수 검증 및 경고 메시지

### 3. Calibration 메서드 선택
**문제**: 사용자가 어떤 메서드를 선택해야 할지 모름
**해결**: Recipe Builder에서 가이드 제공 및 기본값 설정

## 📚 참고 자료
- [Scikit-learn Calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [Beta Calibration](https://github.com/betacal/python)