# 🚀 **Task_Type 제거 및 Catalog 기반 완전 전환 개발 계획**

## 🎯 **개발 목표**
1. Recipe Schema에서 `task_type` 완전 제거 → `task_choice` 활용
2. Catalog에서 `supported_tasks` 제거 → `data_handler` 필드로 일원화
3. 모든 분기 로직을 `task_choice` 기반으로 통일

---

## 📋 **Phase 1: Recipe Schema 개선** (1-2일)

### **1.1 Recipe Schema 수정**
```python
# src/settings/recipe.py

class Recipe(BaseModel):
    """Recipe 최상위에 task_choice 추가"""
    name: str
    task_choice: Literal["classification", "regression", "clustering", "causal", "timeseries"] = Field(
        ..., 
        description="사용자가 Recipe Builder에서 선택한 ML 태스크"
    )
    
    data: DataSection
    model: ModelSection
    validation: ValidationSection
    preprocessor: Optional[PreprocessorSection] = None

class DataInterface(BaseModel):
    """task_type 필드 완전 제거"""
    target_column: str = Field(..., description="타겟 컬럼 이름")
    
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="피처 컬럼 목록 (None이면 target, treatment, entity 제외 모든 컬럼 사용)"
    )
    
    treatment_column: Optional[str] = Field(
        None, 
        description="처치 변수 컬럼 (causal task에서만 사용)"
    )
    
    entity_columns: List[str] = Field(..., description="엔티티 컬럼 목록")
    timestamp_column: Optional[str] = Field(None, description="시계열 타임스탬프 컬럼")
    
    # task_type 필드 제거!
    
    @model_validator(mode='after')
    def validate_task_specific_fields(self):
        """task_choice 기반 검증으로 변경"""
        # 이 로직은 Recipe 레벨에서 처리하도록 이동
        return self
```

### **1.2 Recipe Validation 로직 이동**
```python
# src/settings/recipe.py

class Recipe(BaseModel):
    # ... 기존 필드들
    
    @model_validator(mode='after')
    def validate_task_choice_compatibility(self):
        """task_choice와 다른 설정들의 호환성 검증"""
        task = self.task_choice
        data_interface = self.data.data_interface
        
        # Timeseries task 검증
        if task == "timeseries":
            if not data_interface.timestamp_column:
                raise ValueError("Timeseries task에는 timestamp_column이 필수입니다")
        
        # Causal task 검증  
        if task == "causal":
            if not data_interface.treatment_column:
                raise ValueError("Causal task에는 treatment_column이 필수입니다")
        
        return self
```

---

## 📋 **Phase 2: Model Catalog 일원화** (1-2일)

### **2.1 모든 Catalog YAML 수정**
**수정할 파일들:**
- `src/models/catalog/**/*.yaml` (33개 파일)

**수정 내용:**
```yaml
# 기존
supported_tasks: ["binary_classification", "multiclass_classification"]
feature_requirements:
  numerical: true
  categorical: true
  text: false

# 개선 (supported_tasks 제거, data_handler 추가)
data_handler: "tabular"  # 또는 "deeplearning", "timeseries"
feature_requirements:
  numerical: true
  categorical: true
  text: false
```

**매핑 규칙:**
- Classification/*.yaml → `data_handler: "tabular"`
- Regression/*.yaml → `data_handler: "tabular"`  
- Clustering/*.yaml → `data_handler: "tabular"`
- Causal/*.yaml → `data_handler: "tabular"`
- Timeseries/*.yaml → `data_handler: "timeseries"`
- DeepLearning/*.yaml → `data_handler: "deeplearning"`

### **2.2 Catalog 수정 스크립트**
```python
# scripts/update_catalog_schema.py
import os
import yaml
from pathlib import Path

def update_catalog_files():
    """모든 catalog 파일에서 supported_tasks 제거하고 data_handler 추가"""
    catalog_root = Path("src/models/catalog")
    
    # 디렉토리별 data_handler 매핑
    handler_mapping = {
        "Classification": "tabular",
        "Regression": "tabular", 
        "Clustering": "tabular",
        "Causal": "tabular",
        "Timeseries": "timeseries",
        "DeepLearning": "deeplearning"
    }
    
    for task_dir in catalog_root.iterdir():
        if task_dir.is_dir() and task_dir.name in handler_mapping:
            handler_type = handler_mapping[task_dir.name]
            
            for yaml_file in task_dir.glob("*.yaml"):
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # supported_tasks 제거
                if 'supported_tasks' in data:
                    del data['supported_tasks']
                
                # data_handler 추가
                data['data_handler'] = handler_type
                
                with open(yaml_file, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                
                print(f"✅ Updated: {yaml_file}")

if __name__ == "__main__":
    update_catalog_files()
```

---

## 📋 **Phase 3: DataHandler Registry 단순화** (반나절)

### **3.1 Registry 로직 단순화**
```python
# src/components/datahandler/registry.py

@classmethod
def get_handler_for_task(cls, task_choice: str, settings, model_class_path: str = None) -> BaseDataHandler:
    """
    Model catalog 기반 DataHandler 선택 (task_choice는 호환성 검증용으로만 사용)
    
    Args:
        task_choice: Recipe의 task_choice (검증용)
        settings: Settings 인스턴스  
        model_class_path: 모델 클래스 경로
    """
    # 🔍 모델 catalog에서 data_handler 정보 추출
    catalog_handler = cls._get_data_handler_from_catalog(model_class_path)
    
    if catalog_handler in cls.handlers:
        # 📋 Task와 Handler 호환성 검증 (선택사항)
        cls._validate_task_handler_compatibility(task_choice, catalog_handler)
        
        logger.info(f"🧠 Catalog 기반 핸들러 선택: {catalog_handler} (task: {task_choice})")
        return cls.create(catalog_handler, settings)
    
    available = list(cls.handlers.keys())
    raise ValueError(f"지원하지 않는 data_handler: '{catalog_handler}'. 사용 가능한 핸들러: {available}")

@classmethod 
def _get_data_handler_from_catalog(cls, model_class_path: str) -> str:
    """모델 catalog에서 data_handler 추출"""
    if not model_class_path:
        return "tabular"  # 기본값
        
    catalog = cls._load_model_catalog(model_class_path)
    if catalog and 'data_handler' in catalog:
        handler = catalog['data_handler']
        logger.debug(f"📋 Catalog에서 data_handler 발견: {handler}")
        return handler
    
    # Fallback: 기본값
    logger.debug(f"📋 Catalog에 data_handler가 없어 기본값 사용: tabular")
    return "tabular"

@classmethod
def _validate_task_handler_compatibility(cls, task_choice: str, handler_type: str):
    """Task와 Handler 호환성 검증 (선택사항)"""
    # 예: timeseries task인데 tabular handler 사용 시 경고
    if task_choice == "timeseries" and handler_type == "tabular":
        logger.warning("⚠️ Timeseries task에 tabular handler 사용. 의도한 것이 맞나요?")
```

---

## 📋 **Phase 4: 시스템 전반 task_type 제거** (1-2일)

### **4.1 Factory 수정**
```python
# src/factory/factory.py

def create_datahandler(self) -> Any:
    """DataHandler 생성 (task_choice 활용)"""
    # ...기존 캐싱 로직
    
    # 모델 클래스 경로 추출
    model_class_path = getattr(self._recipe.model, 'class_path', None)
    
    # task_choice 활용
    task_choice = self._recipe.task_choice
    
    # Registry 패턴으로 catalog 기반 핸들러 선택
    datahandler = DataHandlerRegistry.get_handler_for_task(
        task_choice, 
        self.settings, 
        model_class_path=model_class_path
    )

def create_evaluator(self) -> Any:
    """Evaluator 생성 (task_choice 활용)"""
    # ...기존 캐싱 로직
    
    # task_choice 활용
    task_choice = self._recipe.task_choice
    data_interface = self._recipe.data.data_interface
    
    evaluator = EvaluatorRegistry.create(task_choice, data_interface)
```

### **4.2 Trainer 수정**
```python
# src/components/trainer/trainer.py

def _fit_model(self, model, X, y, additional_data):
    """task_choice에 따라 모델 학습"""
    task_choice = self.settings.recipe.task_choice  # task_type → task_choice
    
    if task_choice in ["classification", "regression"]:
        model.fit(X, y)
    elif task_choice == "clustering":
        model.fit(X)
    elif task_choice == "causal":
        model.fit(X, additional_data['treatment'], y)
    elif task_choice == "timeseries":
        model.fit(X, y)
    else:
        raise ValueError(f"지원하지 않는 task_choice: {task_choice}")

def _get_training_methodology(self):
    """학습 방법론 메타데이터 (task_choice 활용)"""
    task_choice = self.settings.recipe.task_choice  # task_type → task_choice
    
    return {
        # ...기존 필드들
        'task_choice': task_choice,  # task_type → task_choice
        # ...
    }

def _get_stratify_col(self):
    """Stratification 컬럼 결정"""
    di = self.settings.recipe.data.data_interface
    task_choice = self.settings.recipe.task_choice
    
    if task_choice == "classification":
        return di.target_column
    elif task_choice == "causal":
        return di.treatment_column
    else:
        return None
```

### **4.3 Trainer Data Handler 모듈 수정**
```python
# src/components/trainer/modules/data_handler.py

def prepare_training_data(df: pd.DataFrame, settings: Settings) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """동적 데이터 준비 (task_choice 활용)"""
    data_interface = settings.recipe.data.data_interface
    task_choice = settings.recipe.task_choice  # task_type → task_choice
    exclude_cols = _get_exclude_columns(settings, df)
    
    if task_choice in ["classification", "regression"]:
        # ...기존 로직
    elif task_choice == "clustering":
        # ...기존 로직  
    elif task_choice == "causal":
        # ...기존 로직
    else:
        raise ValueError(f"지원하지 않는 task_choice: {task_choice}")

def _determine_stratify_split(df: pd.DataFrame, data_interface) -> Optional[pd.Series]:
    """Stratify 분할 결정 (task_choice 활용)"""
    # 이 함수는 settings를 받도록 수정하여 task_choice 접근
    pass
```

---

## 📋 **Phase 5: Recipe Builder 및 Templates 수정** (1일)

### **5.1 Recipe Template 수정**
```yaml
# src/cli/templates/recipes/recipe.yaml.j2

name: "{{ recipe_name }}"
task_choice: "{{ task_choice }}"  # ✅ 새로 추가

data:
  loader:
    source_uri: "{{ data_source_uri }}"
  
  data_interface:
    # task_type: "{{ task_type }}"  # ❌ 제거
    target_column: "{{ target_column }}"
    entity_columns: {{ entity_columns }}
    {% if timestamp_column %}
    timestamp_column: "{{ timestamp_column }}"
    {% endif %}
    {% if treatment_column %}
    treatment_column: "{{ treatment_column }}"
    {% endif %}

model:
  class_path: "{{ model_class_path }}"
  hyperparameters:
    # ...
```

### **5.2 Recipe Builder 로직 수정**
```python
# tests/helpers/recipe_builder.py 또는 해당 Builder 코드

def build_recipe_config(task_choice: str, model_name: str, **kwargs):
    """Recipe Builder에서 task_choice를 최상위로 설정"""
    
    recipe_data = {
        "name": kwargs.get("name", f"{task_choice}_{model_name}_recipe"),
        "task_choice": task_choice,  # ✅ 최상위에 설정
        
        "data": {
            "loader": {
                "source_uri": kwargs.get("source_uri", "data.csv")
            },
            "data_interface": {
                # "task_type": task_choice,  # ❌ 제거
                "target_column": kwargs.get("target_column", "target"),
                "entity_columns": kwargs.get("entity_columns", ["id"]),
            }
        },
        
        "model": {
            "class_path": get_model_class_path(model_name),
            "hyperparameters": kwargs.get("hyperparameters", {})
        }
    }
    
    # Task별 특수 필드 추가
    if task_choice == "timeseries":
        recipe_data["data"]["data_interface"]["timestamp_column"] = kwargs.get("timestamp_column")
    elif task_choice == "causal":
        recipe_data["data"]["data_interface"]["treatment_column"] = kwargs.get("treatment_column")
    
    return recipe_data
```

---

## 📋 **Phase 6: 테스트 코드 업데이트** (1일)

### **6.1 모든 테스트에서 task_type → task_choice 변경**
**수정 대상 파일들:**
- `tests/unit/components/test_datahandler/*.py`
- `tests/unit/components/test_trainer/*.py`
- `tests/unit/components/test_evaluator/*.py`
- `tests/unit/factory/*.py`
- `tests/integration/*.py`

**수정 예시:**
```python
# 기존
recipe_data = {
    "data": {
        "data_interface": {
            "task_type": "classification"  # ❌
        }
    }
}

# 수정
recipe_data = {
    "task_choice": "classification",  # ✅ 최상위로 이동
    "data": {
        "data_interface": {
            # task_type 제거
        }
    }
}
```

---

## 📋 **Phase 7: Legacy 호환성 및 마이그레이션** (반나일)

### **7.1 기존 Recipe 파일 자동 마이그레이션**
```python
# scripts/migrate_existing_recipes.py

def migrate_recipe_file(recipe_path: Path):
    """기존 recipe 파일을 새 스키마로 마이그레이션"""
    with open(recipe_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # task_type을 task_choice로 이동
    if 'data' in data and 'data_interface' in data['data']:
        task_type = data['data']['data_interface'].get('task_type')
        if task_type:
            data['task_choice'] = task_type
            del data['data']['data_interface']['task_type']
    
    # 백업 생성 후 덮어쓰기
    backup_path = recipe_path.with_suffix('.yaml.backup')
    shutil.copy(recipe_path, backup_path)
    
    with open(recipe_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✅ Migrated: {recipe_path}")
```

### **7.2 Deprecated Warning 추가**
```python
# src/settings/recipe.py

class DataInterface(BaseModel):
    # ...기존 필드들
    
    # 임시 호환성을 위한 deprecated 필드
    task_type: Optional[str] = Field(
        None, 
        description="DEPRECATED: Use recipe.task_choice instead", 
        deprecated=True
    )
    
    @model_validator(mode='after')
    def warn_deprecated_task_type(self):
        if self.task_type:
            import warnings
            warnings.warn(
                "task_type field is deprecated. Use recipe.task_choice instead.", 
                DeprecationWarning, 
                stacklevel=2
            )
        return self
```

---

## 📋 **Phase 8: 통합 테스트 및 검증** (1일)

### **8.1 End-to-End 테스트**
```python
# tests/integration/test_task_choice_integration.py

def test_classification_task_with_tabular_model():
    """Classification task + sklearn 모델 테스트"""
    recipe_data = {
        "name": "test_classification",
        "task_choice": "classification",  # ✅ 최상위 설정
        "data": {
            "data_interface": {
                "target_column": "target",
                "entity_columns": ["id"]
            }
        },
        "model": {
            "class_path": "sklearn.ensemble.RandomForestClassifier"
        }
    }
    
    # Factory 생성 및 컴포넌트 테스트
    settings = Settings(recipe=Recipe(**recipe_data))
    factory = Factory(settings)
    
    # DataHandler: catalog에서 "tabular" 선택되는지 확인
    datahandler = factory.create_datahandler()
    assert isinstance(datahandler, TabularDataHandler)
    
    # Evaluator: task_choice 기반 선택되는지 확인  
    evaluator = factory.create_evaluator()
    assert isinstance(evaluator, ClassificationEvaluator)

def test_timeseries_task_with_deeplearning_model():
    """Timeseries task + LSTM 모델 테스트"""
    recipe_data = {
        "name": "test_timeseries",
        "task_choice": "timeseries",  # ✅ 최상위 설정
        "data": {
            "data_interface": {
                "target_column": "value",
                "entity_columns": ["id"],
                "timestamp_column": "timestamp"
            }
        },
        "model": {
            "class_path": "src.models.custom.lstm_timeseries.LSTMTimeSeries"
        }
    }
    
    settings = Settings(recipe=Recipe(**recipe_data))
    factory = Factory(settings)
    
    # DataHandler: catalog에서 "deeplearning" 선택되는지 확인
    datahandler = factory.create_datahandler()
    assert isinstance(datahandler, DeepLearningDataHandler)
    
    # Evaluator: task_choice 기반 선택되는지 확인
    evaluator = factory.create_evaluator()
    assert isinstance(evaluator, TimeSeriesEvaluator)
```

### **8.2 Backward Compatibility 테스트**
```python
def test_deprecated_task_type_still_works():
    """Deprecated task_type 필드가 여전히 동작하는지 확인"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        recipe_data = {
            "name": "test_legacy",
            "task_choice": "classification", 
            "data": {
                "data_interface": {
                    "task_type": "classification",  # ❌ Deprecated
                    "target_column": "target",
                    "entity_columns": ["id"]
                }
            }
        }
        
        settings = Settings(recipe=Recipe(**recipe_data))
        
        # Warning이 발생했는지 확인
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message)
```

---

## 🚀 **실행 순서 및 타임라인**

**Week 1:**
- Phase 1: Recipe Schema 개선 (1-2일)
- Phase 2: Model Catalog 일원화 (1-2일)  
- Phase 3: DataHandler Registry 단순화 (반나절)

**Week 2:**
- Phase 4: 시스템 전반 task_type 제거 (1-2일)
- Phase 5: Recipe Builder 및 Templates 수정 (1일)
- Phase 6: 테스트 코드 업데이트 (1일)

**Week 3:**
- Phase 7: Legacy 호환성 및 마이그레이션 (반나절)
- Phase 8: 통합 테스트 및 검증 (1일)
- 🎉 **완료 및 배포**

## ✅ **최종 검증 체크리스트**

- [ ] Recipe Builder에서 task_choice 한 번만 선택
- [ ] 모든 catalog에 data_handler 필드 존재
- [ ] supported_tasks 필드 완전 제거
- [ ] DataInterface에서 task_type 필드 제거/Deprecated
- [ ] 모든 시스템이 task_choice 기반으로 동작
- [ ] 기존 recipe 파일 마이그레이션 완료
- [ ] End-to-End 테스트 통과

**결과: 사용자는 Recipe Builder에서 Task 한 번, Model 한 번만 선택하면 모든 것이 자동으로 연결되는 완전한 시스템!** 🎯