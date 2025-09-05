# 🔧 Recipe 구조 변경 완전 리팩토링 계획

> **실제 소스코드 기반 분석 결과**  
> 새로운 Recipe YAML 구조에 맞춰 시스템을 완전히 리팩토링하는 단계별 실행 계획입니다.

## 📋 변경 개요

### **현재 구조** → **새로운 구조**

```yaml
# 현재 (OLD)
data:
  loader:
    source_uri: "sql/train.sql"
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_timestamp
  fetcher:
    type: feature_store
    features:
      - feature_namespace: user_features
        features: [age, gender]
  data_interface:
    task_type: classification
    target_column: label
    id_column: user_id           # ❌ 제거

# 새로운 (NEW)
data:
  loader:
    source_uri: "sql/train.sql"  # entity_schema 제거
  fetcher:
    type: feature_store
    feature_views:               # ✅ 새로운 구조
      user_features:
        join_key: user_id
        features: [age, gender]
    timestamp_column: event_timestamp  # ✅ fetcher로 이동
  data_interface:
    task_type: classification
    target_column: label
    entity_columns: [user_id]    # ✅ id_column → entity_columns
    treatment_column: campaign   # ✅ causal 전용 추가
    feature_columns: null        # ✅ null = 자동 선택
```

---

## 🎯 Step 1: Schema 정의 수정 (Priority: 🔴 Critical)

### **파일: `src/settings/recipe.py`**

#### **1.1 새로운 클래스 추가**

```python
# line 95 근처에 추가
class FeatureView(BaseModel):
    """Feast FeatureView 정의 (개별 피처 그룹)"""
    join_key: str = Field(..., description="Join할 기준 컬럼 (user_id, item_id 등)")
    features: List[str] = Field(..., description="해당 FeatureView에서 가져올 피처 목록")
```

#### **1.2 기존 클래스 제거**

```python
# ❌ 완전 제거: EntitySchema 클래스 (line 69-73)
# class EntitySchema(BaseModel):
#     entity_columns: List[str] = ...
#     timestamp_column: str = ...

# ❌ 완전 제거: FeatureNamespace 클래스 (line 95-99)  
# class FeatureNamespace(BaseModel): ...
```

#### **1.3 Loader 클래스 수정**

```python
# line 75-93 수정
class Loader(BaseModel):
    """데이터 로더 설정"""
    source_uri: str = Field(..., description="데이터 소스 URI (SQL 파일 경로 또는 데이터 파일 경로)")
    # ❌ entity_schema 필드 완전 제거
    
    def get_adapter_type(self) -> str:
        # 기존 로직 유지
        ...
```

#### **1.4 Fetcher 클래스 완전 재작성**

```python
# line 101-123 완전 교체
class Fetcher(BaseModel):
    """피처 페처 설정 - Feature Store 통합"""
    type: Literal["feature_store", "pass_through"] = Field(..., description="페처 타입")
    
    # ✅ 새로운 구조: feature_views
    feature_views: Optional[Dict[str, FeatureView]] = Field(
        None, 
        description="Feast FeatureView 설정 (feature_store 타입에서 사용)"
    )
    
    # ✅ 새로운 필드: timestamp_column
    timestamp_column: Optional[str] = Field(
        None,
        description="Point-in-time join 기준 타임스탬프 컬럼"
    )
    
    @field_validator('feature_views')
    def validate_feature_views(cls, v, info):
        """feature_store 타입일 때 feature_views 검증"""
        if info.data.get('type') == 'feature_store':
            if not v:
                return {}  # 빈 dict 반환
        return v
```

#### **1.5 DataInterface 클래스 수정**

```python
# line 125-141 수정
class DataInterface(BaseModel):
    """데이터 인터페이스 설정"""
    task_type: Literal["classification", "regression", "clustering", "causal"] = Field(
        ..., 
        description="ML 태스크 타입"
    )
    target_column: str = Field(..., description="타겟 컬럼 이름")
    
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="피처 컬럼 목록 (None이면 target, treatment, entity 제외 모든 컬럼 사용)"
    )
    
    treatment_column: Optional[str] = Field(
        None, 
        description="처치 변수 컬럼 (causal task에서만 사용)"
    )
    
    # ✅ id_column → entity_columns 변경
    entity_columns: List[str] = Field(..., description="엔티티 컬럼 목록 (user_id, item_id 등)")
```

---

## 🎯 Step 2: Recipe Builder 수정 (Priority: 🟡 High)

### **파일: `src/cli/utils/recipe_builder.py`**

#### **2.1 Entity Columns 입력 추가**

```python
# line 249 이후에 추가
        # Entity columns 설정 (새로 추가)
        self.ui.show_info("🔗 Entity Columns 설정")
        entity_columns_str = self.ui.text_input(
            "Entity column(s) 이름 (쉼표로 구분, 예: user_id,item_id)",
            default="user_id"
        )
        entity_columns = [col.strip() for col in entity_columns_str.split(",")]
        selections["entity_columns"] = entity_columns
        
        # Feature columns 처리 방법 안내
        self.ui.show_info("📊 Feature Columns 자동 처리")
        self.ui.show_info(
            "💡 Feature columns는 자동 처리됩니다:\n"
            "   - Target, Treatment, Entity columns를 제외한 모든 컬럼 사용\n"
            "   - 별도 설정 불필요"
        )
```

#### **2.2 Template 변수 업데이트**

```python
# generate_recipe 메서드에서 template_vars 업데이트
        template_vars = {
            # ... 기존 변수들 ...
            "entity_columns": selections["entity_columns"],
            # feature_columns는 항상 null (자동 처리)
        }
```

---

## 🎯 Step 3: Factory 클래스 수정 (Priority: 🔴 Critical)

### **파일: `src/factory/factory.py`**

#### **3.1 create_data_adapter 메서드 수정**

```python
# line 162-182 수정
    def create_data_adapter(self, adapter_type: Optional[str] = None) -> "BaseAdapter":
        # 기존 로직 유지하되 경로 변경
        if adapter_type:
            target_type = adapter_type
        else:
            # ✅ 새로운 경로: entity_schema 경유하지 않음
            source_uri = self._data.loader.source_uri
            target_type = self._detect_adapter_type_from_uri(source_uri)
            # ... 나머지 동일
```

#### **3.2 create_pyfunc_wrapper 메서드 수정**

```python
# line 410-463 수정: entity_schema 접근 방식 변경
    def create_pyfunc_wrapper(
        self, 
        trained_model: Any, 
        trained_preprocessor: Optional[BasePreprocessor],
        trained_fetcher: Optional['BaseFetcher'],
        training_df: Optional[pd.DataFrame] = None,
        training_results: Optional[Dict[str, Any]] = None
    ) -> PyfuncWrapper:
        """PyfuncWrapper 생성 - 새로운 스키마 구조 대응"""
        from src.factory.artifact import PyfuncWrapper
        logger.info("Creating PyfuncWrapper artifact...")
        
        signature, data_schema = None, None
        if training_df is not None:
            logger.info("Generating model signature and data schema from training_df...")
            from src.utils.integrations.mlflow_integration import create_enhanced_model_signature_with_schema
            
            # ✅ 새로운 구조에서 데이터 수집
            fetcher_conf = self._recipe.data.fetcher
            data_interface = self._recipe.data.data_interface
            
            # Timestamp 컬럼 처리
            ts_col = fetcher_conf.timestamp_column if fetcher_conf else None
            if ts_col and ts_col in training_df.columns:
                import pandas as pd
                if not pd.api.types.is_datetime64_any_dtype(training_df[ts_col]):
                    training_df = training_df.copy()
                    training_df[ts_col] = pd.to_datetime(training_df[ts_col], errors='coerce')

            # ✅ 새로운 구조로 data_interface_config 구성
            data_interface_config = {
                'entity_columns': data_interface.entity_columns,
                'timestamp_column': ts_col,
                'task_type': data_interface.task_type,
                'target_column': data_interface.target_column,
                'treatment_column': getattr(data_interface, 'treatment_column', None),
            }
            
            signature, data_schema = create_enhanced_model_signature_with_schema(
                training_df, 
                data_interface_config
            )
            logger.info("✅ Signature and data schema created successfully.")
        
        return PyfuncWrapper(
            settings=self.settings,
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            trained_fetcher=trained_fetcher,
            training_results=training_results,
            signature=signature,
            data_schema=data_schema,
        )
```

---

## 🎯 Step 4: Data Handler 수정 (Priority: 🟡 High)

### **파일: `src/components/trainer/modules/data_handler.py`**

#### **4.1 _get_exclude_columns 함수 수정**

```python
# line 19-33 수정
def _get_exclude_columns(settings: Settings, df: pd.DataFrame) -> list:
    preproc = getattr(settings.recipe.model, "preprocessor", None)
    params = getattr(preproc, "params", None) if preproc else None
    recipe_exclude = params.get("exclude_cols", []) if isinstance(params, dict) else []

    # ✅ 새로운 구조에서 엔티티/타임스탬프 컬럼 수집
    data_interface = settings.recipe.data.data_interface
    fetcher_conf = settings.recipe.data.fetcher
    
    default_exclude = []
    
    # Entity columns 추가
    try:
        default_exclude.extend(data_interface.entity_columns or [])
    except Exception:
        pass
    
    # Timestamp column 추가
    try:
        ts_col = fetcher_conf.timestamp_column if fetcher_conf else None
        if ts_col:
            default_exclude.append(ts_col)
    except Exception:
        pass

    # 교차 적용
    candidates = set(default_exclude) | set(recipe_exclude)
    return [c for c in candidates if c in df.columns]
```

#### **4.2 prepare_training_data 함수 수정**

```python
# line 34-70 수정: feature_columns null 처리 로직 추가
def prepare_training_data(df: pd.DataFrame, settings: Settings) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """동적 데이터 준비 + feature_columns null 처리"""
    data_interface = settings.recipe.data.data_interface
    task_type = data_interface.task_type
    exclude_cols = _get_exclude_columns(settings, df)
    
    if task_type in ["classification", "regression"]:
        target_col = data_interface.target_column
        
        # ✅ feature_columns null 처리 로직
        if data_interface.feature_columns is None:
            # 자동 선택: target, treatment, entity 제외 모든 컬럼
            auto_exclude = [target_col] + exclude_cols
            if data_interface.treatment_column:
                auto_exclude.append(data_interface.treatment_column)
            
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            logger.info(f"Feature columns 자동 선택: {list(X.columns)}")
        else:
            # 명시적 선택
            X = df[data_interface.feature_columns]
            
        # 숫자형 컬럼만 사용
        X = X.select_dtypes(include=[np.number])
        y = df[target_col]
        additional_data = {}
        
    elif task_type == "clustering":
        # ✅ feature_columns null 처리
        if data_interface.feature_columns is None:
            auto_exclude = exclude_cols
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            logger.info(f"Feature columns 자동 선택 (clustering): {list(X.columns)}")
        else:
            X = df[data_interface.feature_columns]
            
        X = X.select_dtypes(include=[np.number])
        y = None
        additional_data = {}
        
    elif task_type == "causal":
        target_col = data_interface.target_column
        treatment_col = data_interface.treatment_column
        
        # ✅ feature_columns null 처리
        if data_interface.feature_columns is None:
            auto_exclude = [target_col, treatment_col] + exclude_cols
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            logger.info(f"Feature columns 자동 선택 (causal): {list(X.columns)}")
        else:
            X = df[data_interface.feature_columns]
            
        X = X.select_dtypes(include=[np.number])
        y = df[target_col]
        additional_data = {
            'treatment': df[treatment_col],
            'treatment_value': getattr(data_interface, 'treatment_value', 1)
        }
    else:
        raise ValueError(f"지원하지 않는 task_type: {task_type}")
    
    return X, y, additional_data
```

---

## 🎯 Step 5: Feature Store Fetcher 수정 (Priority: 🟡 High)

### **파일: `src/components/fetcher/modules/feature_store_fetcher.py`**

#### **5.1 fetch 메서드 완전 수정**

```python
# line 18-60 완전 교체
    def fetch(self, df: pd.DataFrame, run_mode: str = "batch") -> pd.DataFrame:
        logger.info("Feature Store를 통해 피처 증강을 시작합니다.")

        # ✅ 새로운 구조에서 설정 수집
        data_interface = self.settings.recipe.data.data_interface
        fetcher_conf = self.settings.recipe.data.fetcher

        # ✅ 새로운 feature_views 구조에서 features 리스트 구성
        features: List[str] = []
        if fetcher_conf and fetcher_conf.feature_views:
            for view_name, view_config in fetcher_conf.feature_views.items():
                for feature in view_config.features:
                    features.append(f"{view_name}:{feature}")

        # ✅ 새로운 구조로 data_interface_config 구성
        data_interface_config: Dict[str, Any] = {
            'entity_columns': data_interface.entity_columns,
            'timestamp_column': fetcher_conf.timestamp_column if fetcher_conf else None,
            'task_type': data_interface.task_type,
            'target_column': data_interface.target_column,
            'treatment_column': getattr(data_interface, 'treatment_column', None),
        }

        if run_mode in ("train", "batch"):
            # 오프라인 PIT 조회
            result = self.feature_store_adapter.get_historical_features_with_validation(
                entity_df=df,
                features=features,
                data_interface_config=data_interface_config,
            )
            logger.info("피처 증강 완료(offline).")
            return result
        elif run_mode == "serving":
            # 온라인 조회
            entity_rows = df[data_interface.entity_columns].to_dict(orient="records")
            result = self.feature_store_adapter.get_online_features(
                entity_rows=entity_rows,
                features=features,
            )
            logger.info("피처 증강 완료(online).")
            return result
        else:
            raise ValueError(f"Unsupported run_mode: {run_mode}")
```

---

## 🎯 Step 6: Schema Utils 수정 (Priority: 🟡 High)

### **파일: `src/utils/system/schema_utils.py`**

#### **6.1 validate_schema 함수 수정**

```python
# line 25-35 수정
def validate_schema(df: pd.DataFrame, settings: "Settings", for_training: bool = False):
    """스키마 검증 - 새로운 구조 대응"""
    logger.info(f"모델 입력 데이터 스키마를 검증합니다... (for_training: {for_training})")

    # ✅ 새로운 구조에서 설정 수집
    data_interface = settings.recipe.data.data_interface
    fetcher_conf = settings.recipe.data.fetcher
    
    errors = []
    required_columns = []
    
    if not for_training:
        # 원본 데이터 검증: Entity + Timestamp 필수
        required_columns = data_interface.entity_columns[:]
        if fetcher_conf and fetcher_conf.timestamp_column:
            required_columns.append(fetcher_conf.timestamp_column)
        
        # Target 컬럼 (clustering 제외)
        if data_interface.task_type != "clustering" and data_interface.target_column:
            required_columns.append(data_interface.target_column)
    else:
        # 모델 학습용 데이터: entity/timestamp 제외
        logger.info("모델 학습용 데이터 검증: entity_columns, timestamp_column 제외")
        required_columns = []
        
    # Treatment 컬럼 (causal 전용)
    if data_interface.task_type == "causal" and data_interface.treatment_column:
        required_columns.append(data_interface.treatment_column)
    
    # 필수 컬럼 존재 여부 검증
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"- 필수 컬럼 누락: '{col}' (task_type: {data_interface.task_type})")
    
    # Timestamp 타입 검증
    ts_col = fetcher_conf.timestamp_column if fetcher_conf else None
    if ts_col and ts_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            try:
                pd.to_datetime(df[ts_col])
                logger.info(f"Timestamp 컬럼 '{ts_col}' 자동 변환 가능")
            except Exception:
                errors.append(f"- Timestamp 컬럼 '{ts_col}' 타입 오류: datetime 변환 불가")

    if errors:
        error_message = "모델 입력 데이터 스키마 검증 실패:\n" + "\n".join(errors)
        error_message += f"\n\n필수 컬럼: {required_columns}"
        error_message += f"\n실제 컬럼: {list(df.columns)}"
        raise TypeError(error_message)
    
    logger.info(f"스키마 검증 성공 (task_type: {data_interface.task_type})")
```

---

## 🎯 Step 7: Template 교체 (Priority: 🟢 Low)

### **파일: `src/cli/templates/recipes/recipe.yaml.j2`**

사용자가 제공한 새로운 템플릿으로 **완전 교체**하면 됩니다.

---

## 🎯 Step 8: 테스트 수정 (Priority: 🟡 High)

### **영향받는 테스트 파일들**

1. **`tests/unit/settings/test_recipe.py`** - Recipe 스키마 테스트
2. **`tests/conftest.py`** - 테스트 픽스처
3. **`tests/helpers/builders.py`** - 테스트 빌더
4. **Feature Store 관련 테스트들**

#### **8.1 기본 수정 방향**

```python
# 기존 EntitySchema 사용 → 새로운 구조로 변경
# OLD
entity_schema = EntitySchema(
    entity_columns=["user_id"],
    timestamp_column="event_timestamp"
)

# NEW  
fetcher = Fetcher(
    type="feature_store",
    feature_views={
        "user_features": FeatureView(
            join_key="user_id",
            features=["age", "gender"]
        )
    },
    timestamp_column="event_timestamp"
)

data_interface = DataInterface(
    task_type="classification",
    target_column="label",
    entity_columns=["user_id"]  # id_column → entity_columns
)
```

---

## 📋 실행 순서 (Critical Path)

### **Phase 1: 핵심 스키마 변경** ⚡
1. `src/settings/recipe.py` 수정
2. `src/factory/factory.py` 수정
3. 기본 테스트 수정

### **Phase 2: 데이터 처리 로직** 🔧
4. `src/components/trainer/modules/data_handler.py` 수정
5. `src/components/fetcher/modules/feature_store_fetcher.py` 수정
6. `src/utils/system/schema_utils.py` 수정

### **Phase 3: 사용자 인터페이스** 🖥️
7. `src/cli/utils/recipe_builder.py` 수정
8. Template 교체

### **Phase 4: 통합 테스트** ✅
9. 모든 테스트 수정
10. 통합 테스트 실행

---

## ⚠️ 주의사항

### **호환성 보장**
- 기존 Recipe 파일들이 **즉시 깨집니다**
- 마이그레이션 스크립트 필요할 수 있음
- 단계적 배포 권장

### **데이터 검증**
- `feature_columns: null` 로직 철저한 테스트 필요
- Entity columns 중복 처리 확인
- Causal task에서 treatment_column 필수 검증

### **Feature Store 통합**
- Feast adapter 호환성 확인
- Point-in-time join 로직 검증
- Online/Offline store 모두 테스트

---

## 🎯 완료 기준

- [ ] 새로운 Recipe YAML로 정상 학습 가능
- [ ] Feature Store 연동 정상 동작  
- [ ] Causal task에서 treatment_column 정상 처리
- [ ] feature_columns null일 때 자동 선택 정상 동작
- [ ] 모든 테스트 통과
- [ ] CLI Recipe Builder 정상 동작

이 계획대로 진행하면 새로운 Recipe 구조로 완전히 리팩토링할 수 있습니다! 🚀