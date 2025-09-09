# Adapter Architecture Refactoring Plan
## Recipe 기반 Adapter 시스템 개선

### 📌 Executive Summary
BigQueryAdapter를 SqlAdapter로 통합하고, source_uri 기반 adapter 분기 로직을 유지하면서 config-recipe validation을 강화하는 리팩토링 계획입니다.

### 🎯 핵심 원칙
- **기존 구조 존중**: Recipe의 source_uri 기반 adapter 분기 유지
- **BigQuery 통합**: BigQueryAdapter 삭제, SqlAdapter가 완전 지원
- **Validation 강화**: Config adapter_type와 source_uri 충돌 검증
- **사용자 투명성**: 서비스명은 유지, 내부적으로 적절한 adapter 매핑

---

## 📋 Phase별 리팩토링 계획

### **Phase 1: SqlAdapter BigQuery 완전 지원**

#### 1.1 SqlAdapter 수정
**파일**: `src/components/adapter/modules/sql_adapter.py`

```python
class SqlAdapter(BaseAdapter):
    def __init__(self, settings: Settings, **kwargs):
        self.settings = settings
        self.engine = self._create_engine()
        
        # BigQuery 전용 플래그 및 설정
        self.use_pandas_gbq = False
        if self.db_type == 'bigquery':
            config = settings.config.data_source.config
            self.use_pandas_gbq = config.get('use_pandas_gbq', False)
            self.project_id = config.get('project_id')
            self.dataset_id = config.get('dataset_id')
            self.location = config.get('location', 'US')
    
    def read(self, source: str, params: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
        """BigQuery 지원 강화"""
        if self.db_type == 'bigquery' and self.use_pandas_gbq:
            try:
                import pandas_gbq
                return pandas_gbq.read_gbq(
                    source, 
                    project_id=self.project_id,
                    location=self.location,
                    **kwargs
                )
            except ImportError:
                console.warning("pandas_gbq not installed, using SQLAlchemy")
        
        # 기존 SQLAlchemy 방식
        return pd.read_sql_query(source, self.engine, params=params, **kwargs)
    
    def write(self, df: pd.DataFrame, target: str, **kwargs):
        """BigQuery write 지원"""
        if self.db_type == 'bigquery' and (self.use_pandas_gbq or kwargs.get('if_exists') == 'replace'):
            try:
                import pandas_gbq
                destination_table = f"{self.dataset_id}.{target}" if '.' not in target else target
                pandas_gbq.to_gbq(
                    df,
                    destination_table=destination_table,
                    project_id=self.project_id,
                    location=self.location,
                    if_exists=kwargs.get('if_exists', 'append'),
                    **{k: v for k, v in kwargs.items() if k not in ['if_exists']}
                )
                console.info(f"BigQuery write complete: {len(df)} rows to {destination_table}")
                return
            except ImportError:
                pass  # SQLAlchemy fallback
        
        # 기존 SQLAlchemy 방식
        df.to_sql(target, self.engine, **kwargs)
```

#### 1.2 관련 테스트 수정
**파일**: `tests/unit/components/adapters/test_sql_adapter.py`

```python
def test_bigquery_support_in_sql_adapter(self, settings_builder):
    """Test SqlAdapter handles BigQuery correctly"""
    # Given: BigQuery configuration
    settings = settings_builder \
        .with_data_source("sql", config={
            "connection_uri": "bigquery://test-project/test-dataset",
            "project_id": "test-project",
            "dataset_id": "test-dataset",
            "use_pandas_gbq": True
        }) \
        .build()
    
    # When: Creating SqlAdapter
    adapter = SqlAdapter(settings)
    
    # Then: BigQuery configuration is properly set
    assert adapter.db_type == 'bigquery'
    assert adapter.use_pandas_gbq == True
    assert adapter.project_id == "test-project"
```

---

### **Phase 2: Factory source_uri 감지 수정**

#### 2.1 Factory 수정
**파일**: `src/factory/factory.py`

```python
def _detect_adapter_type_from_uri(self, source_uri: str) -> str:
    """source_uri 패턴 분석하여 adapter type 결정"""
    uri_lower = source_uri.lower()
    
    # SQL 패턴
    if uri_lower.endswith('.sql') or 'select' in uri_lower or 'from' in uri_lower:
        return 'sql'
    
    # BigQuery 패턴 → SQL adapter로 통합
    if uri_lower.startswith('bigquery://'):
        return 'sql'  # 'bigquery' 대신 'sql' 반환
    
    # Cloud Storage 패턴
    if any(uri_lower.startswith(prefix) for prefix in ['s3://', 'gs://', 'az://']):
        return 'storage'
    
    # File 패턴
    if any(uri_lower.endswith(ext) for ext in ['.csv', '.parquet', '.json', '.tsv']):
        return 'storage'
    
    # 기본값
    self.console.warning(f"Unknown source_uri pattern: {source_uri}, using 'storage'")
    return 'storage'
```

#### 2.2 관련 테스트 수정
**파일**: `tests/unit/test_factory.py`

```python
def test_detect_bigquery_uri_returns_sql(self, factory):
    """Test bigquery:// URIs are mapped to sql adapter"""
    adapter_type = factory._detect_adapter_type_from_uri("bigquery://project/dataset")
    assert adapter_type == "sql"
    
def test_detect_various_sql_patterns(self, factory):
    """Test various SQL patterns are detected correctly"""
    test_cases = [
        ("SELECT * FROM table", "sql"),
        ("query.sql", "sql"),
        ("bigquery://project/dataset", "sql"),
        ("postgresql://localhost/db", "sql"),
    ]
    for uri, expected in test_cases:
        assert factory._detect_adapter_type_from_uri(uri) == expected
```

---

### **Phase 3: Settings 모델 수정**

#### 3.1 Settings 모델 수정
**파일**: `src/settings/config.py`

```python
class DataSource(BaseModel):
    """데이터 소스 설정"""
    name: str = Field(..., description="데이터 소스 이름")
    adapter_type: Literal["sql", "storage"] = Field(..., description="어댑터 타입")  # bigquery 제거
    config: Dict[str, Any] = Field(default_factory=dict, description="어댑터별 설정")

class OutputTarget(BaseModel):
    """출력 타겟 설정"""
    name: str = Field(..., description="출력 타겟 이름")
    enabled: bool = Field(True, description="저장 활성화 여부")
    adapter_type: Optional[Literal["storage", "sql"]] = Field(None, description="저장 어댑터 타입")  # bigquery 제거
    config: Optional[Dict[str, Any]] = Field(default=None, description="어댑터별 설정")
```

#### 3.2 관련 테스트 수정
**파일**: `tests/unit/test_settings.py`

```python
def test_bigquery_uses_sql_adapter_type():
    """Test BigQuery configuration uses sql adapter_type"""
    config = {
        "data_source": {
            "name": "BigQuery",
            "adapter_type": "sql",  # bigquery가 아닌 sql
            "config": {
                "connection_uri": "bigquery://project/dataset"
            }
        }
    }
    settings = Settings.from_dict(config)
    assert settings.config.data_source.adapter_type == "sql"
```

---

### **Phase 4: CLI 템플릿 수정**

#### 4.1 Config 템플릿 수정
**파일**: `src/cli/templates/configs/config.yaml.j2`

```yaml
data_source:
  name: {{ data_source }}
  {%- if data_source == "BigQuery" %}
  adapter_type: sql  # bigquery → sql 변경
  config:
    # BigQuery는 connection_uri 또는 개별 설정 사용 가능
    connection_uri: "bigquery://${GCP_PROJECT_ID}/${BQ_DATASET_ID}"
    project_id: "${GCP_PROJECT_ID}"
    dataset_id: "${BQ_DATASET_ID}"
    location: "${BQ_LOCATION:US}"
    use_pandas_gbq: true  # pandas_gbq 사용 옵션
  {%- elif data_source == "PostgreSQL" %}
  adapter_type: sql
  config:
    connection_uri: "postgresql://..."
```

#### 4.2 Output 템플릿 수정
```yaml
output:
  inference:
    {%- if inference_output_source == "BigQuery" %}
    adapter_type: sql  # bigquery → sql 변경
    config:
      connection_uri: "bigquery://${GCP_PROJECT_ID}/${BQ_DATASET_ID}"
      table: "${INFER_OUTPUT_BQ_TABLE}"
```

---

### **Phase 5: BigQueryAdapter 제거 및 정리**

#### 5.1 파일 삭제
```bash
# BigQueryAdapter 관련 파일 삭제
rm src/components/adapter/modules/bigquery_adapter.py
rm tests/unit/components/adapters/test_bigquery_adapter.py
```

#### 5.2 Import 정리
**파일**: `src/components/adapter/__init__.py`

```python
from .registry import AdapterRegistry
from .modules.storage_adapter import StorageAdapter
from .modules.sql_adapter import SqlAdapter

try:
    from .modules.feast_adapter import FeastAdapter
except ImportError:
    FeastAdapter = None

# BigQueryAdapter import 제거

__all__ = [
    "AdapterRegistry",
    "StorageAdapter", 
    "SqlAdapter",
    "FeastAdapter",
    # "BigQueryAdapter" 제거
]
```

---

### **Phase 6: FeastAdapter 및 테스트 수정**

#### 6.1 FeastAdapter 설정 검증
**파일**: `src/settings/config.py`

```python
# DataSource의 adapter_type에 "feast" 추가 안함
# FeastAdapter는 feature_store provider로만 사용

class FeatureStore(BaseModel):
    provider: Literal["feast", "none"] = Field(..., description="Feature store provider")
    feast_config: Optional[FeastConfig] = Field(None, description="Feast 설정")
```

#### 6.2 테스트 Fixture 수정
**파일**: `tests/conftest.py`

```python
def with_data_source(self, adapter_type: str, name: str = "test_source", config: Dict = None):
    """데이터 소스 설정"""
    # BigQuery 호환성 처리
    if adapter_type == "bigquery":
        adapter_type = "sql"
        config = config or {}
        if 'connection_uri' not in config:
            config['connection_uri'] = f"bigquery://{config.get('project_id', 'test-project')}"
    
    # feast/feature_store는 data_source가 아님
    if adapter_type in ["feast", "feature_store"]:
        raise ValueError("Feast는 feature_store provider로 설정하세요")
    
    self._data_source = DataSource(
        name=name,
        adapter_type=adapter_type,
        config=config or {}
    )
    return self

def with_feature_store(self, provider: str = "feast", config: Dict = None):
    """Feature Store 설정 (별도 메서드)"""
    self._feature_store = FeatureStore(
        provider=provider,
        feast_config=config if provider == "feast" else None
    )
    return self
```

#### 6.3 Feast Adapter 테스트 수정
**파일**: `tests/unit/components/adapters/test_feast_adapter.py`

```python
def test_feast_adapter_initialization(self, settings_builder):
    """Test FeastAdapter initialization as feature_store"""
    # Given: Feature Store 설정
    settings = settings_builder \
        .with_data_source("sql", config={"connection_uri": "sqlite:///:memory:"}) \
        .with_feature_store("feast", config={
            "project": "test_project",
            "registry": "./test_registry.db"
        }) \
        .build()
    
    # When: Creating FeastAdapter through Factory
    factory = Factory(settings)
    adapter = factory.create_feature_store_adapter()
    
    # Then: Adapter is properly initialized
    assert isinstance(adapter, FeastAdapter)
```

---

### **Phase 7: Validation 로직 강화**

#### 7.1 Validator 수정
**파일**: `src/settings/validator.py`

```python
def _validate_compatibility(self, settings) -> List[str]:
    """Config와 Recipe 간 호환성 검증"""
    errors = []
    
    # 데이터 어댑터 호환성
    loader_adapter = settings.recipe.data.loader.get_adapter_type()
    config_adapter = settings.config.data_source.adapter_type
    
    # SQL은 모든 SQL 타입과 호환 (bigquery 포함)
    if loader_adapter == "sql" and config_adapter != "sql":
        errors.append(
            f"Recipe loader가 SQL 타입이지만 Config adapter가 {config_adapter}입니다"
        )
    elif loader_adapter == "storage" and config_adapter != "storage":
        errors.append(
            f"Recipe loader가 storage 타입이지만 Config adapter가 {config_adapter}입니다"
        )
    
    # BigQuery URI와 SQL adapter 호환성 확인
    source_uri = settings.recipe.data.loader.source_uri
    if source_uri.startswith('bigquery://') and config_adapter != 'sql':
        errors.append(
            f"BigQuery URI이지만 Config adapter가 {config_adapter}입니다 (sql이어야 함)"
        )
    
    return errors
```

#### 7.2 Validation 테스트
**파일**: `tests/unit/test_validator.py`

```python
def test_validate_bigquery_uri_with_sql_adapter():
    """Test BigQuery URI requires sql adapter"""
    settings = Settings(
        config=Config(
            data_source=DataSource(
                name="BigQuery",
                adapter_type="sql",  # Correct
                config={"connection_uri": "bigquery://project/dataset"}
            )
        ),
        recipe=Recipe(
            data=Data(
                loader=Loader(source_uri="bigquery://project/dataset")
            )
        )
    )
    validator = Validator()
    errors = validator._validate_compatibility(settings)
    assert len(errors) == 0

def test_validate_bigquery_uri_with_wrong_adapter():
    """Test BigQuery URI with wrong adapter type fails"""
    settings = Settings(
        config=Config(
            data_source=DataSource(
                name="BigQuery",
                adapter_type="storage",  # Wrong!
                config={}
            )
        ),
        recipe=Recipe(
            data=Data(
                loader=Loader(source_uri="bigquery://project/dataset")
            )
        )
    )
    validator = Validator()
    errors = validator._validate_compatibility(settings)
    assert "BigQuery URI이지만 Config adapter가 storage입니다" in errors[0]
```

---

## ✅ 검증 체크리스트

### 통합 테스트
```bash
# 전체 adapter 테스트 실행
uv run pytest tests/unit/components/adapters/ -v

# Factory 테스트
uv run pytest tests/unit/test_factory.py::test_detect_adapter_type -v

# Validation 테스트
uv run pytest tests/unit/test_validator.py::test_validate_compatibility -v

# Settings 테스트
uv run pytest tests/unit/test_settings.py::test_data_source -v
```

### 수동 검증
1. **BigQuery 연결 테스트**
   ```bash
   mmp train --config configs/bigquery_test.yaml --recipe recipes/test.yaml
   ```

2. **Config 생성 테스트**
   ```bash
   mmp get-config --env prod
   # BigQuery 선택 시 adapter_type: sql 확인
   ```

3. **Validation 테스트**
   ```bash
   # 충돌 케이스 테스트
   mmp validate --config configs/sql_config.yaml --recipe recipes/csv_recipe.yaml
   # 오류 메시지 확인
   ```

---

## 📊 예상 결과

### Before
- 4개 Adapter: SqlAdapter, BigQueryAdapter, StorageAdapter, FeastAdapter
- Settings Literal: ["sql", "bigquery", "storage"]
- 중복 코드: BigQuery 처리 로직 2곳

### After
- 3개 Adapter: SqlAdapter, StorageAdapter, FeastAdapter
- Settings Literal: ["sql", "storage"]
- 통합 코드: SqlAdapter가 BigQuery 완전 지원
- 강화된 Validation: source_uri와 adapter_type 충돌 검증

---

## 🚦 실행 순서 요약

1. **Phase 1**: SqlAdapter BigQuery 지원 강화
2. **Phase 2**: Factory source_uri 감지 수정
3. **Phase 3**: Settings 모델 수정
4. **Phase 4**: CLI 템플릿 수정
5. **Phase 5**: BigQueryAdapter 제거 및 정리
6. **Phase 6**: FeastAdapter 및 테스트 수정
7. **Phase 7**: Validation 로직 강화

각 Phase 완료 후 해당 테스트 실행하여 검증 필요.