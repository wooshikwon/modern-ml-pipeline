# Phase 5: API/배포 지원 - 종합 리팩토링 계획

## 📊 **Ultra Think 분석 결과**

### **현재 구조 분석**

#### **기존 CLI 구조**:
- `train`: `--recipe-file`, `--env-name`, `--params`
- `batch-inference`: `--run-id`, `--env-name`, `--params`
- `serve-api`: `--run-id`, `--env-name`, `--host`, `--port`

#### **현재 파이프라인 흐름**:
1. **학습**: `settings.recipe.data.loader.source_uri` → 데이터 로드 → MLflow 저장
2. **배치추론**: `wrapped_model.loader_sql_snapshot` → 데이터 로드 → 예측
3. **API서빙**: `data_schema/loader_sql` → API 스키마 동적 생성 → 서빙

#### **현재 문제점들**:
- Recipe에서 데이터 경로 고정 → 유연성 부족
- 원본 데이터 구조 미보존 → 컬럼 검증 부정확
- Timeseries API 응답 구조 미완성
- DataInterface 필수 컬럼 외 불필요한 검증

---

## 🚀 **리팩토링 목표**

### **1. CLI 인자 구조 혁신**
```bash
# 📌 새로운 CLI 구조
mmp train --config-path configs/dev.yaml --recipe-path recipes/ts_model.yaml --data-path data/train.csv
mmp batch-inference --config-path configs/prod.yaml --run-id abc123 --data-path data/inference.sql
mmp serve-api --config-path configs/prod.yaml --run-id abc123
```

### **2. DataInterface 기반 정확한 컬럼 검증**
- 학습 시: DataInterface 필수 컬럼만 검증
- 배치추론 시: 해당 run_id의 DataInterface 필수 컬럼 검증  
- API서빙 시: API 인터페이스가 DataInterface 필수 컬럼과 정확히 일치

### **3. Timeseries 완전 지원**
- 시계열 특화 API 응답 구조
- 미래 예측 타임스탬프 정보 포함
- Timeseries 모델 유형별 최적화

---

## 🔧 **상세 구현 계획**

### **Phase 5.1: DataInterface 검증 로직 구현**

#### **`src/utils/system/data_validation.py` (신규 생성)**
```python
from typing import List, Set
import pandas as pd
from src.settings.recipe import DataInterface

def get_required_columns_from_data_interface(data_interface: DataInterface) -> List[str]:
    """DataInterface에서 필수 컬럼 목록 추출"""
    required = [data_interface.target_column] + data_interface.entity_columns
    
    # Task별 특수 컬럼 추가
    if data_interface.task_type == "timeseries" and data_interface.timestamp_column:
        required.append(data_interface.timestamp_column)
    elif data_interface.task_type == "causal" and data_interface.treatment_column:
        required.append(data_interface.treatment_column)
    
    # 명시적 feature_columns 추가
    if data_interface.feature_columns:
        required.extend(data_interface.feature_columns)
    
    return list(set(required))  # 중복 제거

def validate_data_interface_columns(df: pd.DataFrame, data_interface: DataInterface) -> None:
    """DataInterface 필수 컬럼 검증"""
    required_columns = get_required_columns_from_data_interface(data_interface)
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(
            f"DataInterface 필수 컬럼 누락:\n"
            f"누락된 컬럼: {sorted(missing_columns)}\n"
            f"필요한 컬럼: {sorted(required_columns)}\n"
            f"실제 컬럼: {sorted(df.columns.tolist())}"
        )

def create_data_interface_schema_for_storage(data_interface: DataInterface, df: pd.DataFrame) -> dict:
    """PyfuncWrapper 저장용 DataInterface 스키마 생성"""
    required_columns = get_required_columns_from_data_interface(data_interface)
    
    return {
        'data_interface': data_interface.model_dump(),
        'required_columns': required_columns,
        'column_dtypes': {col: str(df[col].dtype) for col in required_columns if col in df.columns},
        'validation_timestamp': pd.Timestamp.now().isoformat()
    }
```

### **Phase 5.2: PyfuncWrapper 확장**

#### **`src/factory/artifact.py` 수정**
```python
class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        settings: Settings,
        trained_model: Any,
        trained_datahandler: Optional[BaseDataHandler],
        trained_preprocessor: Optional[BasePreprocessor],
        trained_fetcher: Optional[BaseFetcher],
        data_interface_schema: Dict[str, Any],  # 🆕 핵심 추가
        training_results: Optional[Dict[str, Any]] = None,
        signature: Optional[Any] = None,
        data_schema: Optional[Any] = None,
    ):
        # ... 기존 코드 ...
        self.data_interface_schema = data_interface_schema  # DataInterface 기반 검증용
    
    def predict(self, context, model_input, params=None):
        run_mode = params.get("run_mode", "batch") if params else "batch"
        
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        
        # 🆕 DataInterface 기반 컬럼 검증
        self._validate_data_interface_columns(model_input)
        
        if self._task_type == "timeseries":
            return self._predict_timeseries(model_input, run_mode)
        else:
            return self._predict_traditional(model_input, run_mode)
    
    def _validate_data_interface_columns(self, df: pd.DataFrame):
        """DataInterface 필수 컬럼 검증"""
        from src.utils.system.data_validation import validate_data_interface_columns
        from src.settings.recipe import DataInterface
        
        data_interface = DataInterface(**self.data_interface_schema['data_interface'])
        validate_data_interface_columns(df, data_interface)
        logger.info("✅ DataInterface 필수 컬럼 검증 완료")
    
    def _predict_timeseries(self, model_input: pd.DataFrame, run_mode: str) -> Dict[str, Any]:
        """시계열 전용 예측 - 미래 타임스탬프 포함 응답"""
        # 기존 DataHandler 파이프라인 실행
        fetched_df = self.trained_fetcher.fetch(model_input, run_mode=run_mode)
        X, _, additional_data = self.trained_datahandler.prepare_data(fetched_df)
        
        if self.trained_preprocessor:
            X = self.trained_preprocessor.transform(X)
        
        predictions = self.trained_model.predict(X)
        
        # 🆕 시계열 특화 응답 구조
        return {
            'predictions': predictions.tolist(),
            'input_timestamps': additional_data['timestamp'].dt.isoformat().tolist(),
            'forecast_type': 'point_forecast',
            'model_type': 'timeseries',
            'task_type': self._task_type,
            'model_class': self.settings.recipe.model.class_path
        }
```

### **Phase 5.3: CLI 명령어 리팩토링**

#### **`src/cli/commands/train_command.py` 수정**
```python
def train_command(
    recipe_path: Annotated[str, typer.Option("--recipe-path", "-r", help="Recipe 파일 경로")],
    config_path: Annotated[str, typer.Option("--config-path", "-c", help="Config 파일 경로")],
    data_path: Annotated[str, typer.Option("--data-path", "-d", help="학습 데이터 파일 경로")],
    params: Annotated[Optional[str], typer.Option("--params", "-p", help="추가 파라미터 (JSON)")] = None,
) -> None:
    """학습 파이프라인 실행 (Phase 5 리팩토링)"""
    try:
        # 1. Settings 생성 (recipe + config 분리 로드)
        settings = load_settings_from_separate_files(recipe_path, config_path)
        
        # 2. --data-path로 직접 데이터 로드
        from src.utils.system.data_loading import load_data_from_path
        original_df = load_data_from_path(data_path)
        
        # 3. DataInterface 컬럼 검증
        from src.utils.system.data_validation import (
            validate_data_interface_columns, 
            create_data_interface_schema_for_storage
        )
        validate_data_interface_columns(original_df, settings.recipe.data.data_interface)
        
        # 4. DataInterface 스키마 생성
        data_interface_schema = create_data_interface_schema_for_storage(
            settings.recipe.data.data_interface, original_df
        )
        
        # 5. 학습 파이프라인 실행 (데이터 전달)
        run_train_pipeline_with_data(
            settings=settings, 
            training_df=original_df,
            data_interface_schema=data_interface_schema,
            context_params=json.loads(params) if params else None
        )
        
        logger.info("✅ Phase 5 학습 완료")
        
    except Exception as e:
        logger.error(f"학습 실패: {e}")
        raise typer.Exit(code=1)
```

#### **`src/cli/commands/inference_command.py` 수정**
```python
def batch_inference_command(
    run_id: Annotated[str, typer.Option("--run-id", help="MLflow Run ID")],
    config_path: Annotated[str, typer.Option("--config-path", "-c", help="Config 파일 경로")],
    data_path: Annotated[str, typer.Option("--data-path", "-d", help="추론 데이터 파일 경로")],
    params: Annotated[Optional[str], typer.Option("--params", "-p", help="추가 파라미터 (JSON)")] = None,
) -> None:
    """배치 추론 실행 (Phase 5 리팩토링)"""
    try:
        # 1. Config만 로드 (Recipe는 MLflow에서)
        config_data = load_config_files(config_path=config_path)
        settings = create_settings_for_inference(config_data)
        
        # 2. --data-path로 직접 데이터 로드
        from src.utils.system.data_loading import load_data_from_path
        inference_df = load_data_from_path(data_path)
        
        # 3. 배치 추론 실행 (데이터 전달 + 자동 검증)
        run_inference_pipeline_with_data(
            settings=settings,
            run_id=run_id,
            inference_df=inference_df,
            context_params=json.loads(params) if params else None
        )
        
        logger.info("✅ Phase 5 배치 추론 완료")
        
    except Exception as e:
        logger.error(f"배치 추론 실패: {e}")
        raise typer.Exit(code=1)
```

#### **`src/cli/commands/serve_command.py` 수정**
```python
def serve_api_command(
    run_id: Annotated[str, typer.Option("--run-id", help="MLflow Run ID")],
    config_path: Annotated[str, typer.Option("--config-path", "-c", help="Config 파일 경로")],
    host: Annotated[str, typer.Option("--host", help="호스트")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="포트")] = 8000,
) -> None:
    """API 서버 실행 (Phase 5 리팩토링)"""
    try:
        # 1. Config만 로드
        config_data = load_config_files(config_path=config_path)
        settings = create_settings_for_inference(config_data)
        
        # 2. API 서버 실행 (DataInterface 기반 스키마 자동 생성)
        run_api_server_with_data_interface_validation(
            settings=settings,
            run_id=run_id,
            host=host,
            port=port
        )
        
    except Exception as e:
        logger.error(f"API 서버 실행 실패: {e}")
        raise typer.Exit(code=1)
```

### **Phase 5.4: 파이프라인 업데이트**

#### **`src/pipelines/train_pipeline.py` 수정**
```python
def run_train_pipeline_with_data(
    settings: Settings, 
    training_df: pd.DataFrame,
    data_interface_schema: Dict[str, Any],
    context_params: Optional[Dict[str, Any]] = None
):
    """Phase 5: 데이터 직접 전달 방식 학습 파이프라인"""
    logger.info("🆕 Phase 5 학습 파이프라인 시작 (데이터 직접 전달)")
    
    with mlflow_utils.start_run(settings, run_name=settings.recipe.model.computed["run_name"]) as run:
        factory = Factory(settings)
        
        # 컴포넌트 생성 (기존과 동일)
        fetcher = factory.create_fetcher()
        datahandler = factory.create_datahandler()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()
        evaluator = factory.create_evaluator()
        trainer = factory.create_trainer()
        
        # 학습 실행 (training_df 직접 사용)
        trained_model, trained_preprocessor, metrics, training_results = trainer.train(
            df=training_df,  # --data-path에서 로드된 데이터
            model=model,
            fetcher=fetcher,
            datahandler=datahandler,
            preprocessor=preprocessor,
            evaluator=evaluator,
            context_params=context_params,
        )
        
        # 🆕 Phase 5: DataInterface 스키마 포함 PyfuncWrapper 생성
        pyfunc_wrapper = factory.create_pyfunc_wrapper(
            trained_model=trained_model,
            trained_datahandler=datahandler,
            trained_preprocessor=trained_preprocessor,
            trained_fetcher=fetcher,
            data_interface_schema=data_interface_schema,  # 핵심 추가
            training_df=training_df,
            training_results=training_results,
        )
        
        # MLflow 저장 (기존과 동일)
        # ...
```

#### **`src/pipelines/inference_pipeline.py` 수정**
```python
def run_inference_pipeline_with_data(
    settings: Settings, 
    run_id: str, 
    inference_df: pd.DataFrame,
    context_params: Optional[Dict] = None
):
    """Phase 5: 데이터 직접 전달 방식 배치 추론"""
    logger.info("🆕 Phase 5 배치 추론 파이프라인 시작")
    
    with start_run(settings, run_name=f"batch_inference_{run_id}") as run:
        # 모델 로드
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 🆕 자동 DataInterface 검증 (PyfuncWrapper 내부에서)
        predictions_df = model.predict(inference_df)
        
        # 메타데이터 추가 및 저장 (기존과 동일)
        # ...
```

### **Phase 5.5: API 서빙 강화**

#### **`src/serving/_lifespan.py` 수정**
```python
def setup_api_context_with_data_interface(run_id: str, settings: Settings):
    """Phase 5: DataInterface 기반 API 컨텍스트 설정"""
    try:
        bootstrap(settings)
        model_uri = f"runs:/{run_id}/model"
        app_context.model = mlflow.pyfunc.load_model(model_uri)
        app_context.model_uri = model_uri
        app_context.settings = settings
        
        wrapped_model = app_context.model.unwrap_python_model()
        
        # 🆕 DataInterface 스키마에서 API 스키마 생성
        data_interface_schema = getattr(wrapped_model, 'data_interface_schema', {})
        if data_interface_schema and 'required_columns' in data_interface_schema:
            api_columns = data_interface_schema['required_columns']
        else:
            # Fallback: 기존 방식
            api_columns = parse_select_columns(wrapped_model.loader_sql_snapshot)
        
        # 동적 API 스키마 생성
        app_context.PredictionRequest = create_dynamic_prediction_request(
            model_name="DataInterfacePredictionRequest", 
            pk_fields=api_columns
        )
        app_context.BatchPredictionRequest = create_batch_prediction_request(
            app_context.PredictionRequest
        )
        
        logger.info(f"✅ Phase 5 API 컨텍스트 설정 완료: {api_columns}")
        
    except Exception as e:
        logger.error(f"API 컨텍스트 설정 실패: {e}")
        raise
```

### **Phase 5.6: Timeseries API 응답 강화**

#### **`src/serving/schemas.py` 추가**
```python
class TimeseriesPredictionResponse(BaseModel):
    """시계열 전용 예측 응답 스키마"""
    predictions: List[float] = Field(..., description="예측 값들")
    input_timestamps: List[str] = Field(..., description="입력 타임스탬프들 (ISO format)")
    forecast_type: str = Field(default="point_forecast", description="예측 유형")
    model_type: str = Field(default="timeseries", description="모델 타입")
    task_type: str = Field(..., description="태스크 타입")
    model_class: str = Field(..., description="모델 클래스 경로")
    model_uri: str = Field(..., description="모델 URI")

def create_task_specific_response_model(task_type: str) -> Type[BaseModel]:
    """Task 타입별 응답 모델 생성"""
    if task_type == "timeseries":
        return TimeseriesPredictionResponse
    else:
        return MinimalPredictionResponse
```

---

## 🧪 **검증 및 테스트 계획**

### **Phase 5.7: 통합 테스트**

#### **DataInterface 검증 테스트**
```python
def test_data_interface_validation():
    """DataInterface 필수 컬럼 검증 테스트"""
    # 정상 케이스: 모든 필수 컬럼 포함
    # 오류 케이스: 필수 컬럼 누락
    # Task별 특수 컬럼 테스트 (timeseries, causal)
```

#### **CLI 통합 테스트**
```python
def test_phase5_cli_integration():
    """새로운 CLI 구조 통합 테스트"""
    # train --config-path --recipe-path --data-path
    # batch-inference --config-path --run-id --data-path  
    # serve-api --config-path --run-id
```

#### **Timeseries End-to-End 테스트**
```python
def test_timeseries_e2e_pipeline():
    """시계열 전체 파이프라인 테스트"""
    # 학습 → 배치추론 → API서빙
    # DataInterface 검증
    # 시계열 특화 응답 구조
```

---

## 📋 **구현 우선순위**

### **1단계 (핵심 기반)**:
- [ ] DataInterface 검증 로직 구현
- [ ] PyfuncWrapper data_interface_schema 추가
- [ ] CLI 명령어 인자 구조 변경

### **2단계 (파이프라인)**:
- [ ] 학습 파이프라인 데이터 직접 전달 방식
- [ ] 배치 추론 파이프라인 데이터 직접 전달 방식
- [ ] API 서빙 DataInterface 기반 스키마 생성

### **3단계 (고도화)**:
- [ ] Timeseries 특화 API 응답 구조
- [ ] Task별 응답 모델 동적 생성
- [ ] 전체 통합 테스트 및 검증

---

## 🎯 **기대 효과**

### **유연성 대폭 향상**:
- Recipe와 데이터 경로 분리 → 동일 Recipe 다른 데이터 학습 가능
- Config 파일 경로 명시 → 환경별 설정 관리 명확화

### **정확성 보장**:
- DataInterface 기반 정확한 컬럼 검증
- 학습/추론/API 단계별 일관된 검증 로직

### **Timeseries 완전 지원**:
- 시계열 특화 API 응답 구조
- 미래 예측 정보 포함한 풍부한 메타데이터

### **운영 안정성**:
- 명확한 오류 메시지와 검증 로직
- 컬럼 불일치 조기 감지 및 명확한 안내

---

**Phase 5 리팩토링을 통해 현대적이고 안정적인 ML 파이프라인 구축 완성!** 🚀