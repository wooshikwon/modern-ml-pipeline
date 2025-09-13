# CLI Settings 통합 및 완전 재현성 구현 계획

## 1. 현재 상황 완전 분석

### 1.1 CLI 명령어별 Settings 생성 현황

#### Train Command (`train_command.py`)
**복잡도**: 🔴 매우 높음 (60+ 줄)
```python
# 현재 구조
def train_command(recipe_path, config_path, data_path, context_params, record_reqs):
    # 1. 파라미터 파싱 (JSON)
    # 2. Settings 생성 (load_settings)
    # 3. 템플릿 렌더링 처리 (.sql.j2)
    # 4. source_uri 동적 설정
    # 5. 데이터 소스 호환성 검증
    # 6. 파이프라인 실행
```

**특징:**
- **완전한 Recipe**: `load_settings(recipe_path, config_path)`로 실제 Recipe 로드
- **복잡한 로직**: CLI에서 템플릿 렌더링, 검증, 파라미터 처리 등 모든 로직 담당
- **동적 처리**: Jinja 템플릿 파일 렌더링, source_uri 동적 설정

#### Serve API Command (`serve_command.py`)
**복잡도**: 🟢 매우 낮음 (3줄)
```python
# 현재 구조
def serve_api_command(run_id, config_path, host, port):
    config_data = load_config_files(config_path)           # Config만 로드
    settings = create_settings_for_inference(config_data)  # 🚨 더미 Recipe 생성
    run_api_server(settings, run_id, host, port)
```

**특징:**
- **더미 Recipe**: 하드코딩된 가짜 Recipe 사용
- **재현성 부재**: 학습 당시 설정과 완전히 다른 환경
- **Config만 활용**: 실제 Recipe 정보 전혀 사용 안함

#### Batch Inference Command (`inference_command.py`)
**복잡도**: 🟢 매우 낮음 (5줄)
```python
# 현재 구조
def batch_inference_command(run_id, config_path, data_path, context_params):
    config_data = load_config_files(config_path)           # Config만 로드
    settings = create_settings_for_inference(config_data)  # 🚨 더미 Recipe 생성
    run_inference_pipeline(settings, run_id, data_path, context_params)
```

**특징:**
- **더미 Recipe**: serve-api와 동일한 하드코딩 문제
- **불완전한 처리**: 학습시와 다른 data_interface 설정
- **데이터 처리 한계**: CLI data_path 또는 모델의 loader_sql_snapshot에만 의존

### 1.2 MLflow 저장/복원 현황

#### 학습시 MLflow 저장 (`train_pipeline.py`)
```python
# 현재 저장되는 정보
- PyfuncWrapper 모델
- data_interface_schema (제한적)
- data_schema (일부)
- loader_sql_snapshot (SQL만)
- signature, input_example
```

**🚨 중대한 누락사항:**
- **Recipe 전체**: 학습시 사용된 완전한 Recipe 정보 미저장
- **Context Parameters**: Jinja 템플릿 파라미터 정보 손실
- **설정 스냅샷**: 학습 당시의 완전한 Settings 정보 부재

#### 추론시 MLflow 복원 (`inference_pipeline.py`, `serving/`)
```python
# 현재 복원 방식
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")  # 모델만 로드
wrapped_model = model.unwrap_python_model()
data_interface = wrapped_model.data_interface_schema.get('data_interface_config', {})  # 제한적 정보만
```

**🚨 재현성 문제:**
- **불완전한 복원**: 학습시 Recipe의 5% 정도만 복원 가능
- **하드코딩 의존**: 더미 Recipe로 학습 환경과 완전히 다른 설정 사용
- **데이터 처리 불일치**: 학습시와 추론시 데이터 로딩 방식이 완전히 다름

### 1.3 Settings Loader 현황 (`settings/loader.py`)

#### 현재 함수들의 역할
```python
# 1. load_settings(recipe_path, config_path) -> Settings
#    - 학습용 완전한 Settings 생성
#    - Recipe + Config 모두 로드

# 2. create_settings_for_inference(config_data) -> Settings
#    - 🚨 하드코딩된 더미 Recipe 생성
#    - Config만 사용, Recipe는 가짜 데이터

# 3. load_config_files(config_path) -> Dict
#    - Config 파일만 로드
#    - create_settings_for_inference의 전처리용
```

**🚨 아키텍처 문제:**
- **일관성 부재**: 학습용과 추론용 Settings 생성 방식이 완전히 다름
- **재현성 불가**: 추론시 학습 당시 Recipe를 복원할 수 없는 구조
- **중복 로직**: train_command에서 복잡한 로직을 중복 구현

## 2. 근본 문제 정의

### 2.1 MLOps 재현성 원칙 위반

**문제**: 학습과 추론이 다른 환경에서 실행됨
- **학습시**: 완전한 Recipe (data_interface, fetcher, preprocessor 설정 등)
- **추론시**: 하드코딩된 더미 Recipe (classification, target, sklearn 등)

**결과**:
- 모델 성능 불일치 가능성
- 디버깅 불가능
- 프로덕션 환경에서 예측 불가능한 오류

### 2.2 CLI 아키텍처 불일치

**문제**: 3개 명령어의 책임과 복잡도가 완전히 다름
- **train**: 모든 로직을 CLI에서 직접 처리 (60+ 줄)
- **serve-api/batch-inference**: 단순 위임 (3-5줄)

**결과**:
- 코드 유지보수 어려움
- 새로운 기능 추가시 일관성 부재
- 테스트 복잡도 증가

### 2.3 데이터 처리 로직 분산

**문제**: 템플릿 렌더링, 검증 로직이 여러 곳에 분산
- **train_command**: CLI에서 복잡한 템플릿 처리
- **inference_pipeline**: data_io.py의 load_inference_data 함수 사용
- **serving**: 모델의 loader_sql_snapshot 의존

**결과**:
- 로직 중복 및 불일치
- 오류 발생시 디버깅 어려움
- 새로운 데이터 소스 추가시 여러 곳 수정 필요

## 3. 이상적인 구조 설계

### 3.1 완전 재현성 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        학습 단계                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Recipe + Config → Complete Settings                     │
│ 2. Training Pipeline 실행                                    │
│ 3. MLflow에 저장:                                           │
│    ├── Model (PyfuncWrapper)                               │
│    ├── Recipe Snapshot (완전한 recipe.yaml)                 │
│    ├── Config Snapshot (완전한 config.yaml)                 │
│    ├── Context Parameters (Jinja 파라미터들)                │
│    └── Settings Metadata (완전한 학습 환경 정보)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        추론 단계                              │
├─────────────────────────────────────────────────────────────┤
│ 1. MLflow에서 복원:                                          │
│    ├── Recipe Snapshot → 학습시와 동일한 Recipe             │
│    ├── Config (현재) + Recipe (학습시) → Complete Settings  │
│    └── Context Parameters → 동일한 템플릿 처리 환경         │
│ 2. Inference/Serving Pipeline 실행                         │
│ 3. 100% 재현성 보장                                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 통합 Settings 생성 시스템

```python
# 새로운 통합 구조 - 단순한 클래스명 사용
class SettingsFactory:
    """모든 CLI 명령어가 사용하는 통합 Settings 팩토리"""

    @classmethod
    def for_training(cls, recipe_path: str, config_path: str,
                    data_path: str, context_params: Dict = None) -> Settings:
        """학습용 Settings 생성 + MLflow 저장 준비"""

    @classmethod
    def for_inference(cls, config_path: str, run_id: str) -> Settings:
        """추론용 Settings 생성 - MLflow에서 Recipe 완전 복원"""

    @classmethod
    def for_serving(cls, config_path: str, run_id: str) -> Settings:
        """서빙용 Settings 생성 - 추론과 동일한 로직"""
```

### 3.3 CLI 명령어 일관성

```python
# 모든 CLI 명령어가 동일한 패턴
def train_command(...):
    settings = SettingsFactory.for_training(...)
    setup_logging(settings)
    run_train_pipeline(settings, ...)

def serve_api_command(...):
    settings = SettingsFactory.for_serving(...)
    setup_logging(settings)
    run_api_server(settings, ...)

def batch_inference_command(...):
    settings = SettingsFactory.for_inference(...)
    setup_logging(settings)
    run_inference_pipeline(settings, ...)
```

## 4. 구체적 구현 계획

### 4.1 Phase 1: MLflow Recipe 저장/복원 시스템

#### 4.1.1 학습시 Recipe 저장 강화

**파일**: `src/utils/integrations/mlflow_integration.py`

```python
def log_complete_training_artifacts(
    settings: Settings,
    recipe_path: str,
    config_path: str,
    context_params: Optional[Dict] = None,
    python_model = None,
    signature = None,
    input_example = None
):
    """학습시 완전한 재현성을 위한 모든 정보 저장"""

    # 1. 기존 모델 저장 (호환성 유지)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=python_model,
        signature=signature,
        input_example=input_example
    )

    # 2. 🆕 Recipe 스냅샷 저장
    mlflow.log_artifact(recipe_path, "training_artifacts/recipe_snapshot.yaml")

    # 3. 🆕 Config 스냅샷 저장
    mlflow.log_artifact(config_path, "training_artifacts/config_snapshot.yaml")

    # 4. 🆕 Context Parameters 저장
    if context_params:
        mlflow.log_dict(context_params, "training_artifacts/context_params.json")

    # 5. 🆕 Complete Settings 메타데이터 저장
    settings_metadata = {
        "recipe_name": settings.recipe.name,
        "task_type": settings.recipe.task_choice,
        "data_interface": settings.recipe.data.data_interface.model_dump(),
        "fetcher_config": settings.recipe.data.fetcher.model_dump(),
        "model_config": settings.recipe.model.model_dump(),
        "timestamp": datetime.now().isoformat(),
        "mmp_version": "2.0"
    }
    mlflow.log_dict(settings_metadata, "training_artifacts/settings_metadata.json")
```

#### 4.1.2 추론시 Recipe 복원 시스템

**파일**: `src/settings/mlflow_recipe_restore.py` (신규)

```python
class MLflowRecipeRestorer:
    """MLflow에서 학습시 Recipe를 완전히 복원하는 클래스"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.client = mlflow.tracking.MlflowClient()

    def restore_recipe(self) -> Recipe:
        """학습시 사용된 Recipe를 완전히 복원"""
        try:
            # MLflow artifacts에서 recipe_snapshot.yaml 다운로드
            recipe_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="training_artifacts/recipe_snapshot.yaml"
            )

            # Recipe 객체로 복원
            with open(recipe_path, 'r', encoding='utf-8') as f:
                recipe_data = yaml.safe_load(f)

            # 환경변수 치환 (현재 환경 기준)
            recipe_data = resolve_env_variables(recipe_data)

            return Recipe(**recipe_data)

        except Exception as e:
            raise ValueError(f"Recipe 복원 실패 (run_id: {self.run_id}): {e}")

    def restore_context_params(self) -> Optional[Dict]:
        """학습시 사용된 Context Parameters 복원"""
        try:
            context_params_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="training_artifacts/context_params.json"
            )

            with open(context_params_path, 'r') as f:
                return json.load(f)

        except Exception:
            # Context Parameters가 없으면 None 반환 (정상)
            return None

    def get_training_metadata(self) -> Dict:
        """학습시 메타데이터 조회"""
        try:
            metadata_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="training_artifacts/settings_metadata.json"
            )

            with open(metadata_path, 'r') as f:
                return json.load(f)

        except Exception as e:
            raise ValueError(f"Training metadata 조회 실패: {e}")
```

### 4.2 Phase 2: 통합 Settings 팩토리 구현

**파일**: `src/settings/factory.py` (신규)

```python
class SettingsFactory:
    """모든 CLI 명령어를 위한 통합 Settings 생성 팩토리"""

    @classmethod
    def for_training(cls, recipe_path: str, config_path: str,
                    data_path: str, context_params: Optional[Dict] = None) -> Settings:
        """
        학습용 Settings 생성
        - 기존 train_command의 복잡한 로직을 모두 여기로 이관
        - CLI는 단순 위임만 담당
        """
        logger.info(f"Training Settings 생성: recipe={recipe_path}, config={config_path}")

        # 1. 기본 Settings 로드
        settings = load_settings(recipe_path, config_path)

        # 2. 데이터 경로 처리 (기존 train_command 로직 이관)
        cls._process_training_data_path(settings, data_path, context_params)

        # 3. 검증 실행
        settings.validate_data_source_compatibility()

        # 4. 학습용 런타임 필드 추가
        cls._add_training_computed_fields(settings, recipe_path)

        logger.info(f"Training Settings 생성 완료: task={settings.recipe.task_choice}")
        return settings

    @classmethod
    def for_inference(cls, config_path: str, run_id: str) -> Settings:
        """
        추론용 Settings 생성
        - MLflow에서 학습시 Recipe 완전 복원
        - 현재 Config + 학습시 Recipe 조합
        """
        logger.info(f"Inference Settings 생성: run_id={run_id}")

        # 1. 현재 Config 로드
        config = _load_config(config_path)

        # 2. MLflow에서 학습시 Recipe 복원
        restorer = MLflowRecipeRestorer(run_id)
        recipe = restorer.restore_recipe()

        # 3. Settings 생성 (현재 Config + 학습시 Recipe)
        settings = Settings(config, recipe)

        # 4. 추론용 런타임 필드 추가
        training_metadata = restorer.get_training_metadata()
        cls._add_inference_computed_fields(settings, run_id, training_metadata)

        logger.info(f"Inference Settings 생성 완료: task={recipe.task_choice}")
        return settings

    @classmethod
    def for_serving(cls, config_path: str, run_id: str) -> Settings:
        """
        서빙용 Settings 생성
        - 추론과 동일한 로직 사용
        """
        return cls.for_inference(config_path, run_id)

    @classmethod
    def _process_training_data_path(cls, settings: Settings, data_path: str,
                                   context_params: Optional[Dict]):
        """학습용 데이터 경로 처리 (기존 train_command 로직)"""
        if not data_path:
            raise ValueError("--data-path는 필수입니다.")

        if data_path.endswith('.sql.j2') or (data_path.endswith('.sql') and context_params):
            # Jinja 템플릿 렌더링 (기존 로직 그대로)
            cls._process_template_data_path(settings, data_path, context_params)
        else:
            # 일반 파일 경로
            settings.recipe.data.loader.source_uri = data_path

    @classmethod
    def _process_template_data_path(cls, settings: Settings, data_path: str,
                                   context_params: Dict):
        """템플릿 데이터 경로 처리 (기존 train_command 로직)"""
        from src.utils.template.templating_utils import render_template_from_string
        from pathlib import Path

        template_path = Path(data_path)
        if not template_path.exists():
            raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {data_path}")

        template_content = template_path.read_text()
        if context_params:
            rendered_sql = render_template_from_string(template_content, context_params)
            settings.recipe.data.loader.source_uri = rendered_sql
            logger.info(f"✅ Jinja 템플릿 렌더링 성공: {data_path}")
        else:
            raise ValueError(f"Jinja 템플릿 파일({data_path})에는 --params가 필요합니다")
```

### 4.3 Phase 3: CLI 명령어 일관성 개선

#### 4.3.1 train_command 단순화

**파일**: `src/cli/commands/train_command.py`

```python
def train_command(recipe_path: str, config_path: str, data_path: str,
                 context_params: Optional[str] = None, record_reqs: bool = False):
    """학습 파이프라인 실행 - 단순화된 인터페이스"""
    try:
        # 1. 파라미터 파싱
        params = json.loads(context_params) if context_params else None

        # 2. Settings 생성 (복잡한 로직은 UnifiedSettingsFactory에서 처리)
        settings = UnifiedSettingsFactory.for_training(
            recipe_path=recipe_path,
            config_path=config_path,
            data_path=data_path,
            context_params=params
        )
        setup_logging(settings)

        # 3. 학습 정보 로깅
        logger.info(f"Recipe: {recipe_path}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Data: {data_path}")
        run_name = settings.recipe.model.computed.get("run_name", "unknown")
        logger.info(f"Run Name: {run_name}")

        # 4. 파이프라인 실행
        if record_reqs:
            run_train_pipeline(settings=settings, context_params=params,
                             record_requirements=True)
        else:
            run_train_pipeline(settings=settings, context_params=params)

        logger.info("✅ 학습이 성공적으로 완료되었습니다.")

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"학습 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)
```

#### 4.3.2 serve_command 개선

**파일**: `src/cli/commands/serve_command.py`

```python
def serve_api_command(run_id: str, config_path: str,
                     host: str = "0.0.0.0", port: int = 8000):
    """API 서버 실행 - 학습시 Recipe 완전 복원"""
    try:
        # Settings 생성 (MLflow에서 Recipe 복원)
        settings = UnifiedSettingsFactory.for_serving(
            config_path=config_path,
            run_id=run_id
        )
        setup_logging(settings)

        # 서버 정보 로깅
        logger.info(f"Config: {config_path}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Task Type: {settings.recipe.task_choice}")  # 🆕 학습시 정보
        logger.info(f"Server: {host}:{port}")

        # API 서버 실행
        run_api_server(settings=settings, run_id=run_id, host=host, port=port)

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"API 서버 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)
```

#### 4.3.3 inference_command 개선

**파일**: `src/cli/commands/inference_command.py`

```python
def batch_inference_command(run_id: str, config_path: str, data_path: str,
                           context_params: Optional[str] = None):
    """배치 추론 실행 - 학습시 Recipe 완전 복원"""
    try:
        # 파라미터 파싱
        params = json.loads(context_params) if context_params else None

        # Settings 생성 (MLflow에서 Recipe 복원)
        settings = UnifiedSettingsFactory.for_inference(
            config_path=config_path,
            run_id=run_id
        )
        setup_logging(settings)

        # 추론 정보 로깅
        logger.info(f"Config: {config_path}")
        logger.info(f"Data: {data_path}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Task Type: {settings.recipe.task_choice}")  # 🆕 학습시 정보

        # 배치 추론 실행
        run_inference_pipeline(
            settings=settings,
            run_id=run_id,
            data_path=data_path,
            context_params=params or {}
        )

        logger.info("✅ 배치 추론이 성공적으로 완료되었습니다.")

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"배치 추론 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)
```

### 4.4 Phase 4: 기존 함수 정리 및 마이그레이션

#### 4.4.1 settings/loader.py 정리

```python
# 🗑️ 삭제할 함수들
- create_settings_for_inference()  # 더미 Recipe 생성 함수
- load_config_files()              # 중복 함수

# ✅ 유지할 함수들
- load_settings()                  # SettingsFactory에서 내부 사용
- resolve_env_variables()          # 공통 유틸리티
- _load_config(), _load_recipe()   # 내부 헬퍼 함수

# 🆕 추가할 함수들
- migrate_legacy_settings()       # 기존 코드 호환성용 (임시)
```

#### 4.4.2 factory/artifact.py 완전 삭제

**현재 상태 분석**:
```python
# src/factory/artifact.py - 하위호환성 라우터만 제공
from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper  # noqa: F401
```

**사용처 확인**:
- `tests/unit/factory/test_component_creation_methods.py`: 테스트에서만 사용
- 실제 프로덕션 코드에서는 직접 `src.utils.integrations.pyfunc_wrapper` 사용

**삭제 계획**:
1. 테스트 파일의 import 경로 수정
2. `factory/artifact.py` 파일 완전 삭제
3. 모든 PyfuncWrapper import를 정식 경로로 통일

```python
# Before (삭제 예정)
from src.factory.artifact import PyfuncWrapper

# After (표준 경로)
from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper
```

#### 4.4.3 Pydantic 검증 로직 검토 및 최적화

**현재 검증 로직 현황**:

**🟡 검토 필요한 검증들**:
```python
# src/settings/recipe.py의 복잡한 검증들
@field_validator('fixed', 'tunable')  # 하이퍼파라미터 튜닝 검증
@field_validator('values')            # 고정값 검증
@model_validator(mode='after')        # 캘리브레이션 설정 검증
@field_validator('feature_views')     # FeatureStore 검증
# ... 총 8개의 복잡한 검증 로직

# src/settings/loader.py의 런타임 검증들
def _validate(self) -> None:                           # Config-Recipe 호환성
def validate_data_source_compatibility(self) -> None:  # 데이터 소스 호환성
```

**🔍 검증 필요성 분석**:

1. **과도한 검증으로 의심되는 부분들**:
   - `HyperparametersTuning`의 복잡한 tunable 파라미터 구조 검증
   - `DataInterface`의 세부적인 컬럼 존재 여부 검증
   - `validate_data_source_compatibility`의 파일 확장자 기반 어댑터 타입 추론

2. **유지해야 할 핵심 검증들**:
   - Feature Store provider와 fetcher 타입 일치성
   - 필수 필드 존재 여부 (Pydantic 기본 기능)
   - 캘리브레이션 활성화시 method 필수 검증

**🎯 최적화 제안**:
```python
# 🗑️ 삭제 검토 대상
- validate_data_source_compatibility(): Factory에서 런타임 체크로 충분
- HyperparametersTuning의 과도한 구조 검증: 실행시 오류로 충분
- DataInterface의 컬럼 검증: 실제 데이터 로드시 체크로 충분

# ✅ 유지할 검증들
- Feature Store 일관성 검증: 설정 오류 조기 발견 중요
- 캘리브레이션 설정 검증: 논리적 오류 방지
- 기본 타입 및 필수 필드 검증: Pydantic 표준 기능
```

**사용자와 상의할 사항**:
1. `validate_data_source_compatibility()` 완전 삭제 vs 단순화
2. `HyperparametersTuning`의 구조 검증 레벨 조정
3. 런타임 검증을 Factory 레벨로 이관할지 여부

#### 4.4.4 train_pipeline.py MLflow 저장 로직 강화

```python
def run_train_pipeline(settings: Settings, context_params: Optional[Dict] = None,
                      record_requirements: bool = False):
    """학습 파이프라인 실행 + 완전한 재현성 저장"""

    # ... 기존 학습 로직 ...

    # 🆕 완전한 재현성을 위한 저장 (마지막 단계)
    log_complete_training_artifacts(
        settings=settings,
        recipe_path=context_params.get('original_recipe_path'),  # CLI에서 전달
        config_path=context_params.get('original_config_path'),  # CLI에서 전달
        context_params=context_params,
        python_model=pyfunc_model,
        signature=signature,
        input_example=input_example
    )
```

## 5. 단계별 실행 방안

### 5.1 1단계: MLflow 저장/복원 시스템 구축 (1-2주)

**목표**: 학습시 Recipe 완전 저장, 추론시 완전 복원

**작업 항목**:
1. `MLflowRecipeRestorer` 클래스 구현
2. `log_complete_training_artifacts` 함수 구현
3. 기존 `train_pipeline.py`에 저장 로직 추가
4. 단위 테스트 작성

**검증 방법**:
```bash
# 학습 실행
mmp train --recipe-path recipes/test.yaml --config-path configs/dev.yaml --data-path data/test.csv

# MLflow UI에서 확인
# - training_artifacts/recipe_snapshot.yaml 존재
# - training_artifacts/settings_metadata.json 존재

# 복원 테스트 (Python)
restorer = MLflowRecipeRestorer(run_id)
recipe = restorer.restore_recipe()
assert recipe.name == "원본과 동일"
```

### 5.2 2단계: 통합 Settings 팩토리 구현 (1주)

**목표**: `SettingsFactory` 구현 및 내부 테스트

**작업 항목**:
1. `SettingsFactory` 클래스 구현
2. 기존 `train_command` 로직을 `for_training` 메서드로 이관
3. `for_inference`, `for_serving` 메서드 구현
4. 통합 테스트 작성

**검증 방법**:
```python
# 학습용 테스트
settings1 = SettingsFactory.for_training(recipe_path, config_path, data_path)

# 추론용 테스트
settings2 = SettingsFactory.for_inference(config_path, run_id)

# 핵심 정보 일치 확인
assert settings1.recipe.task_choice == settings2.recipe.task_choice
assert settings1.recipe.data.data_interface == settings2.recipe.data.data_interface
```

### 5.3 3단계: CLI 명령어 일관성 개선 (1주)

**목표**: 3개 CLI 명령어 모두 `SettingsFactory` 사용

**작업 항목**:
1. `train_command` 단순화 (60줄 → 15줄)
2. `serve_command`, `inference_command` Recipe 복원 로직 추가
3. 에러 처리 표준화
4. 통합 E2E 테스트

**검증 방법**:
```bash
# 전체 워크플로우 테스트
# 1. 학습
mmp train --recipe-path recipes/test.yaml --config-path configs/dev.yaml --data-path data/train.csv

# 2. 서빙 (학습시 Recipe로 실행)
mmp serve-api --run-id {학습_RUN_ID} --config-path configs/prod.yaml

# 3. 추론 (학습시 Recipe로 실행)
mmp batch-inference --run-id {학습_RUN_ID} --config-path configs/prod.yaml --data-path data/inference.csv

# 결과: 모든 단계에서 동일한 Recipe 설정 사용됨
```

### 5.4 4단계: 레거시 코드 정리 및 최적화 (1주)

**목표**: 불필요한 함수 제거, 성능 최적화, 검증 로직 최적화

**작업 항목**:
1. `create_settings_for_inference`, `load_config_files` 함수 제거
2. `factory/artifact.py` 완전 삭제 및 import 경로 정리
3. Pydantic 검증 로직 최적화 (사용자와 상의 후)
4. Recipe 복원 성능 최적화 (캐싱 등)
5. 문서화 업데이트 및 마이그레이션 가이드 작성

**Pydantic 검증 최적화 세부 작업**:
1. `validate_data_source_compatibility()` 삭제/단순화 결정
2. `HyperparametersTuning` 검증 로직 간소화
3. 런타임 검증을 Factory 레벨로 이관
4. 성능 테스트 및 검증

## 6. 예상 효과

### 6.1 재현성 보장
- **학습 → 추론 → 서빙**: 100% 동일한 Recipe 설정 사용
- **디버깅 용이성**: 문제 발생시 정확한 학습 환경 재현 가능
- **MLOps 표준 준수**: 실험 재현성 완전 보장

### 6.2 코드 일관성 향상
- **CLI 명령어 통일**: 3개 명령어 모두 동일한 패턴과 길이
- **로직 중앙화**: 복잡한 설정 로직이 한 곳에 집중
- **유지보수성**: 새로운 기능 추가시 일관된 방식 적용

### 6.3 개발 생산성 증대
- **테스트 용이성**: Settings 생성 로직을 독립적으로 테스트 가능
- **확장성**: 새로운 CLI 명령어 추가시 기존 팩토리 재사용
- **오류 감소**: 표준화된 설정 생성으로 휴먼 에러 최소화

## 7. 위험 요소 및 대응 방안

### 7.1 호환성 문제
**위험**: 기존 MLflow Run들은 Recipe 스냅샷이 없음
**대응**:
- Legacy Run 감지 로직 구현
- 기존 더미 Recipe 생성 방식을 fallback으로 유지
- 점진적 마이그레이션 지원

### 7.2 성능 문제
**위험**: MLflow에서 Recipe 복원시 네트워크 지연
**대응**:
- Recipe 캐싱 시스템 구현
- 비동기 복원 로직 적용
- 서빙시 startup 단계에서 미리 로드

### 7.3 MLflow 의존성
**위험**: MLflow 서비스 장애시 추론 불가
**대응**:
- Recipe 로컬 캐싱 구현
- 오프라인 모드 지원
- 대체 스토리지 옵션 제공

## 8. 개선 사항 요약

### 8.1 사용자 제안사항 반영

✅ **1. 클래스명 단순화**: `UnifiedSettingsFactory` → `SettingsFactory`
- 더 직관적이고 간결한 명명
- 모든 문서와 코드 예시에서 일관되게 적용

✅ **2. factory/artifact.py 완전 삭제**:
- 하위호환성 라우터 역할만 하는 불필요한 파일 제거
- 테스트 파일의 import 경로를 정식 경로로 수정
- 코드베이스 정리 및 import 경로 일관성 확보

✅ **3. Pydantic 검증 로직 검토 및 최적화**:
- 과도한 검증으로 의심되는 부분들 식별
- 런타임 검증을 Factory 레벨로 이관하는 방안 검토
- 사용자와 상의 후 불필요한 검증 로직 제거 예정

### 8.2 핵심 개선 효과

**🎯 단순성 증대**:
- 클래스명 단순화로 개발자 경험 개선
- 불필요한 파일 제거로 코드베이스 정리
- 과도한 검증 제거로 성능 향상

**🔧 유지보수성 향상**:
- import 경로 일관성으로 리팩토링 용이
- 검증 로직 중앙화로 수정 포인트 최소화
- 레거시 코드 제거로 기술 부채 해소

**⚡ 성능 최적화**:
- 불필요한 검증 로직 제거
- Recipe 복원 캐싱 시스템 도입
- 런타임 검증 최적화

## 9. 결론

현재 MLOps 시스템의 가장 중대한 문제인 **학습-추론 간 설정 불일치**를 완전히 해결하는 종합적인 계획입니다.

**핵심 원칙 (개선됨)**:
1. **완전 재현성**: MLflow 기반 Recipe 완전 저장/복원
2. **일관성**: 모든 CLI 명령어가 `SettingsFactory` 사용
3. **단순성**: 클래스명 단순화, 불필요한 파일/검증 제거
4. **점진적 적용**: 기존 코드 호환성 유지하며 단계적 개선
5. **성능 최적화**: 검증 로직 최적화 및 캐싱 시스템 도입

**추가 개선 사항**:
- 사용자 친화적인 네이밍 컨벤션 적용
- 코드베이스 정리 및 기술 부채 해소
- Pydantic 검증 성능 최적화

이 개선된 계획을 통해 **Production-Ready MLOps 시스템**으로 진화할 수 있으며, 모든 ML 워크플로우에서 **100% 재현 가능한 환경**을 보장할 수 있습니다.