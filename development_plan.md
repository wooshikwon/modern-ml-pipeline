# Modern ML Pipeline - 상세 개발 계획서

## 1. 프로젝트 목표

본 문서는 `blueprint.md`의 철학과 `next_step.md`의 로드맵을 기반으로, Modern ML Pipeline을 성공적인 라이브러리로 전환하기 위한 **구체적이고 실행 가능한 수준의 상세 개발 계획**을 정의한다.

모든 개발은 코드의 최종 구현뿐만 아니라, 전체 시스템과의 **구조적 일관성, 인터페이스 호환성, 명확한 책임 분리**를 보장하는 것을 최우선 목표로 한다.

## 2. 개발 원칙

1.  **Top-Down-Design:** 항상 전체 파이프라인의 흐름(`main` -> `pipelines` -> `engine` -> `components`)을 먼저 고려하고, 개별 파일의 구현이 상위 흐름과 어떻게 통합될지를 정의한다.
2.  **Interface First:** 함수를 구현하기 전에, 해당 함수가 파이프라인의 다른 부분과 주고받을 **인자(Arguments)와 반환값(Return Values)의 형태와 타입을 명확히 정의**한다.
3.  **Dependency Inversion:** 상위 모듈은 하위 모듈의 구체적인 구현에 의존해서는 안 된다. `Factory`와 `Registry`를 통해 의존성을 주입하여 결합도를 낮춘다.

---

## 3. 상세 개발 계획

### **Phase 1: 기반 재설계**

#### **Step 1.1: 3계층 아키텍처 확립 및 Import 경로 수정**

- **목표:** `src/core` 디렉토리에 대한 모든 의존성을 제거하고, 새로운 3계층 아키텍처(`components`, `engine`, `pipelines`)에 맞는 `import` 경로를 적용하여 시스템의 기본 구조를 안정화한다.
- **개발 순서:**
    1. 아래 명시된 모든 파일의 `from src.core...` 구문을 새로운 경로로 수정한다.
    2. 경로 수정 후, `src/core` 디렉토리를 삭제한다.
    3. `pytest`를 실행하여 최소한의 기본 테스트가 깨지지 않는지 확인한다.
- **파일별 개발 계획:**

| 파일 경로 | 기존 Import | 변경 후 Import | 비고 |
| :--- | :--- | :--- | :--- |
| `src/interface/base_trainer.py` | `from src.core.preprocessor import Preprocessor` | `from src.components.preprocessor import Preprocessor` | `Preprocessor`는 이제 ML 컴포넌트임. |
| `src/interface/base_factory.py` | `from src.core.preprocessor import Preprocessor` | `from src.components.preprocessor import Preprocessor` | `Preprocessor`는 이제 ML 컴포넌트임. |
| `src/pipelines/train_pipeline.py` | `from src.core.factory import Factory`<br>`from src.core.trainer import Trainer` | `from src.engine.factory import Factory`<br>`from src.components.trainer import Trainer` | `Factory`는 엔진, `Trainer`는 컴포넌트. |
| `src/pipelines/inference_pipeline.py` | `from src.core.factory import Factory` | `from src.engine.factory import Factory` | `Factory`는 시스템 엔진의 일부. |
| `src/utils/adapters/feature_store_adapter.py` | `from src.core.factory import Factory` | `from src.engine.factory import Factory` | Phase 2에서 삭제될 파일이지만 임시 수정. |
| `src/components/trainer.py` | `from src.core.augmenter import BaseAugmenter`<br>`from src.core.preprocessor import BasePreprocessor`<br>`from src.core.factory import Factory`<br>`from src.core.evaluator import ...` | `from src.components.augmenter import BaseAugmenter`<br>`from src.components.preprocessor import BasePreprocessor`<br>`from src.engine.factory import Factory`<br>`from src.components.evaluator import ...` | 컴포넌트 간에는 상대경로 `from .` 사용도 가능. |
| `src/engine/factory.py` | `from src.core.augmenter import ...`<br>`from src.core.preprocessor import ...`<br>`from src.core.registry import AdapterRegistry`<br>`from src.core.evaluator import ...` | `from src.components.augmenter import ...`<br>`from src.components.preprocessor import ...`<br>`from src.engine.registry import AdapterRegistry`<br>`from src.components.evaluator import ...` | `Factory`는 이제 `components`를 생성함. |

- **주의사항:**
    - 수정 후 `git status`를 통해 `src/core` 관련 파일이 `deleted`로 표시되는지 반드시 확인해야 한다.
    - 순환 참조(Circular Import) 오류가 발생할 경우, import 위치를 함수 내부로 옮기는 등의 추가 조치가 필요할 수 있다.

#### **Step 1.2: Registry & Factory 역할 재정립**

- **목표:** 복잡하고 암시적인 로직을 제거하고, "Registry는 명시적 등록", "Factory는 단순 생성"이라는 명확한 단일 책임 원칙(SRP)을 확립한다.
- **개발 순서:**
    1. `src/engine/registry.py`를 아래 계획에 따라 완전히 새로 작성한다.
    2. `src/engine/factory.py`가 새로운 `registry.py`를 사용하도록 수정하고, 불필요한 로직을 모두 제거한다.
    3. `config/` 디렉토리에 임시로 `data_adapters.yaml` 파일을 생성하여, 변경된 `Factory`가 설정을 읽어오는 방식을 테스트할 수 있도록 준비한다.

- **파일별 개발 계획:**

  - **`src/engine/registry.py`**
    - **Class `AdapterRegistry`:**
        - `_adapters: Dict[str, Type[BaseAdapter]] = {}`: 어댑터를 저장할 클래스 변수.
        - `register(adapter_type: str, adapter_class: Type[BaseAdapter])`: 데코레이터가 아닌, 타입을 직접 받아 등록하는 단순한 클래스 메서드.
        - `create(adapter_type: str, settings: Settings, **kwargs) -> BaseAdapter`: `_adapters` 딕셔너리에서 `adapter_type`을 찾아 인스턴스를 생성하여 반환.
    - **Function `register_all_adapters()`:**
        - 시스템에서 사용하는 모든 어댑터를 명시적으로 등록하는 함수.
        - Phase 2에서 통합 어댑터가 만들어지기 전까지, 기존 개별 어댑터들을 임시로 등록하는 `_register_legacy_adapters_temporarily()` 헬퍼 함수를 호출한다. (순환 참조 방지를 위해 `importlib` 사용)
    - **제거 대상:** 데코레이터, 자동 스캔, 메타클래스 등 모든 암시적 로직.

  - **`src/engine/factory.py`**
    - `import`문 변경: `from src.engine.registry import AdapterRegistry`
    - **Method `create_data_adapter(self, adapter_type: str)`:**
        - **인자 변경:** 더 이상 `adapter_purpose`나 `source_path`를 받지 않고, `adapter_type` 문자열(e.g., "sql", "storage")만 받도록 인터페이스를 단순화한다.
        - **로직 변경:** 내부의 모든 설정 파싱, 타입 매핑, 클래스 조회 로직을 제거한다. 오직 `return AdapterRegistry.create(adapter_type, self.settings)` 한 줄만 남겨 책임을 `Registry`에 완전히 위임한다.
    - **제거 대상:**
        - `_get_adapter_class()` 메서드.
        - `create_data_adapter_legacy()` 메서드.
        - `create_redis_adapter()` 등 특정 어댑터 전용 생성 메서드 (이제 `create_data_adapter("redis")`로 대체 가능).
        - `settings.data_adapters` 설정을 파싱하는 모든 로직.

- **주의사항:**
    - 이 단계는 시스템의 심장을 수술하는 것과 같습니다. 수정 후에는 `train_pipeline`과 `inference_pipeline`이 `Factory`를 통해 어댑터를 생성하는 부분이 정상적으로 동작하는지 집중적으로 확인해야 합니다.
    - 임시 `data_adapters.yaml` 파일은 Phase 3에서 정식으로 구조화될 것입니다. 지금은 테스트를 위한 최소한의 설정만 포함합니다.

### **Phase 2: 어댑터 현대화 및 통합**

#### **Step 2.1: 통합 `SqlAdapter`, `StorageAdapter`, `FeastAdapter` 구현**

- **목표:** 업계 표준 라이브러리(`SQLAlchemy`, `fsspec`, `feast`)를 기반으로 하는 가볍고 강력한 통합 어댑터를 구현하여, 기존의 복잡하고 개별적인 어댑터들을 대체한다.
- **개발 순서:**
    1. `pyproject.toml`에 `sqlalchemy`, `google-cloud-bigquery-storage`, `psycopg2-binary`, `fsspec`, `gcsfs`, `s3fs`, `feast` 등 필요한 라이브러리 의존성을 추가한다.
    2. 각 어댑터 파일을 아래 계획에 따라 신규 생성한다.
- **파일별 개발 계획:**

  - **`src/utils/adapters/sql_adapter.py`**
    - `import sqlalchemy`
    - **Class `SqlAdapter(BaseAdapter)`:**
        - `__init__(self, settings: Settings, **kwargs)`:
            - `settings` 객체로부터 현재 환경에 맞는 DB 연결 정보를 읽어, `SQLAlchemy`가 요구하는 연결 URI(e.g., `"bigquery://project-id/dataset"`, `"postgresql://user:pass@host/db"`)를 동적으로 생성하는 `_build_connection_uri()` 내부 메서드를 호출한다.
            - `sqlalchemy.create_engine()`을 사용하여 DB 엔진을 생성하고, `self.engine`에 저장한다.
        - `read(self, sql_query: str, **kwargs) -> pd.DataFrame`:
            - `pd.read_sql(sql_query, self.engine, **kwargs)`를 호출하여 쿼리를 실행하고 결과를 반환한다.
        - `write(self, df: pd.DataFrame, table_name: str, **kwargs)`:
            - `df.to_sql(table_name, self.engine, **kwargs)`를 호출하여 데이터를 DB에 쓴다.

  - **`src/utils/adapters/storage_adapter.py`**
    - `import fsspec`
    - **Class `StorageAdapter(BaseAdapter)`:**
        - `__init__(self, settings: Settings, **kwargs)`:
            - `settings` 객체로부터 GCS/S3 등에 필요한 인증 정보를 읽어 `fsspec`이 요구하는 `storage_options` 딕셔너리를 생성하고 `self.storage_options`에 저장한다.
        - `read(self, uri: str, **kwargs) -> pd.DataFrame`:
            - `pd.read_parquet(uri, storage_options=self.storage_options, **kwargs)`를 호출한다. `fsspec`이 URI 스킴(`gs://`, `s3://`, `file://`)을 보고 알아서 처리한다.
        - `write(self, df: pd.DataFrame, uri: str, **kwargs)`:
            - `df.to_parquet(uri, storage_options=self.storage_options, **kwargs)`를 호출한다.

  - **`src/utils/adapters/feast_adapter.py`**
    - `from feast import FeatureStore`
    - **Class `FeastAdapter(BaseAdapter)`:**
        - `__init__(self, settings: Settings, **kwargs)`:
            - `settings.feature_store.feast_config`에서 설정을 `dict` 형태로 가져온다.
            - **`FeatureStore(config=feast_config_dict)`**를 호출하여 `feast` 스토어 객체를 초기화한다. (기존의 복잡한 임시 파일 생성 로직 완전 제거)
        - `get_historical_features(...)`: `self.store.get_historical_features(...).to_df()`를 호출하는 래퍼.
        - `get_online_features(...)`: `self.store.get_online_features(...).to_df()`를 호출하는 래퍼.
        - `read` / `write` 메서드는 `BaseAdapter` 호환성을 위해 형식적으로 구현하거나, 필요 시 `get_historical_features`를 호출하도록 구현한다.

- **주의사항:**
    - 각 라이브러리의 인증 처리 방식을 사전에 명확히 파악해야 한다. (e.g., `gcloud auth application-default login`, `aws configure` 등 환경 변수 기반 인증 활용 권장)
    - 기존 `feature_store_adapter.py`에 있던 시뮬레이션/Fallback 로직은 모두 제거하고, 각 어댑터는 자신의 책임에만 집중하도록 구현한다.

#### **Step 2.2: `AdapterRegistry` 업데이트 및 레거시 어댑터 제거**

- **목표:** 새로운 통합 어댑터들을 시스템의 기본 I/O 채널로 공식 등록하고, 구시대의 유물인 모든 레거시 개별 어댑터들을 시스템에서 완전히 제거하여 아키텍처를 단순화하고 명확성을 확보한다.
- **개발 순서:**
    1. `src/engine/registry.py`의 `register_all_adapters` 함수를 수정한다.
    2. `src/utils/adapters/` 디렉토리에서 모든 레거시 어댑터 파일을 삭제한다.
    3. `__init__.py` 파일 등에서 발생하는 깨진 `import`문들을 정리한다.
- **파일별 개발 계획:**

  - **`src/engine/registry.py`**
    - **Function `register_all_adapters()` 수정:**
        - 기존의 임시 등록 함수(`_register_legacy_adapters_temporarily()`) 호출 코드를 삭제한다.
        - `try-except` 블록을 사용하여 `SqlAdapter`, `StorageAdapter`, `FeastAdapter`를 명시적으로 `import`하고 등록하는 로직만 남긴다.
        ```python
        def register_all_adapters():
            logger.info("모든 코어 어댑터 등록을 시작합니다...")
            try:
                from src.utils.adapters.sql_adapter import SqlAdapter
                AdapterRegistry.register("sql", SqlAdapter)
            except ImportError:
                logger.warning("SqlAdapter를 찾을 수 없어 등록을 건너뜁니다.")
            # ... StorageAdapter, FeastAdapter 등록 로직 ...
        ```
  
  - **`src/utils/adapters/` 디렉토리**
    - **파일 삭제:**
        - `bigquery_adapter.py`
        - `postgresql_adapter.py`
        - `gcs_adapter.py`
        - `s3_adapter.py`
        - `file_system_adapter.py`
        - `feature_store_adapter.py`
        - `redis_adapter.py`
        - ... 등 `sql_`, `storage_`, `feast_`로 시작하지 않는 모든 어댑터 파일.
    - **`__init__.py` 파일 수정:**
        - 깨진 `import` 구문들을 모두 제거한다. 이제 `__init__.py`는 거의 비어있거나, 새로운 통합 어댑터들을 `export`하는 역할만 하게 될 수 있다.

- **주의사항:**
    - 이 단계는 많은 파일을 삭제하므로, `git` 버전 관리 하에 신중하게 진행해야 한다.
    - 파일 삭제 후, 전체 프로젝트에서 `BigQueryAdapter` 등 삭제된 클래스명을 직접 `import`하거나 사용하는 부분이 남아있는지 `grep`으로 최종 확인하는 것이 안전하다.

### **Phase 3: CLI 및 설정 파이프라인 구축**

#### **Step 3.1: Config 파일 구조 개편**

- **목표:** 설정 파일의 관심사를 명확히 분리하여 유지보수성을 높인다. `data_adapters.yaml`은 "어떤 기술을 쓸지", `base.yaml`은 "어디에 연결할지"를 정의하도록 역할을 나눈다.
- **개발 순서:**
    1. `config/data_adapters.yaml` 파일을 신규 생성한다.
    2. `config/base.yaml`에서 `data_adapters` 섹션을 잘라내어 `data_adapters.yaml`로 옮긴다.
    3. `src/settings/loaders.py`의 `load_config_files` 함수를 수정하여 새로운 파일을 로드하도록 변경한다.
    4. `config/base.yaml`에서 현재 사용되지 않는 불필요한 섹션을 정리한다.
- **파일별 개발 계획:**

  - **`config/data_adapters.yaml` (신규 생성)**
    - **내용:** `base.yaml`에서 옮겨온 `data_adapters` 섹션을 붙여넣는다. 이 파일은 이제 어댑터 기술 스택 정의의 유일한 출처(SSoT)가 된다.
    - **향후 계획:** Phase 2에서 통합 어댑터가 완성되면, 이 파일의 `default_loader` 등은 `"sql"` 이나 `"storage"` 와 같은 통합된 타입으로 변경될 것이다.

  - **`config/base.yaml`**
    - **제거 대상:** `data_adapters` 섹션 전체.
    - **정리 대상 (선택):** `performance_monitoring` 등 현재 활발히 사용되지 않거나, Pydantic 모델의 기본값으로 충분한 설정들을 제거하여 파일을 단순화한다.

  - **`src/settings/loaders.py`**
    - **Function `load_config_files()` 수정:**
        - 기존 로직을 수정하여 `data_adapters.yaml`을 가장 먼저 로드하고, 그 위에 `base.yaml`, `env.yaml`을 순차적으로 덮어쓰는(`_recursive_merge`) 구조로 변경한다.
        ```python
        def load_config_files() -> Dict[str, Any]:
            config_dir = BASE_DIR / "config"
            
            # 1. 어댑터 설정 로드
            adapter_config = _load_yaml_with_env(config_dir / "data_adapters.yaml")
            
            # 2. 기본 인프라 설정 로드
            base_config = _load_yaml_with_env(config_dir / "base.yaml")
            
            # 3. 환경별 설정 로드
            app_env = os.getenv("APP_ENV", "local")
            env_config = _load_yaml_with_env(config_dir / f"{app_env}.yaml")
            
            # 4. 순차적 병합 (오른쪽이 왼쪽을 덮어씀)
            merged_config = _recursive_merge(adapter_config, base_config)
            merged_config = _recursive_merge(merged_config, env_config)
            
            return merged_config
        ```

- **주의사항:**
    - 설정 파일 분리 후, 기존의 모든 테스트(`pytest`)가 여전히 설정을 올바르게 읽어와 통과하는지 반드시 확인해야 한다.
    - `Settings` Pydantic 모델(`models.py`)은 현재 구조 변경 없이 그대로 유지되므로, `loaders.py`의 병합 로직만 정확히 구현하면 된다.

#### **Step 3.2: "설정 빌더" 파이프라인 구현 (Jinja 포함)**

- **목표:** CLI를 통해 동적 파라미터를 주입받아 SQL 템플릿을 렌더링하는, `[YAML 로드 → Jinja 렌더링 → Pydantic 검증]`의 3단계 설정 처리 파이프라인을 완성한다.
- **개발 순서:**
    1. `pyproject.toml`에 `Jinja2` 의존성을 추가한다.
    2. `src/utils/system/templating_utils.py` 파일을 신규 생성하고, 템플릿 렌더링 함수를 구현한다.
    3. `src/settings/loaders.py`의 `load_settings_by_file` 함수를 수정하여, Jinja 렌더링 단계를 파이프라인에 통합한다.
- **파일별 개발 계획:**

  - **`src/utils/system/templating_utils.py` (신규 생성)**
    - `import jinja2`
    - **Function `render_sql_template(template_path: str, context: dict) -> str`:**
        - `jinja2.Environment`와 `FileSystemLoader`를 사용하여 템플릿 파일을 로드한다.
        - `template.render(context)`를 호출하여 템플릿을 렌더링하고, 결과 SQL 문자열을 반환한다.

  - **`src/settings/loaders.py`**
    - **Function `load_settings_by_file(recipe_file: str, context_params: Optional[Dict[str, Any]] = None)` 수정:**
        - **인자 추가:** CLI로부터 `context-params`를 받을 수 있도록 `context_params` 인자를 추가한다.
        - **로직 변경:**
            1. (기존 로직) `load_config_files()`와 `load_recipe_file()`을 호출하여 설정 딕셔너리를 병합한다. (`final_data`)
            2. **(Jinja 렌더링 단계 추가)**
               - `final_data`에서 `loader`와 `augmenter` 설정을 확인한다.
               - `source_uri`가 `.sql.j2`로 끝나고, `context_params`가 제공된 경우에만 렌더링을 수행한다.
               - `templating_utils.render_sql_template`를 호출하여 렌더링된 SQL을 받는다.
               - `final_data` 딕셔너리 내의 `source_uri` 값을 렌더링된 SQL 문자열로 **덮어쓴다.**
            3. (기존 로직) 모든 처리가 끝난 `final_data` 딕셔너리를 `Settings(**final_data)`에 전달하여 최종 `Settings` 객체를 생성하고 반환한다.

- **주의사항:**
    - Jinja 렌더링은 **반드시 Pydantic 검증 이전에** 일반 딕셔너리를 대상으로 수행되어야 한다. Pydantic 모델은 불변(immutable)성을 지향하므로, 생성된 후에 값을 바꾸는 것은 안티패턴이다.
    - `main.py`는 `train` 커맨드에서 `--context-params` 옵션을 받아 이 함수에 올바르게 전달하도록 수정되어야 한다 (Step 3.3에서 다룸).
    - `recipes/` 디렉토리에 테스트용 `.sql.j2` 템플릿 파일과, 이를 사용하는 `templated_exp.yaml` 예제를 추가하여 렌더링 로직을 검증해야 한다.

#### **Step 3.3: 신규 CLI 인터페이스 구현**

- **목표:** 라이브러리 사용자에게 명확하고 실용적인 CLI 인터페이스를 제공하여, ML 파이프라인 실행뿐만 아니라 프로젝트 초기화, 설정 검증 등 다양한 상호작용을 가능하게 한다.
- **개발 순서:**
    1. `pyproject.toml`에 `typer` 의존성을 추가한다.
    2. `main.py`의 전체 구조를 `typer` 기반으로 재작성한다.
    3. 각 서브커맨드에 해당하는 함수와 로직을 구현한다.
- **파일별 개발 계획:**

  - **`main.py`**
    - **구조:**
        ```python
        import typer
        from typing_extensions import Annotated
        # ... other imports ...

        app = typer.Typer(help="Modern ML Pipeline - A robust tool for building and deploying ML.")

        @app.command()
        def train(...):
            # ...

        @app.command()
        def batch_inference(...):
            # ...

        # ... other commands ...

        if __name__ == "__main__":
            app()
        ```
    - **커맨드별 상세 계획:**
        - `**train(recipe_file: str, context_params: Optional[str] = None)**`:
            - `--context-params`를 `json.loads()`로 파싱한다.
            - `load_settings_by_file(recipe_file, context_params)`를 호출하여 `Settings` 객체를 생성한다.
            - `run_training(settings)`를 호출한다.
        - `**batch_inference(run_id: str, context_params: Optional[str] = None)**`:
            - `train`과 유사하게 `context_params`를 파싱하고 `Settings` 객체를 로드한다.
            - `run_batch_inference(settings, run_id, context_params)`를 호출한다.
        - `**serve_api(run_id: str, ...)**`:
            - 기존 로직과 거의 동일하게 유지한다.
        - `**init(dir: str = ".")**`:
            - `pathlib.Path`를 사용하여 지정된 디렉토리(`dir`)에 `config/`와 `recipes/` 폴더를 생성한다.
            - `config/`에는 기본 `base.yaml`과 `data_adapters.yaml`을, `recipes/`에는 예시 `example_recipe.yaml` 파일을 생성해주는 로직을 구현한다.
            - **`data_adapters.yaml` 기본 내용:** `local` 환경에서 즉시 동작할 수 있도록, `default_loader`와 `default_storage`를 `"storage"`(통합 `StorageAdapter`)로 지정하는 내용을 포함한다.
              ```yaml
              # config/data_adapters.yaml (init 시 생성될 기본 내용)
              data_adapters:
                default_loader: storage
                default_storage: storage
                adapters:
                  storage:
                    class_name: StorageAdapter # 통합 StorageAdapter 사용
                    config: {}
                  sql:
                    class_name: SqlAdapter     # 통합 SqlAdapter
                    config:
                      connection_uri: "postgresql://user:pass@localhost:5432/db" # 예시
              ```
        - `**validate(recipe_file: str)**`:
            - `try-except` 블록 안에서 `load_settings_by_file(recipe_file)`를 호출한다.
            - 성공 시 성공 메시지를, `pydantic.ValidationError`나 `ValueError` 발생 시 에러 메시지를 `typer.echo`를 통해 출력한다.
        - `**test-contract()**`:
            - `subprocess.run(["pytest", "tests/integration/test_dev_contract.py"])`를 호출하여 통합 테스트를 실행한다.

- **주의사항:**
    - `jsonargparse` 대신 `Typer`를 사용하기로 결정했으므로, YAML 파일의 특정 값을 CLI로 덮어쓰는 기능은 이번 범위에 포함하지 않는다. 이는 우리의 3번 요구사항과 일치한다.
    - 각 커맨드 함수는 `try-except` 블록으로 감싸, 사용자에게 친절한 에러 메시지를 보여주고 비정상 종료(exit code 1)되도록 처리해야 한다.

### **Phase 4: 최종 통합 및 검증**

#### **Step 4.1: `PyfuncWrapper` 저장 로직 검증**

- **목표:** Jinja 템플릿 렌더링의 최종 결과물인 **완성된 SQL 문자열**이 `PyfuncWrapper`의 `loader_sql_snapshot`에 정확히 저장되는 것을 보장하여, 동적 쿼리의 유연성과 추론 시점의 완벽한 재현성을 모두 달성한다.
- **개발 순서:**
    1. `src/engine/factory.py`의 `create_pyfunc_wrapper`와 `_create_loader_sql_snapshot` 메서드를 다시 한번 검토한다.
    2. `_create_loader_sql_snapshot`이 `self.settings.model.loader.source_uri` 값을 읽는 현재 로직이, **이미 `loaders.py`에서 렌더링이 완료된 `Settings` 객체를 전달받기 때문에** 수정 없이도 올바르게 동작함을 확인한다.
    3. 필요한 경우, 렌더링된 SQL이 아닌 원본 템플릿 파일 경로를 읽으려는 로직이 남아있다면 제거한다. (e.g., `source_uri`에서 파일 경로를 파싱하려는 로직)
- **파일별 개발 계획:**

  - **`src/engine/factory.py`**
    - **Method `_create_loader_sql_snapshot(self) -> str` (검토 및 확인):**
        - 이 함수가 `self.settings.model.loader.source_uri` 값을 **그대로 반환**하도록 보장해야 한다.
        - 만약 이전에 URI에서 파일 경로를 추출하거나 파일을 다시 읽는 로직이 있었다면, 그 부분을 반드시 제거해야 한다. `source_uri` 자체가 이미 완성된 SQL 컨텐츠이기 때문이다.
        ```python
        # AS-IS 예시 (잘못된 로직)
        # def _create_loader_sql_snapshot(self):
        #     path = Path(self.settings.model.loader.source_uri)
        #     return path.read_text()

        # TO-BE (올바른 로직)
        def _create_loader_sql_snapshot(self):
            # settings 객체에 이미 렌더링된 SQL이 담겨 있으므로 그대로 반환
            return self.settings.model.loader.source_uri
        ```

- **주의사항:**
    - 이 단계는 코드 변경보다는 **데이터 흐름에 대한 아키텍처적 검증**에 가깝다. `loaders.py`에서 `factory.py`까지 `Settings` 객체가 어떻게 전달되고 사용되는지 전체 흐름을 이해하는 것이 중요하다.
    - 실제 동작은 `Step 4.2`의 E2E 테스트에서 최종적으로 검증된다.

#### **Step 4.2: End-to-End 통합 테스트**

- **목표:** 리팩토링된 전체 시스템이 실제 사용자 시나리오 하에서 의도한 대로 완벽하게 동작하는지 종합적으로 검증한다.
- **개발 순서 (테스트 시나리오):**
    1. **프로젝트 초기화:** `uv run python main.py init --dir ./test_project`를 실행하여 새로운 프로젝트 구조가 생성되는지 확인한다.
    2. **설정 검증:** `uv run python main.py validate --recipe-file ./test_project/recipes/example_recipe.yaml`를 실행하여 성공 메시지를 확인한다.
    3. **템플릿 기반 학습:** 테스트용 Jinja 템플릿(`.sql.j2`)과 이를 사용하는 레시피를 준비하고, `uv run python main.py train --recipe-file ... --context-params '{"key": "value"}'`를 실행한다.
    4. **아티팩트 검증:** MLflow UI에서 생성된 `run`을 확인하고, `loader_sql_snapshot` 아티팩트를 다운로드하여 SQL이 `context-params`에 맞게 올바르게 렌더링되었는지 직접 확인한다.
    5. **추론/서빙 검증:** `train`에서 얻은 `run-id`를 사용하여 `batch-inference`와 `serve-api` 커맨드가 모두 오류 없이 실행되는지 확인한다.
    6. **인프라 계약 검증:** `uv run python main.py test-contract`를 실행하여 테스트가 통과하는지 확인한다.
- **주의사항:**
    - 이 테스트는 Mocking을 최소화하고, 실제 `mmp-local-dev` 환경과 연동하여 수행하는 것을 원칙으로 한다.
    - 각 단계에서 예상되는 출력(생성된 파일, CLI 메시지, MLflow 아티팩트)을 명확히 정의하고 검증해야 한다.

#### **Step 4.3: 사용자 문서 현대화**

- **목표:** 모든 아키텍처 및 기능 변경 사항을 사용자 문서에 정확하게 반영하여, 라이브러리 사용자가 새로운 시스템을 원활하게 사용할 수 있도록 돕는다.
- **개발 순서:**
    1. `README.md`의 "빠른 시작" 섹션을 새로운 CLI 커맨드(`init`, `validate`, `train`) 중심으로 재작성한다.
    2. `docs/DEVELOPER_GUIDE.md`에 변경된 설정 파일 구조(`data_adapters.yaml`), Jinja 템플릿 사용법, 통합 어댑터의 개념 등을 추가한다.
    3. `docs/INFRASTRUCTURE_STACKS.md`를 검토하여, `FeastAdapter`와 통합 어댑터의 도입으로 인한 확장성 증대 효과를 반영한다.
    4. 모든 문서에서 `v17.0`과 같은 오래된 버전 정보를 제거하고, 최신 아키텍처와 일관성을 맞춘다.
- **주의사항:**
    - 문서는 라이브러리의 얼굴이다. 최종 사용자의 관점에서, 복잡한 내부 구조를 몰라도 쉽게 따라할 수 있도록 명확하고 간결하게 작성해야 한다.
    - 코드 예제는 모두 새로운 CLI와 설정 파일 구조에 맞춰 수정되어야 한다.
