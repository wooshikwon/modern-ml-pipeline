# Output 저장 기능 설계 및 구현 계획

## 목표
- 배치 추론 결과와 전처리 완료 결과의 저장 대상을 사용자가 "데이터 소스"로 선택하면, 저장 형태(file/table) 및 세부 파라미터는 선택된 소스 유형에 따라 자동 결정되도록 한다.
- 각 결과는 저장을 하지 않는 옵션을 지원한다.
- 기존 설정 생성 흐름(템플릿, 빌더, 환경변수 치환)과 동일한 UX와 구조를 유지한다.

## 요구사항 요약
- 두 타겟: inference 결과, preprocessed 결과
- 저장 대상: Local Files, S3, GCS, PostgreSQL, BigQuery 중 한 가지
- 비활성화 옵션: 저장하지 않음(disabled)
- 환경변수 기반 오버라이드: `${VAR:default}` 패턴 유지, `.env.{env_name}`에 시크릿 키 자동 생성

## 스키마 변경 (src/settings/config.py)
- OutputTarget: name, enabled, adapter_type, config
- Output: inference, preprocessed
- Config: `output: Output` 필드 추가

의사코드
```python
class OutputTarget(BaseModel):
    name: str
    enabled: bool = True
    adapter_type: Literal["storage", "sql", "bigquery"]
    config: Dict[str, Any] = Field(default_factory=dict)

class Output(BaseModel):
    inference: OutputTarget
    preprocessed: OutputTarget

class Config(BaseModel):
    ...
    output: Output = Field(...)
```

## 템플릿 변경 (src/cli/templates/configs/config.yaml.j2)
- 기존 `data_source` 스타일 복제: `adapter_type` + `config` 블록으로 구성
- inference/preprocessed 각각 `enabled` 제공

예시
```yaml
output:
  inference:
    name: InferenceOutput
    enabled: ${INFER_OUTPUT_ENABLED:true}
    {%- if inference_output_source in ["Local Files", "S3", "GCS"] %}
    adapter_type: storage
    config:
      {%- if inference_output_source == "Local Files" %}
      base_path: "${INFER_OUTPUT_BASE_PATH:./artifacts/predictions}"
      storage_options: {}
      {%- elif inference_output_source == "S3" %}
      base_path: "s3://${INFER_OUTPUT_S3_BUCKET:mmp-out}/${INFER_OUTPUT_S3_PREFIX:{{ env_name }}/preds}"
      storage_options:
        aws_access_key_id: "${AWS_ACCESS_KEY_ID:}"
        aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY:}"
        region_name: "${AWS_REGION:us-east-1}"
      {%- elif inference_output_source == "GCS" %}
      base_path: "gs://${INFER_OUTPUT_GCS_BUCKET:mmp-out}/${INFER_OUTPUT_GCS_PREFIX:{{ env_name }}/preds}"
      storage_options:
        project: "${GCP_PROJECT_ID:{{ gcp_project|default('') }}}"
        token: "${GOOGLE_APPLICATION_CREDENTIALS:}"
      {%- endif %}
  {%- elif inference_output_source == "PostgreSQL" %}
    adapter_type: sql
    config:
      connection_uri: "postgresql://${DB_USER:postgres}:${DB_PASSWORD:postgres}@${DB_HOST:localhost}:${DB_PORT:5432}/${DB_NAME:mmp_db}"
      table: "${INFER_OUTPUT_PG_TABLE:predictions_{{ env_name }}}"
  {%- elif inference_output_source == "BigQuery" %}
    adapter_type: bigquery
    config:
      project_id: "${GCP_PROJECT_ID:{{ gcp_project|default('') }}}"
      dataset_id: "${INFER_OUTPUT_BQ_DATASET:analytics}"
      table: "${INFER_OUTPUT_BQ_TABLE:predictions_{{ env_name }}}"
      location: "${BQ_LOCATION:US}"
  {%- endif %}

  preprocessed:
    name: PreprocessedOutput
    enabled: ${PREPROC_OUTPUT_ENABLED:true}
    # 위와 동일한 분기 (접두사 PREPROC_OUTPUT_ 사용)
```

## 빌더 변경 (src/cli/utils/config_builder.py)
- 각 타겟(inference, preprocessed)에 대해 아래 순서 적용
  1) "저장을 활성화하시겠습니까?" (기본 true)
  2) 활성화된 경우 저장 데이터 소스 선택: [PostgreSQL, BigQuery, Local Files, S3, GCS]
  3) 선택에 따른 필수 필드 입력
     - Local: base_path
     - S3: bucket, prefix (+ AWS 자격증명은 .env)
     - GCS: bucket, prefix (+ GCP 자격증명은 .env)
     - PostgreSQL: table 이름 (접속정보는 .env 재사용)
     - BigQuery: dataset, table (프로젝트/자격증명은 .env 재사용)
- selections → 템플릿 컨텍스트 → `.env.{env_name}.template`에 필요한 키 생성

## 파이프라인 연동
- inference_pipeline.py
  - if not settings.config.output.inference.enabled: 저장 스킵
  - adapter_type 분기:
    - storage: `base_path/preds_{run_id}.parquet` 저장, `mlflow.log_artifact` (로컬 파일만)
    - sql: `table` append 저장
    - bigquery: `dataset.table` append 저장
- trainer.py (전처리 저장)
  - if not settings.config.output.preprocessed.enabled: 저장 스킵
  - storage: `base_path/preprocessed_{split}_{run_id}.parquet` 저장
  - sql/bigquery: `{table}_{split}` append 저장

## 어댑터/팩토리
- StorageAdapter/SQLAdapter 재사용
- BigQueryAdapter 신규 구현 (pandas-gbq/pyarrow-bq)
- Factory.create_data_adapter("bigquery") 경로 보장 및 Registry 등록

## .env 템플릿 키 목록
- 공통 재사용: DB_*, GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS, AWS_*, BQ_LOCATION
- Inference: INFER_OUTPUT_ENABLED, INFER_OUTPUT_BASE_PATH / INFER_OUTPUT_S3_*, INFER_OUTPUT_GCS_*, INFER_OUTPUT_PG_TABLE, INFER_OUTPUT_BQ_DATASET, INFER_OUTPUT_BQ_TABLE
- Preprocessed: PREPROC_OUTPUT_ENABLED, PREPROC_* 동일 패턴

## 테스트 전략
- 스키마 유효성: enabled 플래그/adapter_type/필수 config 키 검증
- 템플릿 렌더링: 선택별 YAML 생성 확인, 환경변수 기본값 반영
- 빌더: 선택 흐름/필수 입력/ENV 키 포함
- 파이프라인: 저장 분기/무저장 플로우, 어댑터 write 호출 여부(Mock)
- BigQueryAdapter: write 호출 포맷/인증 경로 유효성(통합 테스트는 선택)

## 단계별 작업 순서
1) 스키마 추가 (Output, OutputTarget)
2) 템플릿 수정 (output 블록 + enabled)
3) 빌더 확장 (질문/컨텍스트/ENV 템플릿)
4) 파이프라인 반영 (inference, trainer)
5) BigQueryAdapter 구현 및 등록
6) 테스트 추가 (unit/integration)
7) 문서화 업데이트

## 롤백/호환성
- output 미설정 시 기존 동작 유지(배치 추론 파일은 artifact_store 경로로 저장)
- enabled=false 시 저장 동작 없이 메트릭/로그만 기록 