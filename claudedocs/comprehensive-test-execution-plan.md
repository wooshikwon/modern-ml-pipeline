# 포괄적 테스트 실행 및 메트릭 분석 계획

## 개요
현재 ML 파이프라인 프로젝트의 방대한 테스트 코드를 체계적으로 분석하기 위한 포괄적인 실행 계획입니다. 총 16개 그룹으로 세분화하여 점증적 커버리지, 에러율, 스킵율, 실행 속도를 측정합니다.

## 테스트 그룹 분류 (16개 그룹)

### Phase 1: 빠른 단위 테스트 (독립적, 낮은 의존성)

#### Group 1: CLI 핵심 기능
- **경로**: `tests/unit/cli/`
- **파일 수**: 15개
- **포함 테스트**: CLI 명령어, 헬프, 설정 명령, 시스템 체커
- **예상 실행시간**: 30-60초
- **의존성**: 최소

#### Group 2: CLI 유틸리티
- **경로**: `tests/unit/cli/utils/`
- **파일 수**: 5개
- **포함 테스트**: 설정 빌더, 템플릿 엔진, 레시피 빌더
- **예상 실행시간**: 20-40초
- **의존성**: CLI 핵심에 의존

#### Group 3: 설정 및 팩토리
- **경로**: `tests/unit/settings/`, `tests/unit/factory/`
- **파일 수**: 15개 (설정 8개 + 팩토리 8개)
- **포함 테스트**: 설정 로딩, 검증, 컴포넌트 생성, 레지스트리
- **예상 실행시간**: 45-90초
- **의존성**: 낮음

#### Group 4: 기본 어댑터 및 페처
- **경로**: `tests/unit/components/adapters/`, `tests/unit/components/fetchers/`
- **파일 수**: 5개
- **포함 테스트**: SQL 어댑터, 스토리지 어댑터, 피처 스토어 페처
- **예상 실행시간**: 30-60초
- **의존성**: 낮음

#### Group 5: 유틸리티 핵심
- **경로**: `tests/unit/utils/core/`, `tests/unit/utils/data/`
- **파일 수**: 5개
- **포함 테스트**: 로거, 환경 체크, 재현성, 데이터 I/O
- **예상 실행시간**: 25-50초
- **의존성**: 최소

### Phase 2: 중간 단위 테스트 (중간 의존성)

#### Group 6: 전처리 컴포넌트
- **경로**: `tests/unit/components/preprocessor/`
- **파일 수**: 10개
- **포함 테스트**: 스케일러, 인코더, 이산화, 결측치 처리, 피처 생성
- **예상 실행시간**: 60-120초
- **의존성**: 데이터 핸들러에 의존

#### Group 7: 데이터 핸들러
- **경로**: `tests/unit/components/datahandlers/`
- **파일 수**: 3개
- **포함 테스트**: 분류, 회귀, 시계열 데이터 핸들러
- **예상 실행시간**: 45-90초
- **의존성**: 전처리 컴포넌트와 상호작용

#### Group 8: 모델 컴포넌트
- **경로**: `tests/unit/components/models/`, `tests/unit/models/custom/`
- **파일 수**: 7개
- **포함 테스트**: 모델 인터페이스, PyTorch 모델, scikit-learn 모델, 커스텀 모델
- **예상 실행시간**: 90-180초
- **의존성**: GPU 가용성에 따라 변동

#### Group 9: 평가자 및 캘리브레이션
- **경로**: `tests/unit/components/evaluators/`, `tests/unit/components/calibration/`
- **파일 수**: 8개
- **포함 테스트**: 분류/회귀/클러스터링 평가자, 베타 캘리브레이션, 등등분 회귀
- **예상 실행시간**: 60-120초
- **의존성**: 모델 출력에 의존

#### Group 10: 트레이너 및 최적화
- **경로**: `tests/unit/components/trainer/`
- **파일 수**: 1개
- **포함 테스트**: 옵티마이저 컴포넌트
- **예상 실행시간**: 30-60초
- **의존성**: 모델과 데이터에 의존

#### Group 11: 파이프라인
- **경로**: `tests/unit/pipelines/`
- **파일 수**: 3개
- **포함 테스트**: 훈련 파이프라인, 추론 파이프라인, 데이터 분할
- **예상 실행시간**: 90-180초
- **의존성**: 모든 컴포넌트에 의존

#### Group 12: 서빙
- **경로**: `tests/unit/serving/`
- **파일 수**: 4개
- **포함 테스트**: 컨텍스트, 라이프사이클, 라우터, 스키마
- **예상 실행시간**: 60-120초
- **의존성**: 웹 서버 의존성

#### Group 13: 유틸리티 통합
- **경로**: `tests/unit/utils/` (integrations, mlflow, system, template, deps, database)
- **파일 수**: 8개
- **포함 테스트**: MLFlow 통합, Optuna 통합, 시스템 유틸리티, 템플릿
- **예상 실행시간**: 90-180초
- **의존성**: 외부 서비스 연결

### Phase 3: 통합 및 E2E 테스트 (높은 의존성)

#### Group 14: 데이터베이스 및 MLFlow 통합
- **경로**: `tests/integration/test_database_integration.py`, `tests/integration/test_mlflow_integration.py`, `tests/integration/test_settings_integration.py`
- **파일 수**: 3개
- **포함 테스트**: DB 연결, MLFlow 트래킹, 설정 통합
- **예상 실행시간**: 120-300초
- **의존성**: 외부 서비스 필요

#### Group 15: 컴포넌트 상호작용 및 오류 전파
- **경로**: `tests/integration/test_component_interactions.py`, `tests/integration/test_error_propagation.py`, `tests/integration/test_integration_completeness.py`, `tests/integration/test_production_readiness.py`
- **파일 수**: 4개
- **포함 테스트**: 컴포넌트 간 상호작용, 에러 처리, 프로덕션 준비성
- **예상 실행시간**: 180-400초
- **의존성**: 전체 시스템

#### Group 16: 파이프라인 통합 및 서빙
- **경로**: `tests/integration/test_*pipeline*.py`, `tests/integration/test_serving*.py`, `tests/e2e/`
- **파일 수**: 10개 (통합 9개 + E2E 1개)
- **포함 테스트**: 파이프라인 오케스트레이션, 서빙 엔드포인트, E2E 워크플로우
- **예상 실행시간**: 300-600초
- **의존성**: 전체 시스템 + 외부 서비스

## 메트릭 수집 전략

### 수집 메트릭
1. **테스트 커버리지**: 점증적 코드 커버리지 측정
2. **테스트 에러율**: (실패 테스트 수 / 전체 테스트 수) × 100
3. **테스트 스킵율**: (스킵된 테스트 수 / 전체 테스트 수) × 100
4. **테스트 실행 속도**: 각 그룹별 실행 시간 (초)

### 로깅 전략
- **상세 로그**: 각 테스트 파일별 실행 결과
- **에러 분류**: 실패 원인별 분류 (import 에러, assertion 에러, 타임아웃 등)
- **진행 상황**: 실시간 진행률 표시
- **의존성 추적**: 각 그룹이 이전 그룹에 미치는 영향

## 실행 순서 및 전략

### 단계별 실행
1. **Phase 1** (Groups 1-5): 병렬 실행 가능, 빠른 피드백
2. **Phase 2** (Groups 6-13): 순차 실행, 의존성 고려
3. **Phase 3** (Groups 14-16): 격리된 환경에서 실행

### 실패 처리 전략
- **조기 중단**: 각 페이즈에서 50% 이상 실패 시 중단
- **재시도 로직**: 네트워크 관련 실패 시 3회 재시도
- **격리**: 실패한 그룹이 다른 그룹에 영향을 주지 않도록 격리

## 예상 총 실행 시간
- **Phase 1**: 2-5분 (병렬 실행)
- **Phase 2**: 15-25분 (순차 실행)
- **Phase 3**: 10-20분 (격리 실행)
- **총합**: 27-50분 (환경에 따라 변동)

## 출력 형식

### 실시간 로그
```
[2024-XX-XX 10:00:00] Starting Group 1: CLI 핵심 기능
[2024-XX-XX 10:00:05] Group 1 Progress: 5/15 files completed
[2024-XX-XX 10:00:30] Group 1 Complete: 15 passed, 0 failed, 1 skipped (30s)
[2024-XX-XX 10:00:30] Coverage Update: 12.3% (+2.1%)
```

### 최종 리포트
```json
{
  "execution_summary": {
    "total_duration": "28m 45s",
    "total_tests": 1247,
    "passed": 1189,
    "failed": 23,
    "skipped": 35
  },
  "coverage_progression": [
    {"group": "Group 1", "coverage": 12.3, "delta": 12.3},
    {"group": "Group 2", "coverage": 18.7, "delta": 6.4}
  ],
  "performance_metrics": {
    "fastest_group": "Group 5",
    "slowest_group": "Group 16", 
    "average_test_time": "1.38s"
  }
}
```

## Phase별 실행 명령어

### Phase 1: 빠른 단위 테스트 (병렬 실행 가능)

#### Group 1: CLI 핵심 기능
```bash
# 단독 실행 (상세 로그)
pytest tests/unit/cli/ -v --tb=short --maxfail=10 --cov=src.cli --cov-report=term-missing --cov-report=html:htmlcov/group1

# 병렬 실행 (빠른 피드백)
pytest tests/unit/cli/ -n auto --tb=line --maxfail=5 --cov=src.cli --cov-report=term
```

#### Group 2: CLI 유틸리티
```bash
# 단독 실행
pytest tests/unit/cli/utils/ -v --tb=short --cov=src.cli.utils --cov-report=term-missing --cov-report=html:htmlcov/group2

# 병렬 실행
pytest tests/unit/cli/utils/ -n auto --tb=line --cov=src.cli.utils --cov-report=term
```

#### Group 3: 설정 및 팩토리
```bash
# 단독 실행 (두 디렉토리 동시)
pytest tests/unit/settings/ tests/unit/factory/ -v --tb=short --cov=src.settings --cov=src.factory --cov-report=term-missing --cov-report=html:htmlcov/group3

# 병렬 실행
pytest tests/unit/settings/ tests/unit/factory/ -n auto --tb=line --cov=src.settings --cov=src.factory --cov-report=term
```

#### Group 4: 기본 어댑터 및 페처
```bash
# 단독 실행
pytest tests/unit/components/adapters/ tests/unit/components/fetchers/ -v --tb=short --cov=src.components.adapters --cov=src.components.fetchers --cov-report=term-missing --cov-report=html:htmlcov/group4

# 병렬 실행
pytest tests/unit/components/adapters/ tests/unit/components/fetchers/ -n auto --tb=line --cov=src.components.adapters --cov=src.components.fetchers --cov-report=term
```

#### Group 5: 유틸리티 핵심
```bash
# 단독 실행
pytest tests/unit/utils/core/ tests/unit/utils/data/ -v --tb=short --cov=src.utils.core --cov=src.utils.data --cov-report=term-missing --cov-report=html:htmlcov/group5

# 병렬 실행
pytest tests/unit/utils/core/ tests/unit/utils/data/ -n auto --tb=line --cov=src.utils.core --cov=src.utils.data --cov-report=term
```

### Phase 2: 중간 단위 테스트 (순차 실행 권장)

#### Group 6: 전처리 컴포넌트
```bash
# 단독 실행 (메모리 관리)
pytest tests/unit/components/preprocessor/ -v --tb=short --maxfail=15 --cov=src.components.preprocessor --cov-report=term-missing --cov-report=html:htmlcov/group6

# 선택적 병렬 실행 (메모리 충분시)
pytest tests/unit/components/preprocessor/ -n 2 --tb=line --maxfail=10 --cov=src.components.preprocessor --cov-report=term
```

#### Group 7: 데이터 핸들러
```bash
# 단독 실행
pytest tests/unit/components/datahandlers/ -v --tb=short --cov=src.components.datahandlers --cov-report=term-missing --cov-report=html:htmlcov/group7

# 병렬 실행
pytest tests/unit/components/datahandlers/ -n 2 --tb=line --cov=src.components.datahandlers --cov-report=term
```

#### Group 8: 모델 컴포넌트
```bash
# 단독 실행 (GPU 고려)
pytest tests/unit/components/models/ tests/unit/models/custom/ -v --tb=short --maxfail=20 --timeout=300 --cov=src.components.models --cov=src.models --cov-report=term-missing --cov-report=html:htmlcov/group8

# 순차 실행 (GPU 경합 방지)
pytest tests/unit/components/models/ tests/unit/models/custom/ -n 1 --tb=line --timeout=180 --cov=src.components.models --cov=src.models --cov-report=term
```

#### Group 9: 평가자 및 캘리브레이션
```bash
# 단독 실행
pytest tests/unit/components/evaluators/ tests/unit/components/calibration/ -v --tb=short --cov=src.components.evaluators --cov=src.components.calibration --cov-report=term-missing --cov-report=html:htmlcov/group9

# 병렬 실행
pytest tests/unit/components/evaluators/ tests/unit/components/calibration/ -n 2 --tb=line --cov=src.components.evaluators --cov=src.components.calibration --cov-report=term
```

#### Group 10: 트레이너 및 최적화
```bash
# 단독 실행
pytest tests/unit/components/trainer/ -v --tb=short --timeout=180 --cov=src.components.trainer --cov-report=term-missing --cov-report=html:htmlcov/group10

# 병렬 실행 불필요 (파일 수 적음)
pytest tests/unit/components/trainer/ --tb=line --timeout=120 --cov=src.components.trainer --cov-report=term
```

#### Group 11: 파이프라인
```bash
# 단독 실행 (높은 메모리 사용)
pytest tests/unit/pipelines/ -v --tb=short --maxfail=10 --timeout=300 --cov=src.pipelines --cov-report=term-missing --cov-report=html:htmlcov/group11

# 순차 실행 권장
pytest tests/unit/pipelines/ -n 1 --tb=line --timeout=240 --cov=src.pipelines --cov-report=term
```

#### Group 12: 서빙
```bash
# 단독 실행
pytest tests/unit/serving/ -v --tb=short --cov=src.serving --cov-report=term-missing --cov-report=html:htmlcov/group12

# 병렬 실행
pytest tests/unit/serving/ -n 2 --tb=line --cov=src.serving --cov-report=term
```

#### Group 13: 유틸리티 통합
```bash
# 단독 실행 (외부 의존성 고려)
pytest tests/unit/utils/integrations/ tests/unit/utils/mlflow/ tests/unit/utils/system/ tests/unit/utils/template/ tests/unit/utils/deps/ tests/unit/utils/database/ -v --tb=short --maxfail=15 --timeout=240 --cov=src.utils --cov-report=term-missing --cov-report=html:htmlcov/group13

# 순차 실행 권장 (외부 서비스 경합 방지)
pytest tests/unit/utils/integrations/ tests/unit/utils/mlflow/ tests/unit/utils/system/ tests/unit/utils/template/ tests/unit/utils/deps/ tests/unit/utils/database/ -n 1 --tb=line --timeout=180 --cov=src.utils --cov-report=term
```

### Phase 3: 통합 및 E2E 테스트 (격리 실행)

#### Group 14: 데이터베이스 및 MLFlow 통합
```bash
# 단독 실행 (외부 서비스 필요)
pytest tests/integration/test_database_integration.py tests/integration/test_mlflow_integration.py tests/integration/test_settings_integration.py -v --tb=long --maxfail=5 --timeout=600 --cov=src --cov-report=term-missing --cov-report=html:htmlcov/group14

# 격리 실행 (외부 의존성 고려)
pytest tests/integration/test_database_integration.py tests/integration/test_mlflow_integration.py tests/integration/test_settings_integration.py -x --tb=short --timeout=300 --cov=src --cov-report=term
```

#### Group 15: 컴포넌트 상호작용 및 오류 전파
```bash
# 단독 실행 (전체 시스템 테스트)
pytest tests/integration/test_component_interactions.py tests/integration/test_error_propagation.py tests/integration/test_integration_completeness.py tests/integration/test_production_readiness.py -v --tb=long --maxfail=3 --timeout=800 --cov=src --cov-report=term-missing --cov-report=html:htmlcov/group15

# 격리 실행
pytest tests/integration/test_component_interactions.py tests/integration/test_error_propagation.py tests/integration/test_integration_completeness.py tests/integration/test_production_readiness.py -x --tb=short --timeout=400 --cov=src --cov-report=term
```

#### Group 16: 파이프라인 통합 및 서빙
```bash
# 단독 실행 (모든 통합 테스트)
pytest tests/integration/test_cli_pipeline_integration.py tests/integration/test_inference_pipeline_integration.py tests/integration/test_pipeline_orchestration.py tests/integration/test_preprocessor_pipeline_integration.py tests/integration/test_serving*.py tests/e2e/ -v --tb=long --maxfail=2 --timeout=1200 --cov=src --cov-report=term-missing --cov-report=html:htmlcov/group16

# 격리 실행 (최종 검증)
pytest tests/integration/test_cli_pipeline_integration.py tests/integration/test_inference_pipeline_integration.py tests/integration/test_pipeline_orchestration.py tests/integration/test_preprocessor_pipeline_integration.py tests/integration/test_serving*.py tests/e2e/ -x --tb=short --timeout=600 --cov=src --cov-report=term
```

## 통합 실행 명령어

### 전체 Phase 순차 실행
```bash
# Phase 1 (병렬)
pytest tests/unit/cli/ tests/unit/settings/ tests/unit/factory/ tests/unit/components/adapters/ tests/unit/components/fetchers/ tests/unit/utils/core/ tests/unit/utils/data/ -n auto --tb=line --cov=src --cov-report=term

# Phase 2 (순차)
pytest tests/unit/components/preprocessor/ tests/unit/components/datahandlers/ tests/unit/components/models/ tests/unit/models/custom/ tests/unit/components/evaluators/ tests/unit/components/calibration/ tests/unit/components/trainer/ tests/unit/pipelines/ tests/unit/serving/ tests/unit/utils/integrations/ tests/unit/utils/mlflow/ tests/unit/utils/system/ tests/unit/utils/template/ tests/unit/utils/deps/ tests/unit/utils/database/ -n 1 --tb=line --timeout=300 --cov=src --cov-report=term

# Phase 3 (격리)
pytest tests/integration/ tests/e2e/ -x --tb=short --timeout=600 --cov=src --cov-report=html --cov-report=term-missing
```

### 커버리지 통합 리포트 생성
```bash
# 모든 그룹 실행 후 통합 커버리지
coverage combine
coverage html -d htmlcov/final_report
coverage report --show-missing --fail-under=15
```

## 실행 환경별 최적화

### M1 MacBook (16GB 메모리) 최적화
- **병렬 처리**: `-n 4` 또는 `-n auto` (CPU 코어 활용)
- **메모리 관리**: Phase 2에서는 `-n 2` 제한
- **타임아웃**: GPU 작업은 300초, 일반은 120초

### CI/CD 환경 최적화
- **빠른 피드백**: `--tb=line --maxfail=5`
- **병렬 제한**: `-n 2` (리소스 경합 방지)
- **타임아웃 단축**: 일반 60초, 통합 180초

### 로컬 개발 최적화
- **상세 출력**: `-v --tb=short`
- **커버리지 HTML**: `--cov-report=html` 활용
- **선택적 실행**: `-k` 옵션으로 특정 테스트 필터링

## 도구 및 기술 스택
- **실행 엔진**: pytest with coverage
- **병렬 처리**: pytest-xdist
- **메트릭 수집**: custom Python 스크립트
- **로깅**: Python logging with structured output
- **리포팅**: JSON + Markdown 형식

이 계획을 통해 전체 테스트 스위트를 체계적으로 분석하고, 테스트 품질과 성능을 종합적으로 평가할 수 있습니다.