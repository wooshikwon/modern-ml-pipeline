# Modern ML Pipeline: 테스트 실행 가이드

## 📋 테스트 마커 시스템

MMP는 38개의 pytest 마커를 통해 **환경별, 원칙별, 의존성별** 테스트 실행을 지원합니다.

## 🎯 주요 마커 카테고리

### 환경별 마커
```bash
# 로컬 개발 테스트 (빠른 피드백)
uv run pytest -m "local_env"

# DEV 환경 테스트 (완전한 기능 검증)  
uv run pytest -m "dev_env"

# 프로덕션 환경 테스트
uv run pytest -m "prod_env"
```

### Blueprint 원칙별 마커
```bash
# 원칙 1: 설정과 논리의 분리
uv run pytest -m "blueprint_principle_1"

# 원칙 3: 선언적 파이프라인  
uv run pytest -m "blueprint_principle_3"

# 원칙 4: 모듈화와 확장성
uv run pytest -m "blueprint_principle_4"

# 여러 원칙 조합 테스트
uv run pytest -m "blueprint_principle_1 or blueprint_principle_3"
uv run pytest -m "blueprint_principle_3 and blueprint_principle_4"
```

### 인프라 의존성 마커
```bash
# DEV 스택 의존 테스트 (자동 스킵)
uv run pytest -m "requires_dev_stack"

# 개별 서비스 의존 테스트
uv run pytest -m "requires_postgresql"
uv run pytest -m "requires_redis" 
uv run pytest -m "requires_feast"
```

### 테스트 유형별 마커
```bash
# 단위 테스트 (빠른 실행)
uv run pytest -m "unit"

# 통합 테스트 (중간 실행)
uv run pytest -m "integration" 

# End-to-End 테스트 (느린 실행)
uv run pytest -m "e2e"
```

## 🚀 권장 테스트 실행 전략

### 1. 로컬 개발 중
```bash
# 빠른 피드백 (단위 테스트만)
uv run pytest -m "unit and not requires_dev_stack" -x

# Blueprint 원칙 검증
uv run pytest -m "blueprint_principle_1" -v
```

### 2. PR 제출 전 
```bash
# 전체 단위 + 통합 (DEV 스택 없이)
uv run pytest -m "unit or (integration and not requires_dev_stack)" --tb=short

# 코드 품질 검증
uv run ruff check
uv run lint-imports
```

### 3. DEV 스택 포함 검증
```bash
# DEV 환경 스택 시작
./setup-dev-environment.sh start

# DEV 스택 의존 테스트 실행
uv run pytest -m "requires_dev_stack" -v

# 특정 환경 테스트
uv run pytest -m "dev_env" --tb=short
```

### 4. 성능 및 부하 테스트
```bash
# 성능 테스트
uv run pytest -m "performance" --tb=short

# 벤치마크 테스트  
uv run pytest -m "benchmark" --tb=short
```

## 🛠️ 고급 마커 조합

### 복합 조건 테스트
```bash
# 로컬 환경의 Blueprint 원칙 1 테스트
uv run pytest -m "local_env and blueprint_principle_1"

# DEV 스택 없는 통합 테스트
uv run pytest -m "integration and not requires_dev_stack"

# 모든 Blueprint 원칙 테스트  
uv run pytest -m "blueprint_principle_1 or blueprint_principle_2 or blueprint_principle_3"
```

### 환경별 전체 검증
```bash
# 로컬 환경 전체 검증
uv run pytest -m "local_env or (unit and not requires_dev_stack)"

# DEV 환경 전체 검증 (DEV 스택 필요)
uv run pytest -m "dev_env or requires_dev_stack" 
```

## 📊 테스트 수집 및 분석

### 테스트 수집 확인
```bash
# 특정 마커의 테스트 수집 확인
uv run pytest -m "blueprint_principle_1" --collect-only -q

# 마커별 테스트 수 확인
uv run pytest -m "unit" --collect-only -q | wc -l
```

### 테스트 실행 시간 분석
```bash
# 가장 느린 10개 테스트 확인
uv run pytest -m "integration" --durations=10

# 빠른 실행 시간 측정
time uv run pytest -m "unit and not requires_dev_stack"
```

## 🔍 자동 스킵 시스템

### DEV 스택 자동 스킵
`@pytest.mark.requires_dev_stack` 마커가 적용된 테스트는 DEV 스택이 기동되지 않으면 **자동으로 스킵**됩니다.

```bash
# DEV 스택 상태 확인
./setup-dev-environment.sh status

# DEV 스택 시작
./setup-dev-environment.sh start

# 이제 requires_dev_stack 테스트가 실행됨
uv run pytest -m "requires_dev_stack"
```

### 스킵된 테스트 확인
```bash
# 스킵된 테스트 상세 정보
uv run pytest -m "requires_dev_stack" -v -rs
```

## 📈 CI/CD 단계별 실행

### Stage 1: 빠른 검증
```bash
uv run pytest -m "unit and not requires_dev_stack" --tb=short
```

### Stage 2: 통합 검증  
```bash  
uv run pytest -m "integration and not requires_dev_stack" --tb=short
```

### Stage 3: DEV 스택 검증 (선택적)
```bash
uv run pytest -m "requires_dev_stack" --tb=short
```

### Stage 4: End-to-End (수동 트리거)
```bash
uv run pytest -m "e2e" --tb=short
```

## 💡 팁과 모범 사례

1. **개발 중**: `unit` 마커로 빠른 피드백
2. **PR 전**: `blueprint_principle_*` 마커로 설계 검증  
3. **릴리즈 전**: `requires_dev_stack` 포함 전체 검증
4. **성능 이슈**: `performance` + `benchmark` 마커 활용
5. **환경 문제**: 해당 환경 마커로 집중 테스트

이 가이드를 통해 **효율적이고 체계적인** 테스트 실행이 가능합니다.