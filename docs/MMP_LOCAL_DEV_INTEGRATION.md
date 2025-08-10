# 🔗 MMP Local Dev Integration Guide

**Modern ML Pipeline과 mmp-local-dev 완전 통합 가이드**

이 문서는 Modern ML Pipeline(애플리케이션)과 mmp-local-dev(인프라)가 어떻게 독립적으로 운영되면서도 완벽하게 연동되는지에 대한 가이드입니다.

---

## 🏗️ 아키텍처 개요: 완전한 분리와 명확한 계약

### 독립성 원칙
```yaml
modern-ml-pipeline/     # 애플리케이션 (ML 로직, 파이프라인)
├── src/               # ML 파이프라인 코드
├── config/           # 환경별 설정 (연결 정보만)
├── recipes/          # 모델 정의
└── main.py          # CLI 진입점

../mmp-local-dev/      # 인프라 (PostgreSQL, Redis, MLflow)
├── docker-compose.yml # 서비스 정의
├── scripts/          # DB 초기화 & 데이터 시드
├── feast/           # Feature Store 설정
└── setup.sh         # 원클릭 환경 구성
```

### 연동 계약
두 시스템은 `mmp-local-dev/dev-contract.yml`을 통해 공식적으로 연동됩니다:

```yaml
version: "1.0"

provides_env_variables:
  - POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_DB, POSTGRES_PASSWORD
  - REDIS_HOST, REDIS_PORT  
  - MLFLOW_TRACKING_URI

provides_services:
  - postgresql (port: 5432)
  - redis (port: 6379)
  - mlflow (port: 5002)
```

---

## 🚀 환경 설정 및 시작 가이드

### 1. 초기 환경 구성

```bash
# 1. mmp-local-dev 저장소 클론 (한 번만)
cd ~/workspace  # modern-ml-pipeline와 같은 레벨
git clone https://github.com/wooshikwon/mmp-local-dev.git

# 2. 디렉토리 구조 확인
your-workspace/
├── modern-ml-pipeline/    # 이 프로젝트
└── mmp-local-dev/        # 인프라 저장소

# 3. 개발 환경 시작 (modern-ml-pipeline에서 실행)
cd modern-ml-pipeline
./setup-dev-environment.sh start
```

### 2. 개발 환경 관리 명령어

```bash
# 환경 시작/재시작
./setup-dev-environment.sh start

# 현재 상태 확인
./setup-dev-environment.sh status

# 환경 중지
./setup-dev-environment.sh stop

# 완전 삭제 (데이터 포함)
./setup-dev-environment.sh clean

# 통합 테스트 실행
./setup-dev-environment.sh test
```

---

## 🔑 인증 및 연결 정보

### 기본 인증 정보

Modern ML Pipeline이 사용하는 기본 인증 정보는 `config/dev.yaml`에 정의되어 있습니다:

```yaml
# PostgreSQL 연결
Host: localhost
Port: 5432
Database: mlpipeline
Username: mluser
Password: mysecretpassword

# Redis 연결  
Host: localhost
Port: 6379
Password: (없음)

# MLflow 연결
URL: http://localhost:5002
Authentication: (없음)
```

### 인증 정보 변경 방법

#### Option 1: mmp-local-dev에서 변경
```bash
cd ../mmp-local-dev
nano .env
# 예시: 패스워드 변경
POSTGRES_PASSWORD=mynewpassword
./setup.sh --stop && ./setup.sh
```

#### Option 2: modern-ml-pipeline에서 변경
```bash
# config/dev.yaml 수정
nano config/dev.yaml
# connection_uri 직접 변경
connection_uri: "postgresql://mluser:mynewpassword@127.0.0.1:5432/mlpipeline"
```

---

## 🗃️ 데이터 관리 가이드

중요: 데이터 추가는 mmp-local-dev에서만 수행합니다. modern-ml-pipeline은 데이터 소비자입니다.

### 1) 새로운 피처 테이블 추가

```bash
cd ../mmp-local-dev
nano scripts/seed-features.sql
# 테이블 정의 및 샘플 데이터 추가 후 재시작
./setup.sh --clean && ./setup.sh
```

### 2) Feast 피처 정의 업데이트

```python
# feast/features.py
new_feature_source = PostgreSQLSource(
    name="new_feature_source",
    query="SELECT entity_id, new_feature_value, created_at FROM features.new_feature_table",
    timestamp_field="created_at",
)

new_feature_fv = FeatureView(
    name="new_features",
    entities=[entity],
    ttl=timedelta(days=30),
    schema=[Field(name="new_feature_value", dtype=Float32)],
    source=new_feature_source,
)
```

---

## 🔧 고급 설정 및 트러블슈팅

### 포트 충돌 해결

```yaml
# mmp-local-dev/docker-compose.yml
services:
  postgresql:
    ports:
      - "5433:5432"
  redis:
    ports:
      - "6380:6379"
  mlflow:
    ports:
      - "5003:5000"
```

```yaml
# config/dev.yaml 동기화 예시
data_adapters:
  adapters:
    sql:
      config:
        connection_uri: "postgresql://mluser:mysecretpassword@127.0.0.1:5433/mlpipeline"

feature_store:
  feast_config:
    online_store:
      connection_string: "localhost:6380"

mlflow:
  tracking_uri: http://localhost:5003
```

### Docker 리소스 관리

```bash
docker system df
docker system prune
docker-compose logs postgresql
```

### 네트워크 문제 해결

```bash
# 서비스 연결 테스트
# PostgreSQL
docker exec ml-pipeline-postgres pg_isready -U mluser -d mlpipeline
# Redis
docker exec ml-pipeline-redis redis-cli ping
# MLflow
curl -f http://localhost:5002/api/2.0/mlflow/experiments/list
```

---

## 🧪 통합 테스트 및 검증

### 자동 검증

```bash
cd ../mmp-local-dev
python test-integration.py
```

### Modern ML Pipeline E2E 테스트

```bash
# 개발 환경에서 전체 파이프라인 테스트
cd modern-ml-pipeline
APP_ENV=dev uv run python main.py train --recipe-file recipes/local_classification_test.yaml

# 생성된 run-id로 배치 추론 테스트
APP_ENV=dev uv run python main.py batch-inference --run-id <RUN_ID>

# API 서빙 테스트 (Feature Store 구성 + serving.enabled: true 필요)
APP_ENV=dev uv run python main.py serve-api --run-id <RUN_ID>
```

---

## 📊 모니터링 및 관리

```bash
# Docker Compose 서비스 상태
cd ../mmp-local-dev
docker-compose ps

# 리소스 사용량
docker stats ml-pipeline-postgres ml-pipeline-redis ml-pipeline-mlflow

# 로그 실시간 모니터링
docker-compose logs -f
```

---

## 🎯 Best Practices

### 개발 워크플로우
1. 인프라 먼저: mmp-local-dev 정상 동작 확인
2. 계약 검증: dev-contract.yml 준수 여부 테스트
3. 점진적 개발: 작은 변경부터 테스트
4. 로그 확인: 문제 발생시 각 서비스 로그 우선 확인

### 데이터 관리
1. 분리된 관리: 데이터 추가/변경은 mmp-local-dev에서만
2. 스키마 일관성: Feast 정의와 PostgreSQL 스키마 동기화
3. 테스트 데이터: 개인정보 없는 합성 데이터 사용

### 보안
1. 개발 전용 자격 증명은 운영에서 사용 금지
2. Docker 네트워크 격리 사용
3. mmp-local-dev 저장소 정기 업데이트

---

mmp-local-dev는 Modern ML Pipeline의 개발 경험을 극대화하는 독립 인프라 스택입니다. 명확한 계약과 자동화된 설정으로 복잡한 MLOps 인프라를 즉시 사용할 수 있습니다. 