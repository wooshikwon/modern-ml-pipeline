# 🔗 MMP Local Dev Integration Guide

**Modern ML Pipeline과 mmp-local-dev 완전 통합 가이드**

이 문서는 Modern ML Pipeline(애플리케이션)과 mmp-local-dev(인프라)가 어떻게 독립적으로 운영되면서도 완벽하게 연동되는지에 대한 완전한 가이드입니다.

---

## 🏗️ **아키텍처 개요: 완전한 분리와 명확한 계약**

### **독립성 원칙**
```yaml
modern-ml-pipeline/     # 🎯 애플리케이션 (ML 로직, 파이프라인)
├── src/               # ML 파이프라인 코드
├── config/           # 환경별 설정 (연결 정보만)
├── recipes/          # 모델 정의
└── main.py          # CLI 진입점

../mmp-local-dev/      # 🏗️ 인프라 (PostgreSQL, Redis, MLflow)
├── docker-compose.yml # 서비스 정의
├── scripts/          # DB 초기화 & 데이터 시드
├── feast/           # Feature Store 설정
└── setup.sh         # 원클릭 환경 구성
```

### **연동 계약**
두 시스템은 `mmp-local-dev/dev-contract.yml`을 통해 공식적으로 연동됩니다:

```yaml
# dev-contract.yml - 공식 연동 계약서
version: \"1.0\"

provides_env_variables:    # mmp-local-dev가 제공하는 환경변수
  - POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_DB, POSTGRES_PASSWORD
  - REDIS_HOST, REDIS_PORT  
  - MLFLOW_TRACKING_URI

provides_services:         # mmp-local-dev가 제공하는 서비스
  - postgresql (port: 5432)
  - redis (port: 6379)
  - mlflow (port: 5002)
```

---

## 🚀 **환경 설정 및 시작 가이드**

### **1. 초기 환경 구성**

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

### **2. 개발 환경 관리 명령어**

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

## 🔑 **인증 및 연결 정보**

### **기본 인증 정보**

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
Password: (없음 - 인증 비활성화)

# MLflow 연결
URL: http://localhost:5002
Authentication: (없음 - 로컬 개발용)
```

### **인증 정보 변경 방법**

#### **Option 1: mmp-local-dev에서 변경**
```bash
# 1. mmp-local-dev 디렉토리로 이동
cd ../mmp-local-dev

# 2. .env 파일 수정
nano .env

# 예시: 패스워드 변경
POSTGRES_PASSWORD=mynewpassword

# 3. 환경 재시작
./setup.sh --stop && ./setup.sh
```

#### **Option 2: modern-ml-pipeline에서 변경**
```bash
# config/dev.yaml 수정
nano config/dev.yaml

# connection_uri 직접 변경
connection_uri: \"postgresql://mluser:mynewpassword@127.0.0.1:5432/mlpipeline\"
```

---

## 🗃️ **데이터 관리 가이드**

### **Feature Store 데이터 추가**

**중요**: 데이터 추가는 **mmp-local-dev에서만** 수행합니다. modern-ml-pipeline은 데이터 소비자 역할만 합니다.

#### **1. 새로운 피처 테이블 추가**

```bash
# 1. mmp-local-dev로 이동
cd ../mmp-local-dev

# 2. SQL 스크립트 수정
nano scripts/seed-features.sql

# 3. 새 테이블 정의 추가
CREATE TABLE IF NOT EXISTS new_feature_table (
    entity_id VARCHAR(50) PRIMARY KEY,
    new_feature_value DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

# 4. 샘플 데이터 삽입
INSERT INTO new_feature_table (entity_id, new_feature_value) VALUES
('entity_001', 123.45),
('entity_002', 678.90);
```

#### **2. Feast 피처 정의 업데이트**

```bash
# feast/features.py 수정
nano feast/features.py

# 새 피처 뷰 추가
new_feature_source = PostgreSQLSource(
    name=\"new_feature_source\",
    query=\"SELECT entity_id, new_feature_value, created_at FROM features.new_feature_table\",
    timestamp_field=\"created_at\",
)

new_feature_fv = FeatureView(
    name=\"new_features\",
    entities=[entity],
    ttl=timedelta(days=30),
    schema=[Field(name=\"new_feature_value\", dtype=Float32)],
    source=new_feature_source,
)
```

#### **3. 변경사항 적용**

```bash
# 1. 환경 재시작 (데이터 다시 로드)
./setup.sh --clean && ./setup.sh

# 2. 통합 테스트로 확인
python test-integration.py

# 3. modern-ml-pipeline에서 사용
cd ../modern-ml-pipeline
# recipes/*.yaml에서 새 피처 사용 가능
```

### **대량 데이터 추가**

```bash
# 1. CSV 파일 준비
# data.csv:
# entity_id,feature_value
# entity_001,123.45

# 2. PostgreSQL에 직접 로드
cd ../mmp-local-dev
docker exec -i ml-pipeline-postgres psql -U mluser -d mlpipeline << EOF
\\copy features.new_feature_table(entity_id,new_feature_value) FROM '/tmp/data.csv' DELIMITER ',' CSV HEADER;
EOF
```

---

## 🔧 **고급 설정 및 트러블슈팅**

### **포트 충돌 해결**

```yaml
# mmp-local-dev/docker-compose.yml 수정
services:
  postgresql:
    ports:
      - \"5433:5432\"  # 포트 변경
  
  redis:
    ports:
      - \"6380:6379\"  # 포트 변경
      
  mlflow:
    ports:
      - \"5003:5000\"  # 포트 변경
```

```yaml
# config/dev.yaml 동기화
data_adapters:
  adapters:
    sql:
      config:
        connection_uri: \"postgresql://mluser:mysecretpassword@127.0.0.1:5433/mlpipeline\"

feature_store:
  feast_config:
    online_store:
      connection_string: \"localhost:6380\"

mlflow:
  tracking_uri: http://localhost:5003
```

### **Docker 리소스 관리**

```bash
# 현재 사용 중인 리소스 확인
docker system df

# 사용하지 않는 리소스 정리
docker system prune

# 특정 서비스 로그 확인
docker-compose logs postgresql
docker-compose logs redis
docker-compose logs mlflow
```

### **네트워크 문제 해결**

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

## 🧪 **통합 테스트 및 검증**

### **자동 검증**

```bash
# 전체 계약 준수 테스트
cd ../mmp-local-dev
python test-integration.py

# 출력 예시:
# [SUCCESS] PostgreSQL 연결 성공
# [SUCCESS] Redis 연결 성공  
# [SUCCESS] MLflow 서버 응답 확인
# [SUCCESS] Feast 피처 조회 성공
# [SUCCESS] 모든 계약 조건 준수 확인
```

### **수동 검증**

```bash
# PostgreSQL 데이터 확인
docker exec -it ml-pipeline-postgres psql -U mluser -d mlpipeline -c \"\\dt features.*\"

# Redis 키 확인
docker exec -it ml-pipeline-redis redis-cli keys \"*\"

# MLflow 실험 확인
curl -s http://localhost:5002/api/2.0/mlflow/experiments/list | jq '.experiments[].name'
```

### **Modern ML Pipeline E2E 테스트**

```bash
# 개발 환경에서 전체 파이프라인 테스트
cd modern-ml-pipeline
APP_ENV=dev uv run python main.py train --recipe-file recipes/local_classification_test

# 생성된 run-id로 배치 추론 테스트
APP_ENV=dev uv run python main.py batch-inference --run-id <생성된-run-id>

# API 서빙 테스트
APP_ENV=dev uv run python main.py serve-api --run-id <생성된-run-id>
```

---

## 📊 **모니터링 및 관리**

### **서비스 상태 모니터링**

```bash
# Docker Compose 서비스 상태
cd ../mmp-local-dev
docker-compose ps

# 리소스 사용량
docker stats ml-pipeline-postgres ml-pipeline-redis ml-pipeline-mlflow

# 로그 실시간 모니터링
docker-compose logs -f
```

### **데이터베이스 관리**

```bash
# 백업
docker exec ml-pipeline-postgres pg_dump -U mluser mlpipeline > backup.sql

# 복원
docker exec -i ml-pipeline-postgres psql -U mluser mlpipeline < backup.sql

# 피처 스키마 확인
docker exec -it ml-pipeline-postgres psql -U mluser -d mlpipeline -c \"\\dt features.*\"
```

---

## 🎯 **Best Practices**

### **개발 워크플로우**

1. **인프라 먼저**: mmp-local-dev 환경이 정상 동작하는지 확인
2. **계약 검증**: dev-contract.yml 준수 여부 테스트
3. **점진적 개발**: 작은 변경사항부터 테스트
4. **로그 확인**: 문제 발생 시 각 서비스 로그 우선 확인

### **데이터 관리**

1. **분리된 관리**: 데이터 추가/변경은 mmp-local-dev에서만
2. **스키마 일관성**: Feast 정의와 PostgreSQL 스키마 동기화 유지
3. **테스트 데이터**: 실제와 유사하지만 개인정보 없는 합성 데이터 사용

### **보안**

1. **로컬 전용**: 개발 환경 인증 정보는 절대 운영에서 사용 금지
2. **격리된 네트워크**: Docker 네트워크를 통한 서비스 간 통신
3. **정기적 업데이트**: mmp-local-dev 저장소 정기적 동기화

---

**🌟 결론: mmp-local-dev는 Modern ML Pipeline의 개발 경험을 극대화하는 독립적인 인프라 스택입니다. 명확한 계약과 자동화된 설정을 통해 복잡한 MLOps 인프라를 개발자가 신경 쓰지 않고도 즉시 사용할 수 있게 해줍니다.** 