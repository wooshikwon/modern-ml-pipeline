# 🏗️ Infrastructure Stacks Guide

**Modern ML Pipeline 인프라 구성 가이드 - 당신의 환경에 맞는 최적 조합 찾기**

이 문서는 Modern ML Pipeline을 다양한 인프라 환경에서 구성하는 방법을 안내합니다. 로컬 개발부터 엔터프라이즈 클라우드까지, 당신이 현재 사용 중인 기술 스택에 맞춰 파이프라인을 구성할 수 있습니다.

---

## 🎯 **지원하는 인프라 구성요소**

Modern ML Pipeline은 3가지 핵심 구성요소로 나뉩니다:

### **1. 데이터 레이어**
- **SQL 데이터베이스**: PostgreSQL, BigQuery, Snowflake, MySQL, SQLite
- **파일 스토리지**: 로컬 파일, Google Cloud Storage, Amazon S3, Azure Blob Storage
- **Feature Store**: Feast 기반 (Redis, DynamoDB, PostgreSQL, Bigtable 등)

### **2. ML 플랫폼**
- **실험 추적**: MLflow (로컬/서버/클라우드)
- **모델 저장소**: 파일 시스템, GCS, S3, Azure Blob

### **3. 서빙 플랫폼**
- **API 서버**: FastAPI (로컬/Docker/Kubernetes/서버리스)
- **배치 처리**: 로컬 Python, 클라우드 작업, 컨테이너

---

## 🏠 **환경별 구성 가이드**

### **LOCAL 환경: 즉시 시작**

**추천 대상**: 개인 개발자, 프로토타이핑, 학습 목적

```yaml
# 필요한 것: 아무것도 없음 (git clone만)

데이터 레이어:
  데이터 소스: 로컬 파일 (Parquet, CSV)
  Feature Store: 비활성화 (자동)
  
ML 플랫폼:
  MLflow: 로컬 디렉토리 (./mlruns)
  모델 저장: 로컬 파일 시스템
  
서빙:
  배치 추론: ✅ 지원
  API 서빙: ❌ 의도적 비활성화 (단순성 유지)
```

**설정 방법:**
```bash
# 1. 클론 후 즉시 실행
git clone https://github.com/wooshikwon/modern-ml-pipeline.git
cd modern-ml-pipeline
uv venv && uv sync

# 2. 데이터 파일 준비
# data/ 디렉토리에 .parquet 또는 .csv 파일 배치

# 3. 바로 학습 시작
uv run python main.py train --recipe-file recipes/local_classification_test.yaml
```

### **DEV 환경: 완전한 기능**

**추천 대상**: 팀 개발, 모든 기능 테스트, Feature Store 활용

```yaml
# 필요한 것: Docker, Docker Compose

데이터 레이어:
  SQL DB: PostgreSQL (Docker)
  Feature Store: PostgreSQL + Redis (Docker)
  
ML 플랫폼:
  MLflow: HTTP 서버 (Docker)
  모델 저장: PostgreSQL 백엔드
  
서빙:
  배치 추론: ✅ 완전 지원
  API 서빙: ✅ 완전 지원
```

**설정 방법:**
```bash
# 1. mmp-local-dev 인프라 설정
git clone https://github.com/wooshikwon/mmp-local-dev.git ../mmp-local-dev
cd ../mmp-local-dev
docker-compose up -d

# 2. 연결 확인
curl http://localhost:5002/health  # MLflow
redis-cli ping                     # Redis
psql -h localhost -p 5432 -U mlpipeline_user -d mlpipeline_db -c "SELECT 1;"

# 3. DEV 환경에서 실행
cd modern-ml-pipeline
APP_ENV=dev uv run python main.py train --recipe-file recipes/dev_classification_test.yaml
```

### **PROD 환경: 클라우드 확장**

**추천 대상**: 운영 서비스, 대용량 데이터, 고가용성 필요

#### **Google Cloud Platform 구성**

```yaml
데이터 레이어:
  SQL DB: BigQuery
  Feature Store: BigQuery + Redis Labs
  파일 저장: Google Cloud Storage
  
ML 플랫폼:
  MLflow: Cloud Run + GCS
  모델 저장: GCS
  
서빙:
  API 서버: Cloud Run
  배치 처리: Cloud Run Jobs
```

**설정 파일 예시:**
```yaml
# config/prod.yaml
data_adapters:
  adapters:
    sql:
      class_name: SqlAdapter
      config:
        connection_uri: "bigquery://your-project-id/your-dataset"
    storage:
      class_name: StorageAdapter
      config: {}

mlflow:
  tracking_uri: "https://your-mlflow-server.run.app"

feature_store:
  feast_config:
    offline_store:
      type: "bigquery"
      project_id: "your-project-id"
      dataset: "feast_offline"
    online_store:
      type: "redis"
      connection_string: "redis://your-redis-endpoint:6379"
```

#### **Amazon Web Services 구성**

```yaml
데이터 레이어:
  SQL DB: Snowflake 또는 Redshift
  Feature Store: Snowflake + DynamoDB
  파일 저장: Amazon S3
  
ML 플랫폼:
  MLflow: ECS + S3
  모델 저장: S3
  
서빙:
  API 서버: Lambda 또는 ECS
  배치 처리: ECS Tasks
```

**설정 파일 예시:**
```yaml
# config/prod_aws.yaml
data_adapters:
  adapters:
    sql:
      class_name: SqlAdapter
      config:
        connection_uri: "snowflake://user:pass@account/database/schema"
    storage:
      class_name: StorageAdapter
      config: {}

feature_store:
  feast_config:
    offline_store:
      type: "snowflake"
      account: "your-account"
      database: "feast_db"
    online_store:
      type: "dynamodb"
      region: "us-east-1"
```

#### **Microsoft Azure 구성**

```yaml
데이터 레이어:
  SQL DB: Synapse Analytics
  Feature Store: Synapse + Cosmos DB
  파일 저장: Azure Blob Storage
  
ML 플랫폼:
  MLflow: Container Instances + Blob
  모델 저장: Blob Storage
  
서빙:
  API 서버: Container Instances
  배치 처리: Container Instances
```

---

## 🔧 **인프라별 설정 가이드**

### **데이터베이스 설정**

#### **PostgreSQL**
```yaml
# config/your_env.yaml
data_adapters:
  adapters:
    sql:
      class_name: SqlAdapter
      config:
        connection_uri: "postgresql://user:password@host:5432/database"
```

#### **BigQuery**
```yaml
data_adapters:
  adapters:
    sql:
      class_name: SqlAdapter
      config:
        connection_uri: "bigquery://project-id/dataset-id"
```

#### **Snowflake**
```yaml
data_adapters:
  adapters:
    sql:
      class_name: SqlAdapter
      config:
        connection_uri: "snowflake://user:password@account/database/schema"
```

### **파일 스토리지 설정**

#### **로컬 파일 시스템**
```yaml
# Recipe에서 직접 경로 지정
model:
  loader:
    source_uri: "data/my_dataset.parquet"
```

#### **Google Cloud Storage**
```yaml
model:
  loader:
    source_uri: "gs://your-bucket/path/to/data.parquet"
```

#### **Amazon S3**
```yaml
model:
  loader:
    source_uri: "s3://your-bucket/path/to/data.parquet"
```

#### **Azure Blob Storage**
```yaml
model:
  loader:
    source_uri: "abfs://container@account.dfs.core.windows.net/path/to/data.parquet"
```

### **Feature Store 설정**

#### **Redis (단일 인스턴스)**
```yaml
feature_store:
  feast_config:
    online_store:
      type: "redis"
      connection_string: "redis://localhost:6379"
```

#### **Redis (클러스터)**
```yaml
feature_store:
  feast_config:
    online_store:
      type: "redis"
      redis_type: "redis_cluster"
      connection_string: "redis://redis-cluster-endpoint:6379"
```

#### **DynamoDB**
```yaml
feature_store:
  feast_config:
    online_store:
      type: "dynamodb"
      region: "us-west-2"
      table_name: "feast_online_store"
```

#### **Bigtable**
```yaml
feature_store:
  feast_config:
    online_store:
      type: "bigtable"
      project_id: "your-gcp-project"
      instance_id: "feast-instance"
```

---

## 🎯 **상황별 추천 구성**

### **스타트업 / 개인 프로젝트**

**시나리오**: 비용 최소화, 빠른 프로토타이핑

```yaml
추천 스택:
  개발: LOCAL 환경
  테스트: DEV 환경 (mmp-local-dev)
  운영: GCP Cloud Run + PostgreSQL

월 예상 비용: $0 (개발) + $50-100 (운영)
```

### **중소기업**

**시나리오**: 안정성과 비용 균형, 팀 협업

```yaml
추천 스택:
  개발: DEV 환경 (공유)
  스테이징: DEV 환경 (별도 인스턴스)
  운영: GCP 관리형 서비스

월 예상 비용: $100-500
```

### **대기업 / 엔터프라이즈**

**시나리오**: 거버넌스, 보안, 확장성

```yaml
추천 스택:
  개발: DEV 환경 (개인별)
  스테이징: 클라우드 (운영과 동일)
  운영: 멀티클라우드 또는 하이브리드

월 예상 비용: $1,000+
```

### **데이터 집약적 서비스**

**시나리오**: 페타바이트급 데이터, 실시간 처리

```yaml
추천 스택:
  SQL: BigQuery 또는 Snowflake
  Feature Store: Redis Labs (클러스터)
  API: 글로벌 엣지 배포

월 예상 비용: 사용량 기반
```

---

## 🔄 **환경 전환 가이드**

### **로컬 → 클라우드 전환**

```bash
# 1. 클라우드 인증 설정
gcloud auth application-default login  # GCP
aws configure                          # AWS
az login                              # Azure

# 2. 설정 파일 변경
# config/prod.yaml에서 연결 정보 수정

# 3. 동일한 Recipe로 실행
APP_ENV=prod uv run python main.py train --recipe-file recipes/my_model.yaml
```

### **클라우드 간 전환**

```bash
# GCP → AWS 전환 예시
# 1. Snowflake 설정 (멀티클라우드 DB)
# 2. config/prod_aws.yaml 생성
# 3. 환경 변수만 변경
APP_ENV=prod_aws uv run python main.py train --recipe-file recipes/my_model.yaml
```

---

## 🛠️ **설정 검증 가이드**

### **연결 테스트**

```bash
# 데이터베이스 연결 확인
uv run python -c "
from src.settings import load_config_files
from src.engine.factory import Factory
settings = load_config_files()
factory = Factory(settings)
adapter = factory.create_data_adapter('sql')
print('DB 연결 성공!')
"

# Feature Store 연결 확인  
uv run python -c "
from src.settings import load_config_files
from src.engine.factory import Factory
settings = load_config_files()
factory = Factory(settings)
adapter = factory.create_feature_store_adapter()
print('Feature Store 연결 성공!')
"
```

### **설정 파일 검증**

```bash
# Recipe 파일 검증
uv run python main.py validate --recipe-file recipes/my_model.yaml

# 전체 인프라 계약 테스트
uv run python main.py test-contract
```

---

## 📊 **성능 가이드라인**

### **데이터 크기별 권장 구성**

| 데이터 크기 | 추천 SQL DB | 추천 Feature Store | 예상 성능 |
|------------|-------------|------------------|-----------|
| < 1GB | PostgreSQL | Redis (단일) | 1K rows/sec |
| 1GB - 100GB | PostgreSQL/BigQuery | Redis (단일) | 10K rows/sec |
| 100GB - 10TB | BigQuery/Snowflake | Redis (클러스터) | 100K rows/sec |
| 10TB+ | BigQuery/Snowflake | Redis Labs/Bigtable | 1M+ rows/sec |

### **동시 사용자별 권장 구성**

| 동시 사용자 | 추천 서빙 방식 | 추천 인프라 |
|------------|---------------|-------------|
| 1-10 | 로컬 API 서버 | 단일 인스턴스 |
| 10-100 | Docker 컨테이너 | 로드 밸런서 |
| 100-1K | Kubernetes | 오토스케일링 |
| 1K+ | 서버리스 (Cloud Run/Lambda) | 글로벌 배포 |

---

## 🔧 **트러블슈팅**

### **연결 문제 해결**

**데이터베이스 연결 실패**
```bash
# 1. 연결 문자열 확인
echo $DATABASE_URL

# 2. 네트워크 접근 확인  
telnet your-db-host 5432

# 3. 인증 정보 확인
psql "your-connection-string" -c "SELECT 1;"
```

**Feature Store 연결 실패**
```bash
# Redis 연결 확인
redis-cli -h your-redis-host ping

# DynamoDB 권한 확인
aws dynamodb list-tables --region your-region
```

### **성능 최적화**

**느린 쿼리 개선**
```yaml
# BigQuery 최적화
model:
  loader:
    source_uri: |
      SELECT *
      FROM your_table
      WHERE _PARTITIONTIME >= '2024-01-01'  # 파티션 활용
      LIMIT 1000000  # 적절한 제한
```

**메모리 부족 해결**
```yaml
# 배치 크기 조정
model:
  loader:
    source_uri: "SELECT * FROM large_table LIMIT 100000"  # 샘플링
```

---

**이 가이드를 따라 당신의 현재 인프라에 맞는 최적의 Modern ML Pipeline 구성을 만들어보세요!** 