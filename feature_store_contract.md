# Feature Store & Infrastructure Contract

## 📋 **문서 목적**

이 문서는 **modern-ml-pipeline**과 **mmp-local-dev** 프로젝트 간의 인프라 책임 분리와 연동 방식을 정의합니다. Blueprint v17.0의 **"레시피는 논리, 설정은 인프라"** 원칙에 따라 **완전한 책임 분리**를 구현하는 계약서입니다.

---

## 🏗️ **아키텍처 책임 분리**

### **modern-ml-pipeline 프로젝트 책임**
```yaml
역할: ML 로직 및 어댑터 타입 정의
책임:
  - Recipe 파일 관리 (모델 논리)
  - 어댑터 타입 선택 (config/*.yaml)
  - 환경변수 읽기 및 연결
  - Factory Registry 패턴 구현

관여하지 않는 영역:
  - 실제 인프라 구축
  - 데이터베이스 설정
  - 컨테이너 관리
  - 연결 정보 관리
```

### **mmp-local-dev 프로젝트 책임**
```yaml
역할: 완전한 인프라 관리 및 제공
책임:
  - Docker Compose 인프라 구축
  - 환경변수 템플릿 제공
  - 실제 연결 정보 관리
  - Feature Store 데이터 구축
  - Health Check 및 모니터링

제공하지 않는 영역:
  - ML 모델 논리
  - Recipe 파일 관리
  - 어댑터 구현
  - 비즈니스 로직
```

---

## 🔧 **환경변수 기반 연결 체계**

### **환경변수 구조 설계**
```bash
# mmp-local-dev/.env.example
# PostgreSQL (필수)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=mluser
POSTGRES_DB=mlpipeline
POSTGRES_PASSWORD=  # 필수 설정

# Redis (선택적)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=  # 선택적 설정

# MLflow (선택적)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=./mlruns

# Feature Store (선택적)
FEATURE_STORE_OFFLINE_URI=postgresql://mluser:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}
FEATURE_STORE_ONLINE_URI=redis://${REDIS_HOST}:${REDIS_PORT}
```

### **환경변수 사용 원칙**
```yaml
1. 필수 vs 선택적 분리:
   - 필수: POSTGRES_PASSWORD (보안상 반드시 사용자 입력)
   - 선택적: 기본값 제공, 필요시 오버라이드

2. 조합 가능한 구조:
   - Base 설정 + 환경별 오버라이드
   - 개발자별 로컬 설정 가능

3. 보안 고려사항:
   - 민감정보는 .env에만 저장
   - 기본값은 개발환경에 적합하게 설정
```

---

## 🏭 **Factory Registry 패턴**

### **확장적 어댑터 시스템**
```python
# modern-ml-pipeline/src/core/registry.py
class AdapterRegistry:
    """완전히 확장적인 어댑터 등록 시스템"""
    
    _adapters = {}
    
    @classmethod
    def register(cls, adapter_type: str):
        """어댑터 등록 데코레이터"""
        def decorator(adapter_class):
            cls._adapters[adapter_type] = adapter_class
            return adapter_class
        return decorator
    
    @classmethod
    def create(cls, adapter_type: str, settings: Settings) -> BaseAdapter:
        """동적 어댑터 생성"""
        if adapter_type not in cls._adapters:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        return cls._adapters[adapter_type](settings)
```

### **어댑터 구현 예시**
```python
# modern-ml-pipeline/src/utils/adapters/postgresql_adapter.py
from src.core.registry import AdapterRegistry
import os

@AdapterRegistry.register("postgresql")
class PostgreSQLAdapter(BaseAdapter):
    """환경변수 기반 PostgreSQL 어댑터"""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = int(os.getenv('POSTGRES_PORT', '5432'))
        self.user = os.getenv('POSTGRES_USER', 'mluser')
        self.database = os.getenv('POSTGRES_DB', 'mlpipeline')
        self.password = os.getenv('POSTGRES_PASSWORD')  # 필수
        
        if not self.password:
            raise ValueError("POSTGRES_PASSWORD 환경변수가 설정되지 않았습니다")
    
    def read(self, source_uri: str, **kwargs) -> pd.DataFrame:
        """SQL 파일 실행 및 결과 반환"""
        # 환경변수 기반 연결 정보로 PostgreSQL 접속
        pass
```

### **Config 기반 동적 결정**
```yaml
# modern-ml-pipeline/config/dev.yaml
data_adapters:
  loader: "postgresql"        # Registry에서 PostgreSQLAdapter 선택
  storage: "filesystem"       # Registry에서 FileSystemAdapter 선택
  feature_store: "postgresql" # Registry에서 PostgreSQLAdapter 선택

# 실제 연결 정보는 환경변수에서 주입
# if-else 분기 없이 YAML 설정으로 자연스럽게 결정
```

---

## 🐳 **mmp-local-dev 프로젝트 구조**

### **디렉토리 구조**
```
mmp-local-dev/
├── docker-compose.yml          # 핵심 인프라 정의
├── .env.example               # 환경변수 템플릿
├── setup.sh                   # 원스톱 설치 스크립트
├── scripts/
│   ├── init-database.sql      # PostgreSQL 초기화
│   ├── seed-features.sql      # 샘플 Feature 데이터
│   └── health-check.sh        # 서비스 상태 확인
├── config/
│   ├── postgres.conf          # PostgreSQL 설정
│   └── redis.conf             # Redis 설정
├── feast/
│   ├── feature_store.yaml     # Feast 설정
│   └── feature_definitions.py # 피처 정의
└── README.md                  # 사용법 가이드
```

### **핵심 구성 요소**

#### **1. Docker Compose 인프라**
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-database.sql:/docker-entrypoint-initdb.d/init.sql
      - ./scripts/seed-features.sql:/docker-entrypoint-initdb.d/seed.sql
  
  redis:
    image: redis:7-alpine
    ports:
      - "${REDIS_PORT}:6379"
    volumes:
      - redis_data:/data
  
  mlflow:
    image: python:3.11-slim
    command: >
      sh -c "pip install mlflow psycopg2-binary &&
             mlflow server --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    depends_on:
      - postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}

volumes:
  postgres_data:
  redis_data:
```

#### **2. 원스톱 설치 스크립트**
```bash
# setup.sh
#!/bin/bash
set -e

echo "🚀 MMP Local Dev Environment Setup"

# 환경변수 파일 확인
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  .env 파일에서 POSTGRES_PASSWORD를 설정해주세요"
    exit 1
fi

# 인프라 시작
echo "🐳 Docker 인프라 시작 중..."
docker-compose up -d

# 서비스 대기
echo "⏳ 서비스 준비 대기 중..."
./scripts/health-check.sh

echo "✅ 개발 환경 준비 완료!"
echo "  PostgreSQL: localhost:${POSTGRES_PORT}"
echo "  Redis: localhost:${REDIS_PORT}"
echo "  MLflow: http://localhost:5000"
```

#### **3. Feature Store 데이터 구축**
```sql
-- scripts/seed-features.sql
-- 샘플 피처 데이터 생성
CREATE SCHEMA IF NOT EXISTS features;

-- 사용자 기본 정보
CREATE TABLE features.user_demographics (
    user_id VARCHAR(50) PRIMARY KEY,
    age INTEGER,
    country_code VARCHAR(2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자 구매 요약
CREATE TABLE features.user_purchase_summary (
    user_id VARCHAR(50) PRIMARY KEY,
    ltv DECIMAL(10,2),
    total_purchase_count INTEGER,
    last_purchase_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 제품 상세 정보
CREATE TABLE features.product_details (
    product_id VARCHAR(50) PRIMARY KEY,
    price DECIMAL(10,2),
    category VARCHAR(100),
    brand VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 샘플 데이터 삽입
INSERT INTO features.user_demographics (user_id, age, country_code) VALUES
    ('user_001', 25, 'US'),
    ('user_002', 32, 'UK'),
    ('user_003', 28, 'CA');

INSERT INTO features.user_purchase_summary (user_id, ltv, total_purchase_count, last_purchase_date) VALUES
    ('user_001', 1250.50, 15, '2025-01-10'),
    ('user_002', 850.75, 8, '2025-01-08'),
    ('user_003', 2100.30, 25, '2025-01-12');

INSERT INTO features.product_details (product_id, price, category, brand) VALUES
    ('prod_001', 29.99, 'Electronics', 'TechBrand'),
    ('prod_002', 149.99, 'Fashion', 'StyleCorp'),
    ('prod_003', 79.99, 'Home', 'HomeInc');
```

---

## 🔄 **연동 워크플로우**

### **개발 환경 구성 프로세스**
```bash
# 1. mmp-local-dev 클론 및 설정
git clone https://github.com/your-org/mmp-local-dev.git
cd mmp-local-dev
cp .env.example .env
# .env 파일에서 POSTGRES_PASSWORD 설정

# 2. 인프라 시작
./setup.sh

# 3. modern-ml-pipeline 프로젝트로 이동
cd ../modern-ml-pipeline

# 4. 환경변수 로드 및 실행
source ../mmp-local-dev/.env  # 또는 direnv 사용
APP_ENV=dev python main.py train --recipe-file dev_classification_test
```

### **어댑터 연동 흐름**
```mermaid
graph TD
    A[Recipe 실행] --> B[Config 읽기]
    B --> C[data_adapters.loader = 'postgresql']
    C --> D[Factory.create_data_adapter('loader')]
    D --> E[AdapterRegistry.create('postgresql')]
    E --> F[PostgreSQLAdapter.__init__]
    F --> G[환경변수 읽기]
    G --> H[POSTGRES_HOST, POSTGRES_PORT, POSTGRES_PASSWORD]
    H --> I[실제 PostgreSQL 연결]
```

---

## 🚀 **확장 방식**

### **새로운 어댑터 추가**
```python
# 1. 새 어댑터 구현
@AdapterRegistry.register("snowflake")
class SnowflakeAdapter(BaseAdapter):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.account = os.getenv('SNOWFLAKE_ACCOUNT')
        self.user = os.getenv('SNOWFLAKE_USER')
        self.password = os.getenv('SNOWFLAKE_PASSWORD')
        # ... 환경변수 기반 설정

# 2. Config에서 선택
# config/prod.yaml
data_adapters:
  loader: "snowflake"  # Registry에서 SnowflakeAdapter 자동 선택

# 3. 환경변수 설정
# .env
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USER=your-user
SNOWFLAKE_PASSWORD=your-password
```

### **새로운 환경 추가**
```yaml
# config/staging.yaml
data_adapters:
  loader: "bigquery"
  storage: "gcs"
  feature_store: "bigquery"

# 환경변수만 설정하면 자동으로 연결
# GCP_PROJECT_ID, GCP_CREDENTIALS_PATH 등
```

---

## 🛡️ **보안 고려사항**

### **환경변수 관리**
```bash
# 개발환경
# mmp-local-dev/.env (git ignore에 포함)
POSTGRES_PASSWORD=local_dev_password

# 운영환경
# 시스템 환경변수 또는 시크릿 관리 도구 사용
export POSTGRES_PASSWORD="$(kubectl get secret postgres-secret -o jsonpath='{.data.password}' | base64 -d)"
```

### **접근 제어**
```yaml
보안 원칙:
  - 민감정보는 환경변수에만 저장
  - .env 파일은 반드시 .gitignore에 포함
  - 운영환경에서는 시크릿 관리 도구 사용
  - 개발환경과 운영환경의 완전한 분리
```

---

## 📊 **성능 및 모니터링**

### **Health Check 시스템**
```bash
# scripts/health-check.sh
#!/bin/bash

echo "🔍 서비스 상태 확인 중..."

# PostgreSQL 연결 테스트
docker-compose exec postgres pg_isready -U $POSTGRES_USER -d $POSTGRES_DB
if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL 정상"
else
    echo "❌ PostgreSQL 연결 실패"
    exit 1
fi

# Redis 연결 테스트
docker-compose exec redis redis-cli ping
if [ $? -eq 0 ]; then
    echo "✅ Redis 정상"
else
    echo "❌ Redis 연결 실패"
    exit 1
fi

echo "🎉 모든 서비스 정상 동작 중"
```

### **모니터링 대시보드**
```yaml
제공 서비스:
  - MLflow UI: http://localhost:5000 (실험 추적)
  - pgAdmin: http://localhost:8082 (PostgreSQL 관리)
  - Redis Commander: http://localhost:8081 (Redis 모니터링)
```

---

## 🎯 **사용 시나리오**

### **시나리오 1: 새로운 개발자 온보딩**
```bash
# 1. 저장소 클론
git clone https://github.com/your-org/mmp-local-dev.git
git clone https://github.com/your-org/modern-ml-pipeline.git

# 2. 개발환경 구성 (5분)
cd mmp-local-dev
cp .env.example .env
# .env에서 POSTGRES_PASSWORD 설정
./setup.sh

# 3. 첫 번째 실험 실행 (2분)
cd ../modern-ml-pipeline
APP_ENV=dev python main.py train --recipe-file dev_classification_test

# 총 7분 이내 완전한 개발환경 구축 완료
```

### **시나리오 2: 새로운 데이터 소스 추가**
```python
# 1. 어댑터 구현
@AdapterRegistry.register("mongodb")
class MongoDBAdapter(BaseAdapter):
    def __init__(self, settings: Settings):
        self.connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        # ... 구현

# 2. Config 수정
# config/dev.yaml
data_adapters:
  loader: "mongodb"  # Factory가 자동으로 MongoDBAdapter 선택

# 3. 환경변수 설정
# .env
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/mlpipeline
```

### **시나리오 3: 운영환경 배포**
```yaml
# 1. 운영환경 Config 생성
# config/prod.yaml
data_adapters:
  loader: "bigquery"
  storage: "gcs"
  feature_store: "bigquery"

# 2. 운영환경 환경변수 설정
# 시크릿 관리 도구 또는 시스템 환경변수
export GCP_PROJECT_ID="your-prod-project"
export GCP_CREDENTIALS_PATH="/path/to/credentials.json"

# 3. 동일한 코드로 운영환경 실행
APP_ENV=prod python main.py train --recipe-file prod_model_recipe
```

---

## 🏆 **핵심 장점**

### **Blueprint 철학 완전 구현**
```yaml
1. 완전한 책임 분리:
   - ML 코드는 어댑터 타입만 선택
   - 인프라는 mmp-local-dev가 완전 관리
   - 환경변수를 통한 느슨한 결합

2. 확장성 보장:
   - 새 어댑터 추가 시 Factory 코드 변경 불필요
   - Registry 패턴으로 완전 동적 생성
   - YAML 설정으로 자연스러운 선택

3. 개발자 경험:
   - 5분 이내 완전한 개발환경 구축
   - 코드 변경 없이 환경별 전환
   - 명확한 에러 메시지와 디버깅 지원
```

---

이 계약에 따라 **modern-ml-pipeline**은 ML 로직에만 집중하고, **mmp-local-dev**는 인프라 관리에만 집중하여 **Blueprint v17.0의 완전한 실현**을 달성할 수 있습니다. 🚀