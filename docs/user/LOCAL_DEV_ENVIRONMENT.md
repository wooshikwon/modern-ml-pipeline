# 로컬 개발 환경 설정

Modern ML Pipeline의 로컬 개발 및 통합 테스트를 위한 환경 설정 가이드입니다.


## 개요

로컬 개발 환경은 별도의 GitHub 저장소 [mmp-local-dev](https://github.com/wooshikwon/mmp-local-dev)에서 관리됩니다. Docker Compose 기반으로 PostgreSQL, Redis, MLflow, Feast Feature Store를 제공합니다.

### 구성 요소

| 서비스 | 포트 | 용도 |
|--------|------|------|
| PostgreSQL | 5432 | Feature Store Offline Store / 데이터 저장소 |
| Redis | 6379 | Feature Store Online Store / 캐시 |
| MLflow | 5000 | 실험 추적 및 모델 레지스트리 |


## 설치

### 1. 저장소 클론

```bash
# modern-ml-pipeline과 같은 디렉토리에 클론
git clone https://github.com/wooshikwon/mmp-local-dev.git
cd mmp-local-dev
```

### 2. 환경 변수 설정

```bash
# 환경 변수 파일 생성
cp .env.example .env

# .env 파일에서 POSTGRES_PASSWORD 설정
# POSTGRES_PASSWORD=your_secure_password_here
```

### 3. 서비스 시작

```bash
docker-compose up -d
```

### 4. 서비스 상태 확인

```bash
docker-compose ps

# 연결 테스트
PGPASSWORD=<password> psql -h localhost -p 5432 -U mluser -d mlpipeline -c "SELECT 1;"
redis-cli -h localhost -p 6379 ping
curl http://localhost:5000/health
```


## Fraud Detection 테스트 데이터

Point-in-Time Join 테스트를 위한 신용카드 사기 탐지 데이터셋이 포함되어 있습니다.

### 데이터셋 정보

- **출처**: [Kaggle Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **기간**: 2019-01-01 ~ 2020-06-21
- **크기**: 50,000 거래 (샘플링)
- **Fraud 비율**: ~1.16%

### 테이블 구조

| 테이블 | 설명 | Point-in-Time |
|--------|------|---------------|
| `transactions` | 거래 이벤트 (Entity DataFrame) | event_timestamp |
| `user_features` | 사용자 거래 통계 (시간에 따라 변함) | created_at |
| `user_demographics` | 사용자 인구통계 | created_at |
| `merchant_features` | 머천트 특성 | created_at |
| `category_features` | 카테고리 통계 | created_at |

### 데이터 셋업

```bash
# 1. Kaggle 데이터 다운로드 (API 키 필요)
kaggle datasets download -d kartik2112/fraud-detection -p data --unzip

# 2. 전체 셋업 실행
./setup-fraud-detection.sh
```

수동 셋업:

```bash
# 1. Docker Compose 시작
docker-compose up -d

# 2. 데이터 전처리
python3 scripts/prepare_fraud_data.py

# 3. PostgreSQL 데이터 로드
bash scripts/load_fraud_data.sh

# 4. Feast 적용
cd feast && feast apply && cd ..
```


## Feast Feature Store

### Feature Views

| Feature View | Entity | 피처 수 | 설명 |
|--------------|--------|---------|------|
| `user_demographics` | user_id | 9 | 사용자 인구통계 (정적) |
| `user_transaction_features` | user_id | 15 | 시간별 거래 통계 (동적) |
| `merchant_features` | merchant_id | 10 | 머천트 특성 |
| `category_features` | category | 7 | 카테고리 통계 |

### Recipe에서 Feature Store 피처 사용

MMP는 내부적으로 Feast Point-in-Time Join을 수행합니다. 사용자는 recipe에서 피처를 지정하기만 하면 됩니다:

```yaml
# recipes/fraud-detection.yaml
name: "fraud-detection"
task_choice: "classification"

data:
  loader:
    source_uri: "sql/transactions.sql"  # Entity DataFrame (거래 이벤트)

  data_interface:
    target_column: "is_fraud"
    entity_columns: ["user_id"]
    timestamp_column: "trans_date_trans_time"

  # Feature Store에서 가져올 피처 지정
  fetcher:
    feature_service: "fraud_detection_features"
    features:
      - "user_transaction_features:avg_amount"
      - "user_transaction_features:transactions_7d"
      - "user_demographics:age"
      - "merchant_features:avg_fraud_rate"

model:
  class_path: "xgboost.XGBClassifier"
```

MMP가 자동으로:
1. `source_uri`에서 Entity DataFrame 로드
2. `timestamp_column` 기준으로 Point-in-Time Join 수행
3. 각 거래 시점에 유효했던 피처만 조인 (Data Leakage 방지)


## Modern ML Pipeline 연동

### Config 설정

로컬 개발 환경용 Config 파일 예시:

```yaml
# configs/local-dev.yaml
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: local-dev

storage:
  adapter: sql
  sql:
    connection_string: postgresql://mluser:<password>@localhost:5432/mlpipeline

feature_store:
  provider: feast
  feast_config:
    project: ml_pipeline_local
    registry: /path/to/mmp-local-dev/feast/data/registry.db
    online_store:
      type: redis
      connection_string: redis://localhost:6379
    offline_store:
      type: postgres
      host: localhost
      port: 5432
      database: mlpipeline
      db_schema: features
      user: mluser
      password: <password>
```

### 학습 실행

```bash
# modern-ml-pipeline 디렉토리에서 실행
mmp train --config configs/local-dev.yaml --recipe recipes/fraud-detection.yaml

# MLflow UI에서 결과 확인
open http://localhost:5000
```

### 추론 실행

```bash
# 배치 추론
mmp batch-inference --config configs/local-dev.yaml --run-id <mlflow_run_id> --data data/test.csv

# API 서빙
mmp serve-api --config configs/local-dev.yaml --run-id <mlflow_run_id>
```


## 디렉토리 구조

```text
mmp-local-dev/
├── docker-compose.yml      # Docker 서비스 정의
├── .env.example            # 환경 변수 템플릿
├── setup-fraud-detection.sh # 전체 셋업 스크립트
├── feast/
│   ├── feature_store.yaml  # Feast 설정
│   └── features.py         # Feature View 정의
├── scripts/
│   ├── init-database.sql   # DB 초기화
│   ├── prepare_fraud_data.py # 데이터 전처리
│   ├── load_fraud_data.sh  # 데이터 로드
│   └── test_point_in_time_join.py # PIT 테스트
└── data/
    ├── fraudTrain.csv      # Kaggle 원본
    └── processed/          # 전처리된 데이터
```


## 서비스 관리

```bash
# 시작
docker-compose up -d

# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down

# 볼륨 포함 삭제 (데이터 초기화)
docker-compose down -v
```


## 문제 해결

### PostgreSQL 연결 실패

```bash
# 연결 테스트
PGPASSWORD=<password> psql -h localhost -p 5432 -U mluser -d mlpipeline -c "SELECT 1;"

# 컨테이너 상태 확인
docker-compose logs postgresql
```

### Redis 연결 실패

```bash
# 연결 테스트
redis-cli -h localhost -p 6379 ping

# 컨테이너 상태 확인
docker-compose logs redis
```

### Feast 오류

```bash
# Feast 재설치
pip install feast[postgres,redis] --upgrade

# Registry 초기화
rm -f feast/data/registry.db
cd feast && feast apply && cd ..
```

### MLflow 연결 실패

```bash
# MLflow 상태 확인
curl http://localhost:5000/health

# 컨테이너 로그 확인
docker-compose logs mlflow
```


## 참고 자료

- [mmp-local-dev GitHub](https://github.com/wooshikwon/mmp-local-dev)
- [Feast Documentation](https://docs.feast.dev/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
