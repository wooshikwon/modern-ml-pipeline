# 마이그레이션 가이드

기존 ML 파이프라인 프로젝트를 Modern ML Pipeline CLI로 마이그레이션하는 방법을 안내합니다.

## 📋 마이그레이션 체크리스트

- [ ] 현재 프로젝트 구조 분석
- [ ] Config와 Recipe 분리
- [ ] 환경별 설정 파일 생성
- [ ] SQL 쿼리 파일 이동
- [ ] 실행 스크립트를 CLI 명령어로 교체
- [ ] 테스트 및 검증
- [ ] 팀원 교육

## 🔄 마이그레이션 시나리오

### 시나리오 1: 단일 Config 파일 사용 중

**현재 구조:**
```
project/
├── config/
│   └── config.yaml    # 모든 환경 설정이 하나에
├── scripts/
│   └── train.py       # 학습 스크립트
└── models/
    └── model.pkl      # 저장된 모델
```

**마이그레이션 단계:**

#### 1. 프로젝트 구조 생성
```bash
# Modern ML Pipeline 구조 생성
mmp init --project-name migrated_project
cd migrated_project
```

#### 2. Config 분리
```bash
# 기존 config를 환경별로 분리
cp ../config/config.yaml configs/local.yaml

# 환경별 설정 생성
mmp get-config --env-name dev
mmp get-config --env-name prod
```

#### 3. Config 파일 수정
```yaml
# Before (config/config.yaml)
database:
  host: localhost
  port: 5432
  user: postgres
  password: secret123
  
mlflow:
  tracking_uri: http://localhost:5000

# After (configs/dev.yaml)
data_adapters:
  adapters:
    sql:
      config:
        connection_uri: "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
        
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI:http://localhost:5000}"
```

#### 4. 환경변수 파일 생성
```bash
# .env.dev
ENV_NAME=dev
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=secret123
DB_NAME=ml_dev
MLFLOW_TRACKING_URI=http://localhost:5000
```

#### 5. Recipe 생성
```yaml
# recipes/model.yaml
name: "migrated_model"
model:
  class_path: "sklearn.ensemble.RandomForestClassifier"
  loader:
    adapter: "sql"
    source_uri: "sql/train_query.sql"
  data_interface:
    task_type: "classification"
    target_column: "target"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
```

#### 6. 실행 스크립트 교체
```bash
# Before
python scripts/train.py --config config/config.yaml

# After
mmp train --recipe-file recipes/model.yaml --env-name dev
```

---

### 시나리오 2: 환경별 Config 파일 사용 중

**현재 구조:**
```
project/
├── config/
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
└── train.py
```

**마이그레이션 단계:**

#### 1. Config 파일 이동
```bash
# configs 디렉토리로 이동
mkdir -p migrated_project/configs
cp config/*.yaml migrated_project/configs/
```

#### 2. 환경변수 추출
각 config 파일에서 하드코딩된 값을 환경변수로 추출:

```yaml
# Before (config/dev.yaml)
database:
  connection: "postgresql://user:pass@localhost:5432/dev_db"

# After (configs/dev.yaml)
data_adapters:
  adapters:
    sql:
      config:
        connection_uri: "${DB_CONNECTION_URI}"
```

#### 3. .env 파일 생성
```bash
# 각 환경별로 .env 파일 생성
for env in dev staging prod; do
  mmp get-config --env-name $env --template $env
  # 생성된 템플릿 편집
  vim .env.$env
done
```

---

### 시나리오 3: Notebook 기반 개발

**현재 구조:**
```
notebooks/
├── 01_data_exploration.ipynb
├── 02_feature_engineering.ipynb
└── 03_model_training.ipynb
```

**마이그레이션 단계:**

#### 1. 코드 추출 및 모듈화
```python
# notebooks/utils.py로 공통 함수 추출
def load_data(connection_string):
    # 데이터 로드 로직
    pass

def preprocess_features(df):
    # 특성 전처리 로직
    pass
```

#### 2. SQL 쿼리 분리
```sql
-- sql/train_features.sql
SELECT 
    user_id,
    feature_1,
    feature_2,
    target
FROM ml_features
WHERE created_at >= '{{ start_date }}'
  AND created_at < '{{ end_date }}'
```

#### 3. Recipe 생성
```yaml
# recipes/notebook_model.yaml
name: "notebook_migrated_model"
model:
  class_path: "xgboost.XGBClassifier"
  loader:
    adapter: "sql"
    source_uri: "sql/train_features.sql"
  preprocessor:
    steps:
      - type: "standard_scaler"
        columns: ["feature_1", "feature_2"]
  hyperparameters:
    # Notebook에서 찾은 최적 파라미터
    n_estimators: 150
    max_depth: 8
    learning_rate: 0.1
```

#### 4. 파이프라인 실행
```bash
# Notebook 대신 CLI로 실행
mmp train --recipe-file recipes/notebook_model.yaml --env-name dev
```

---

## 🔧 일반적인 마이그레이션 작업

### 데이터베이스 연결 마이그레이션

#### SQLAlchemy에서
```python
# Before
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@localhost/db")
df = pd.read_sql("SELECT * FROM table", engine)

# After (config)
data_adapters:
  adapters:
    sql:
      config:
        connection_uri: "${DB_CONNECTION_URI}"

# After (recipe)
loader:
  adapter: "sql"
  source_uri: "sql/query.sql"
```

#### psycopg2에서
```python
# Before
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    database="db",
    user="user",
    password="pass"
)

# After (.env.dev)
DB_HOST=localhost
DB_NAME=db
DB_USER=user
DB_PASSWORD=pass
DB_CONNECTION_URI=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}
```

### MLflow 통합 마이그레이션

#### 수동 MLflow 로깅에서
```python
# Before
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")

# After (자동으로 처리됨)
# configs/dev.yaml
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI}"
  experiment_name: "my_experiment"
```

### Feature Store 마이그레이션

#### Feast 통합
```yaml
# configs/prod.yaml에 추가
feature_store:
  provider: "feast"
  feast_config:
    project: "ml_features"
    registry: "gs://bucket/registry.pb"
    online_store:
      type: "redis"
      connection_string: "${REDIS_HOST}:${REDIS_PORT}"
```

### 모델 서빙 마이그레이션

#### Flask/FastAPI에서
```python
# Before (app.py)
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(data)
    return jsonify(prediction)

# After
mmp serve-api --run-id <mlflow-run-id> --env-name prod --port 8000
```

---

## 📝 Recipe 작성 가이드

### 기존 학습 코드를 Recipe로 변환

#### 1. 모델 정의
```python
# Before (train.py)
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100, max_depth=6)

# After (recipe.yaml)
model:
  class_path: "xgboost.XGBClassifier"
  hyperparameters:
    n_estimators: 100
    max_depth: 6
```

#### 2. 데이터 로드
```python
# Before
df = pd.read_sql("SELECT * FROM features", connection)

# After
loader:
  adapter: "sql"
  source_uri: "sql/features.sql"
```

#### 3. 전처리
```python
# Before
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# After
preprocessor:
  steps:
    - type: "standard_scaler"
      columns: ["feature_1", "feature_2"]
```

---

## 🚀 마이그레이션 자동화 스크립트

```bash
#!/bin/bash
# migrate.sh - 기존 프로젝트 마이그레이션 도우미

# 1. 백업 생성
echo "Creating backup..."
cp -r . ../project_backup_$(date +%Y%m%d)

# 2. Modern ML Pipeline 구조 생성
echo "Initializing MMP structure..."
mmp init --project-name $(basename $PWD)_migrated

# 3. Config 파일 복사
echo "Migrating config files..."
if [ -d "config" ]; then
    cp config/*.yaml configs/
elif [ -d "configs" ]; then
    cp configs/*.yaml configs/
fi

# 4. SQL 파일 복사
echo "Migrating SQL files..."
if [ -d "sql" ]; then
    cp -r sql/* sql/
fi

# 5. 환경별 설정 생성
echo "Creating environment configs..."
for env in dev staging prod; do
    if [ -f "configs/$env.yaml" ]; then
        mmp get-config --env-name $env --template $env --non-interactive
    fi
done

echo "Migration structure created. Please:"
echo "1. Update config files to use environment variables"
echo "2. Create Recipe files for your models"
echo "3. Set up .env files with actual values"
```

---

## ⚠️ 주의사항

### 1. 데이터 보안
- 절대 비밀번호나 API 키를 Config 파일에 하드코딩하지 마세요
- .env 파일은 반드시 .gitignore에 추가하세요
- 프로덕션 환경에서는 Secret Manager 사용을 권장합니다

### 2. 경로 처리
- 상대 경로는 프로젝트 루트 기준으로 작성
- SQL 파일은 sql/ 디렉토리에 위치
- Recipe 파일은 recipes/ 디렉토리에 위치

### 3. 호환성
- Python 3.8+ 필요
- uv 패키지 매니저 사용
- 기존 requirements.txt는 pyproject.toml로 변환

---

## 🔍 트러블슈팅

### 문제: Import 오류
```python
ModuleNotFoundError: No module named 'src'
```

**해결:**
```bash
# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 또는 uv 사용
uv sync
uv run mmp train --recipe-file recipes/model.yaml --env-name dev
```

### 문제: Config 파일 형식 오류
```
ValueError: Config file format invalid
```

**해결:**
```bash
# YAML 검증
python -c "import yaml; yaml.safe_load(open('configs/dev.yaml'))"

# 환경변수 치환 테스트
python -c "
from src.cli.utils.env_loader import load_config_with_env
config = load_config_with_env('dev')
print(config)
"
```

### 문제: 데이터베이스 연결 실패
```
psycopg2.OperationalError: could not connect to server
```

**해결:**
```bash
# 연결 테스트
mmp system-check --env-name dev --actionable

# 환경변수 확인
cat .env.dev
source .env.dev
echo $DB_HOST
```

---

## 📚 추가 리소스

- [사용자 가이드](./USER_GUIDE.md)
- [API 레퍼런스](./API_REFERENCE.md)
- [예제 프로젝트](https://github.com/your-org/mmp-examples)
- [FAQ](./FAQ.md)

## 🤝 도움 요청

마이그레이션 중 문제가 발생하면:
1. [GitHub Issues](https://github.com/your-org/modern-ml-pipeline/issues)에 문의
2. [Discord 커뮤니티](https://discord.gg/your-invite) 참여
3. 이메일: support@your-org.com