# 🚀 test-project - Modern ML Pipeline

**test-project** 프로젝트에 오신 것을 환영합니다! 이 가이드는 데이터 준비부터 프로덕션 배포까지 완전한 머신러닝 파이프라인을 구축하는 과정을 안내합니다.

*2025-09-08 00:15:00에 Modern ML Pipeline으로 생성됨*

---

## 🎯 빠른 시작 (5분)

```bash
# 1. 환경 설정
uv sync
uv add modern-ml-pipeline
uv run mmp system-check

# 2. 프로젝트 설정  
uv run mmp get-config --env-name local

# 3. ML 태스크 선택 및 레시피 생성
uv run mmp get-recipe

# 4. 모델 훈련 (실제 레시피 파일명으로 교체)
uv run mmp train --recipe recipes/your_recipe.yaml --env local

# 5. 추론 실행
uv run mmp inference --recipe recipes/your_recipe.yaml --env local --output predictions.csv
```

**완료!** 🎉 ML 모델이 훈련되어 사용할 준비가 됩니다. 자세한 설명은 아래를 계속 읽어보세요.

---

## 📚 목차

1. [🔧 환경 설정](#-환경-설정)
2. [📊 데이터 준비](#-데이터-준비)
3. [⚙️ 구성 설정](#️-구성-설정)
4. [📝 레시피 생성](#-레시피-생성)
5. [🎯 모델 훈련](#-모델-훈련)
6. [🔮 추론 및 예측](#-추론-및-예측)
7. [📈 MLflow 추적](#-mlflow-추적)
8. [🐳 Docker 배포](#-docker-배포)
9. [🛠 문제 해결](#-문제-해결)

---

## 🔧 환경 설정

### 1단계: UV 설치 (Python 패키지 관리자)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 대안: pip 설치
pip install uv
```

### 2단계: 프로젝트 환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd test-project

# 가상 환경 생성 및 동기화
uv sync

# Modern ML Pipeline 설치
uv add modern-ml-pipeline

# 설치 확인
uv run mmp --help
```

### 3단계: 시스템 확인

```bash
uv run mmp system-check --env local
```

**예상 출력:**
```
✅ Python 환경: OK (3.11.x)
✅ 핵심 의존성: OK
✅ ML 라이브러리: OK  
✅ 데이터베이스 연결: OK
✅ MLflow 설정: OK
🎉 시스템 확인이 성공적으로 완료되었습니다!
```

---

## 📊 데이터 준비

### 프로젝트 디렉토리 구조

프로젝트는 이미 다음과 같은 구조를 가지고 있습니다:

```
test-project/
├── configs/            # 환경 설정 파일 (get-config로 생성)
├── data/              # 훈련 및 추론 데이터 파일
├── recipes/           # ML 파이프라인 레시피 (get-recipe로 생성)  
├── sql/               # 데이터베이스 소스용 SQL 쿼리
├── docker-compose.yml # 다중 서비스 배포
├── Dockerfile         # 컨테이너 정의
├── pyproject.toml     # 프로젝트 의존성
└── README.md          # 이 가이드
```

### 지원되는 데이터 소스

#### 📁 로컬 파일 (CSV/Parquet)
`data/` 디렉토리에 데이터 파일을 배치하세요:
- `data/train.csv` - 훈련 데이터
- `data/test.csv` - 테스트 데이터 (선택사항)
- `data/inference.csv` - 예측용 새 데이터

#### 🗃️ SQL 데이터베이스
`sql/` 디렉토리에 SQL 쿼리를 저장하세요:
- PostgreSQL, MySQL, BigQuery 등에 연결
- 레시피 설정에서 쿼리 참조

#### ☁️ 클라우드 스토리지
다음에 대한 액세스 설정:
- Google Cloud Storage (GCS)
- Amazon S3
- Azure Blob Storage

#### 🏪 피처 스토어
피처 스토어와 통합:
- Feast
- Tecton
- 커스텀 피처 스토어

### 데이터 품질 가이드라인

- **형식**: CSV, Parquet, 또는 SQL 쿼리 결과
- **크기**: 훈련용으로 최소 1000행 권장
- **결측값**: 주요 피처에서 30% 미만 결측
- **타겟 컬럼**: 지도학습 태스크에 필요
- **피처 타입**: 수치형과 범주형 피처 혼합 지원

---

## ⚙️ 구성 설정

### 1단계: 환경 설정 생성

```bash
uv run mmp get-config --env-name local
```

이 대화형 명령어는 다음 사항을 안내합니다:
- **환경 이름** (local, dev, staging, prod)
- **MLflow 설정** (추적 URI, 실험 명명)
- **데이터 소스 설정** (데이터베이스, 클라우드, 로컬 파일)
- **피처 스토어 설정** (선택사항)
- **서빙 설정** (API 포트, 호스트 설정)

**생성된 설정 예시:**
```yaml
# configs/local.yaml
environment:
  name: "local"
  
mlflow:
  tracking_uri: "file://./mlruns"
  experiment_name: "test-project_experiment"
  
data_source:
  name: "test-project_data"
  adapter_type: "storage"  # 또는 "sql", "bigquery"
  config:
    base_path: "./data"
    
feature_store:
  provider: "none"  # 또는 "feast"
  
serving:
  enabled: true
  host: "0.0.0.0"
  port: 8000
```

### 2단계: 설정 테스트

```bash
# 모든 연결이 작동하는지 확인
uv run mmp system-check --env-name local --verbose

# 특정 구성요소 테스트
uv run mmp test-connection --env local
```

---

## 📝 레시피 생성

### 1단계: 대화형 레시피 생성

```bash
uv run mmp get-recipe
```

이 대화형 명령어는 다음을 선택하도록 요청합니다:

#### 🎯 **ML 태스크 유형**
- **분류 (Classification)**: 범주 예측 (예: 스팸/정상, 이미지 분류)
- **회귀 (Regression)**: 연속값 예측 (예: 주택 가격, 온도) 
- **군집화 (Clustering)**: 유사한 데이터 포인트 그룹화 (예: 고객 세분화)
- **인과 (Causal)**: 원인-결과 관계 분석
- **시계열 (Time Series)**: 시간 패턴 기반 미래값 예측

#### 🤖 **모델 알고리즘**
태스크 선택에 따라 다음 중에서 선택:

**분류 모델:**
- LogisticRegression - 빠르고 해석 가능
- RandomForestClassifier - 견고하며 혼합 데이터 타입 처리
- XGBClassifier - 고성능 그래디언트 부스팅
- LGBMClassifier - 빠른 그래디언트 부스팅
- CatBoostClassifier - 범주형 피처에 최적

**회귀 모델:**
- LinearRegression - 단순한 베이스라인
- RandomForestRegressor - 비선형 패턴
- XGBRegressor - 고성능
- LGBMRegressor - 빠른 훈련

**더 많은 모델을 사용할 수 있습니다!**

#### 📊 **데이터 설정**
- **타겟 컬럼**: 예측할 컬럼
- **피처 선택**: 피처로 사용할 컬럼들
- **데이터 소스**: 훈련 데이터 경로

### 2단계: 레시피 커스터마이징

생성된 레시피 파일 (예: `recipes/classification_recipe.yaml`)을 커스터마이징할 수 있습니다:

```yaml
name: "test-project_model"
task_choice: "classification"  # 선택에 따라 설정됨

data:
  data_interface:
    target_column: "your_target_column"  # 지정할 컬럼
    drop_columns: []
  
  feature_view:
    name: "test-project_features"
    features: []  # 자동 감지 또는 수동 지정
    source:
      path: "train.csv"  # 데이터 파일

model:
  class_path: "sklearn.linear_model.LogisticRegression"  # 선택 기반
  init_args:
    random_state: 42
    max_iter: 1000

preprocessor:
  steps:
    - name: "encoder"
      type: "categorical"
      params:
        categorical_features: []  # 자동 감지
        encoding_type: "onehot"
    
    - name: "scaler"
      type: "numerical"  
      params:
        method: "standard"
        features: []  # 자동 감지

trainer:
  validation_split: 0.2
  stratify: true  # 분류용
  random_state: 42
```

---

## 🎯 모델 훈련

### 1단계: 훈련 시작

```bash
# 생성된 레시피 파일 사용
uv run mmp train \
  --recipe recipes/your_recipe.yaml \
  --env local \
  --experiment-name "test-project_v1"
```

### 2단계: 훈련 진행 상황 모니터링

**예상 출력 (분류 예시):**
```
🚀 ML 파이프라인 훈련을 시작합니다...
📊 스토리지에서 데이터를 로딩합니다...
✅ 데이터 로딩 완료: 1000 샘플, 5개 피처
🔧 데이터 전처리 중...
   - 범주형 피처 인코딩: ['category_col']
   - 수치형 피처 스케일링: ['numeric_col_1', 'numeric_col_2']
✅ 전처리 완료
🎯 LogisticRegression 모델 훈련 중...
✅ 훈련 완료 (30.2초)

📈 훈련 지표:
  - 정확도: 0.856
  - 정밀도: 0.842  
  - 재현율: 0.871
  - F1-점수: 0.856

🎉 MLflow에 모델 저장됨: runs:/abc123def/model
```

**예상 출력 (회귀 예시):**
```
🚀 ML 파이프라인 훈련을 시작합니다...
📊 스토리지에서 데이터를 로딩합니다...
✅ 데이터 로딩 완료: 1000 샘플, 5개 피처
🔧 데이터 전처리 중...
✅ 전처리 완료
🎯 RandomForestRegressor 모델 훈련 중...
✅ 훈련 완료 (45.1초)

📈 훈련 지표:
  - R² 점수: 0.823
  - MAE: 2.14
  - RMSE: 3.47

🎉 MLflow에 모델 저장됨: runs:/def456ghi/model
```

### 3단계: 훈련 변형

```bash
# 커스텀 하이퍼파라미터로 훈련
uv run mmp train \
  --recipe recipes/your_recipe.yaml \
  --env local \
  --model-params '{"n_estimators": 200, "max_depth": 10}'

# 검증 데이터로 훈련
uv run mmp train \
  --recipe recipes/your_recipe.yaml \
  --env local \
  --validation-data data/validation.csv
```

---

## 🔮 추론 및 예측

### 1단계: 배치 추론

```bash
uv run mmp inference \
  --recipe recipes/your_recipe.yaml \
  --env local \
  --input data/inference.csv \
  --output predictions.csv \
  --model-uri "runs:/abc123def/model"
```

### 2단계: 예측 결과 확인

```bash
head -10 predictions.csv
```

**예상 출력 (분류):**
```csv
id,prediction,probability_0,probability_1
1,1,0.234,0.766
2,0,0.671,0.329
3,1,0.123,0.877
```

**예상 출력 (회귀):**
```csv
id,prediction,confidence_interval_lower,confidence_interval_upper
1,45.67,42.12,49.22
2,67.89,64.34,71.44
3,23.45,19.90,27.00
```

### 3단계: 실시간 API 추론

```bash
# API 서버 시작
uv run mmp serve \
  --recipe recipes/your_recipe.yaml \
  --env local \
  --model-uri "runs:/abc123def/model" \
  --port 8000

# API 엔드포인트 테스트
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature_1": 1.23,
      "feature_2": "category_A",
      "feature_3": 45.67
    }
  }'
```

**API 응답 (분류):**
```json
{
  "prediction": 1,
  "probability": [0.234, 0.766],
  "confidence": "high",
  "model_version": "v1"
}
```

---

## 📈 MLflow 추적

### MLflow UI 접속

```bash
# MLflow UI는 훈련 중에 자동 시작되거나 수동으로 시작:
uv run mlflow ui --host 0.0.0.0 --port 5000

# 브라우저에서 열기
open http://localhost:5000
```

### 사용 가능한 MLflow 기능:

- **📊 실험 추적**: 다양한 모델 실행과 하이퍼파라미터 비교
- **📈 지표 시각화**: 훈련/검증 곡선 및 성능 플롯
- **🏷️ 모델 레지스트리**: 모델의 버전 관리 및 스테이징
- **📝 아티팩트 저장**: 모델 파일, 전처리 파이프라인, 평가 플롯
- **🔄 모델 생명주기**: 개발 → 스테이징 → 프로덕션 워크플로우

### 실험 보기:

1. **실험** → `test-project_experiment`로 이동
2. 지표, 파라미터, 훈련 시간으로 실행 비교
3. 모델 아티팩트와 전처리 파이프라인 다운로드
4. 프로덕션용 최고 성능 모델 등록

---

## 🐳 Docker 배포

### 1단계: Docker 이미지 빌드

```bash
# 프로덕션 이미지 빌드
docker build -f Dockerfile -t test-project:latest .

# 특정 모델 URI로 빌드
docker build \
  --build-arg MODEL_URI="runs:/abc123def/model" \
  -f Dockerfile \
  -t test-project:v1 .
```

### 2단계: 컨테이너 실행

```bash
# API 서버 실행
docker run -d \
  --name test-project-api \
  -p 8000:8000 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/mlruns:/app/mlruns \
  test-project:latest

# 컨테이너 상태 확인
docker logs test-project-api
```

### 3단계: Docker Compose (권장)

```bash
# 전체 스택 시작 (API + MLflow + 데이터베이스)
docker-compose up -d

# 서비스 보기
docker-compose ps

# API 인스턴스 확장
docker-compose up --scale api=3
```

**사용 가능한 서비스:**
- **API 서버**: http://localhost:8000
- **MLflow UI**: http://localhost:5000  
- **헬스 체크**: http://localhost:8000/health

### 4단계: 프로덕션 배포

```bash
# 클라우드 배포 (Google Cloud Run 예시)
gcloud builds submit --tag gcr.io/your-project/test-project
gcloud run deploy test-project \
  --image gcr.io/your-project/test-project \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 🛠 문제 해결

### 일반적인 문제 및 해결책

#### 🚨 환경 문제

**문제**: `command not found: uv`
```bash
# 해결책: uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # 또는 터미널 재시작
```

**문제**: `ModuleNotFoundError: No module named 'modern-ml-pipeline'`
```bash
# 해결책: 패키지 설치
uv add modern-ml-pipeline
```

#### 🚨 설정 문제

**문제**: `get-config 명령어 실패`
```bash
# 해결책: 대화형으로 실행하고 입력 확인
uv run mmp get-config --env-name local --verbose

# 생성된 설정 확인
cat configs/local.yaml
```

**문제**: `데이터베이스 연결 실패`
```bash
# 해결책: 연결을 별도로 테스트
uv run mmp test-connection --env local

# configs/local.yaml에서 데이터베이스 자격 증명 확인
```

#### 🚨 데이터 문제

**문제**: `FileNotFoundError: data/train.csv`
```bash
# 해결책: 데이터 경로와 파일 존재 확인
ls -la data/
# 데이터 파일이 올바른 위치에 있는지 확인
```

**문제**: `타겟 컬럼 'target'을 찾을 수 없음`
```bash
# 해결책: 데이터의 컬럼 이름 확인
uv run python -c "
import pandas as pd
df = pd.read_csv('data/train.csv')
print('사용 가능한 컬럼:', df.columns.tolist())
"
```

#### 🚨 훈련 문제

**문제**: `MLflow 서버에 액세스할 수 없음`
```bash
# 해결책: MLflow 서버 수동 시작
uv run mlflow ui --host 0.0.0.0 --port 5000 &

# 또는 이미 실행 중인지 확인
ps aux | grep mlflow
```

**문제**: `훈련 중 메모리 오류`
```bash
# 해결책: 배치 크기 줄이거나 데이터 샘플링
# 레시피 파일 편집:
# loader:
#   batch_size: 100  # 기본값에서 감소
```

#### 🚨 Docker 문제

**문제**: `Docker 빌드 실패`
```bash
# 해결책: Docker 실행 확인
docker --version
docker info

# Linux: sudo systemctl start docker
# Mac/Windows: Docker Desktop 시작
```

**문제**: `컨테이너가 즉시 종료됨`
```bash
# 해결책: 컨테이너 로그 확인
docker logs test-project-api

# 일반적인 원인:
# - 환경 변수 누락
# - 잘못된 파일 경로
# - 포트 충돌
```

### 디버그 명령어

```bash
# 전체 시스템 상태 확인
uv run mmp system-check --verbose

# 설정 파일 검증
uv run mmp validate-config configs/local.yaml

# 훈련 없이 데이터 연결 테스트
uv run mmp test-connection --env local

# 레시피 형식 검증
uv run mmp validate-recipe recipes/your_recipe.yaml

# MLflow 서버 확인
curl http://localhost:5000/health
```

### 도움 받기

1. **📖 문서**: 이 README는 대부분의 일반적인 시나리오를 다룹니다
2. **🐛 디버그 모드**: 자세한 로그를 위해 모든 명령어에 `--verbose` 추가
3. **📧 이슈**: [GitHub Issues](https://github.com/your-org/modern-ml-pipeline/issues)에서 버그 신고
4. **💬 커뮤니티**: 도움말과 팁을 위한 토론 참여
5. **📝 로그**: 자세한 오류 정보는 `logs/` 디렉토리 확인

---

## 🎉 다음 단계 및 모범 사례

### 🚀 프로덕션 체크리스트

- [ ] **모델 성능**: 정확도/성능 요구사항 충족
- [ ] **데이터 파이프라인**: 자동화되고 안정적인 데이터 수집
- [ ] **모니터링**: 모델 성능 모니터링 설정
- [ ] **API 테스트**: 서빙 엔드포인트 부하 테스트
- [ ] **보안**: 데이터 접근 및 API 인증 검토
- [ ] **문서화**: 특정 세부사항으로 이 README 업데이트
- [ ] **CI/CD**: 자동화된 테스트 및 배포 설정

### 🔄 지속적인 개선

**모델 관리:**
- 새 데이터로 자동 재훈련 설정
- 모델 버전의 A/B 테스팅 구현
- 모델 드리프트 및 데이터 품질 모니터링

**피처 엔지니어링:**
- 훈련된 모델에서 피처 중요도 탐색
- 도메인별 전처리 단계 추가
- 재사용 가능한 피처를 위한 피처 스토어 구현

**고급 MLOps:**
- 확장 가능한 배포를 위한 Kubernetes 통합
- 워크플로우 오케스트레이션을 위한 Airflow 설정
- ML 파이프라인 관리를 위한 Kubeflow 사용

### 📚 고급 기능

이러한 강력한 기능들을 탐색해보세요:

```bash
# 피처 스토어 통합 (설정된 경우)
uv run mmp setup-feast --project test-project

# Optuna를 사용한 하이퍼파라미터 최적화
uv run mmp tune --recipe recipes/your_recipe.yaml --trials 100

# 다중 모델 앙상블
uv run mmp ensemble --models model1,model2,model3

# 모델 모니터링 및 알림
uv run mmp monitor --model-uri "runs:/abc123def/model" --threshold 0.1

# 대용량 데이터셋을 위한 배치 처리
uv run mmp batch-process --recipe recipes/your_recipe.yaml --chunk-size 10000
```

### 🏗️ 프로젝트 커스터마이징

**레시피 업데이트:**
- 커스텀 전처리 단계 추가
- 피처 선택 방법 포함
- 교차 검증 전략 설정

**환경 설정:**
- dev/staging/prod용 별도 설정 생성
- 클라우드 스토리지 및 데이터베이스 설정
- 모니터링 및 알림 설정

**Docker 최적화:**
- 더 작은 이미지를 위한 다단계 빌드
- 헬스 체크 및 graceful shutdown
- 리소스 제한 및 스케일링 정책

---

## 📄 최종 프로젝트 구조

이 가이드를 따른 후 프로젝트는 다음과 같은 구조를 가집니다:

```
test-project/
├── configs/
│   ├── local.yaml              # 로컬 개발 설정
│   ├── staging.yaml           # 스테이징 환경 (선택사항)
│   └── production.yaml        # 프로덕션 환경 (선택사항)
├── data/
│   ├── train.csv              # 훈련 데이터
│   ├── test.csv               # 테스트/검증 데이터
│   └── inference.csv          # 예측용 새 데이터
├── recipes/
│   ├── classification_recipe.yaml    # ML 파이프라인 레시피
│   ├── regression_recipe.yaml        # (선택에 따라 생성)
│   └── your_custom_recipe.yaml
├── sql/
│   ├── training_query.sql     # 데이터베이스 쿼리 (SQL 소스 사용 시)
│   └── feature_extraction.sql
├── mlruns/                    # MLflow 실험 추적
├── logs/                      # 애플리케이션 로그
├── predictions.csv            # 모델 출력
├── docker-compose.yml         # 다중 서비스 배포
├── Dockerfile                 # 컨테이너 정의
├── pyproject.toml            # 프로젝트 의존성
└── README.md                 # 이 가이드 (커스터마이징하세요!)
```

---

**🎊 축하합니다!** Modern ML Pipeline 프로젝트를 성공적으로 설정했습니다. 이제 프로덕션에서 머신러닝 모델을 구축, 훈련, 배포하는 데 필요한 모든 것을 갖추었습니다.

**기억하세요**: 이 README는 시작점일 뿐입니다. 특정 사용 사례, 데이터 세부사항 및 배포 요구사항에 맞게 커스터마이징하세요.

즐거운 모델링하세요! 🤖✨