# {PROJECT_NAME}

이 프로젝트는 Modern ML Pipeline으로 생성되었습니다.

## 🚀 4단계 워크플로우로 빠른 시작

### 1️⃣ Python 환경 설정
```bash
# Python 3.11 설치 확인
python --version  # 3.11.x 이어야 함

# uv 설치 (패키지 매니저)  
curl -LsSf https://astral.sh/uv/install.sh | sh

# 패키지 동기화
uv sync
```

### 2️⃣ 환경 설정 & Config 생성
```bash  
# 환경변수 설정 (.env.template을 참고해서 .env 생성)
cp .env.template .env

# 환경별 설정 예시
echo "APP_ENV=local" >> .env        # 로컬 개발
# 또는
echo "APP_ENV=dev" >> .env          # 개발 서버  
# 또는  
echo "APP_ENV=prod" >> .env         # 운영 서버

# 추가 환경변수 설정 (DB, Redis 등)
vi .env  # 필요한 설정값들 입력

# Config 파일들 생성 (.env 기반으로 configs/*.yaml 자동 생성)
uv run modern-ml-pipeline get-config
```

### 3️⃣ 환경 연결 검증 ✅
```bash
# 생성된 configs/*.yaml 파일들로 실제 서비스 연결 테스트
uv run modern-ml-pipeline system-check

# 🔍 검증 항목:
# - Database 연결 (PostgreSQL)
# - Cache 연결 (Redis)  
# - MLflow 서버 연결
# - Cloud Storage 연결 (GCS/AWS S3)
# - Feature Store 연결
```

### 4️⃣ ML 레시피 생성
```bash
# 환경 독립적 ML 레시피 생성 (task + model 선택만)
uv run modern-ml-pipeline get-recipe

# 대화형으로 선택:
# 1) Task 선택: Classification, Regression, etc.
# 2) Model 선택: 카탈로그에서 사용 가능한 모델들
```

### 5️⃣ 모델 학습 & 추론
```bash
# 모델 학습 (검증된 환경 + ML 레시피)
uv run modern-ml-pipeline train --recipe-file recipes/your_recipe.yaml

# 배치 추론
uv run modern-ml-pipeline batch-inference --recipe-file recipes/your_recipe.yaml

# API 서빙 
uv run modern-ml-pipeline api-serving --recipe-file recipes/your_recipe.yaml
```

## 📁 프로젝트 구조

```
{PROJECT_NAME}/
├── configs/         # 환경별 설정 파일 (get-config로 생성)
├── recipes/         # ML 레시피 파일들 (get-recipe로 생성)  
├── data/           # 데이터 파일들
├── sql/            # SQL 쿼리 파일들
├── .env            # 🔥 환경변수 (모든 환경 전환의 중심!)
├── .env.template   # 환경변수 템플릿
├── pyproject.toml  # uv 패키지 의존성
├── Dockerfile      # 컨테이너 설정
└── README.md       # 이 가이드
```

## 🔄 환경 전환 방법

동일한 코드베이스에서 .env 파일의 설정만으로 환경 전환:

1. **.env 수정**: `APP_ENV=dev` 변경
2. **Config 재생성**: `modern-ml-pipeline get-config`  
3. **연결 검증**: `modern-ml-pipeline system-check`
4. **바로 사용**: 동일한 recipes로 실행!

## 🐳 Docker 컨테이너화 (모든 환경 동일)

### 독립 실행 (내장 DB/Redis/MLflow)
```bash
# Docker Compose로 전체 스택 실행
docker-compose up -d

# 애플리케이션 로그 확인
docker-compose logs -f {PROJECT_NAME}

# 개별 서비스 상태 확인
docker-compose ps
```

### 외부 서비스 연동 (mmp-local-dev 등)
```bash
# .env 파일에서 외부 서비스 정보 설정
APP_ENV=dev
POSTGRES_HOST=external-db-host
REDIS_HOST=external-redis-host
MLFLOW_TRACKING_URI=http://external-mlflow:5000

# 애플리케이션만 실행
docker-compose up {PROJECT_NAME}
```

## 🚨 문제 해결

### Config 생성 실패 시
```bash
# .env 파일 확인
cat .env

# 템플릿 다시 복사  
cp .env.template .env
vi .env  # 올바른 값 입력
modern-ml-pipeline get-config
```

### System Check 실패 시
```bash
# 어떤 서비스가 실패했는지 확인
modern-ml-pipeline system-check

# 해당 서비스 설정 수정 후 재시도
vi .env  # 문제된 서비스 설정 수정
modern-ml-pipeline get-config  # config 재생성
modern-ml-pipeline system-check  # 재검증
```

## 📚 더 많은 정보

- [Modern ML Pipeline 문서](https://github.com/your-org/modern-ml-pipeline)
- [mmp-local-dev 개발환경](https://github.com/your-org/mmp-local-dev)