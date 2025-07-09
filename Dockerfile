# -----------------------------------------------------------------------------
# Dockerfile for ML-PIPELINE
# -----------------------------------------------------------------------------
# 이 Dockerfile은 Multi-stage build 전략을 사용하여,
# 최종 목적(학습, 서빙)에 따라 최적화된 이미지를 생성합니다.
# -----------------------------------------------------------------------------


# --- Stage 1: `base` ---
# 목적: Python과 Poetry, 그리고 공통 의존성을 설치하는 기본 토대 이미지.
#      이 이미지를 기반으로 `train`과 `serve` 이미지가 만들어집니다.
FROM python:3.10-slim as base

# --- 환경 변수 설정 ---
# PYTHONUNBUFFERED=1: Python의 출력이 버퍼링 없이 즉시 터미널에 표시되도록 합니다. (Docker 로그 확인에 유용)
# POETRY_NO_INTERACTION=1: Poetry가 사용자에게 질문(예: '설치하시겠습니까?')을 하지 않고 자동으로 진행하도록 합니다.
# POETRY_VIRTUALENVS_CREATE=false: Poetry가 프로젝트 내에 .venv 가상 환경을 만들지 않도록 합니다.
#                                  Docker 이미지 자체의 Python 환경을 사용하기 때문에 이 설정이 권장됩니다.
ENV PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# 시스템에 Poetry 설치
RUN pip install "poetry==1.8.2"

# 작업 디렉토리 설정
WORKDIR /app

# --- 의존성 설치 (Docker 캐시 최적화) ---
# 소스 코드를 전부 복사하기 전에, 의존성 정의 파일만 먼저 복사합니다.
# 이렇게 하면, 소스 코드가 변경되어도 의존성이 변경되지 않았다면,
# Docker는 이 레이어의 캐시를 재사용하여 빌드 시간을 크게 단축시킵니다.
COPY pyproject.toml poetry.lock ./

# --no-root: 프로젝트 자체(ml-pipeline)는 설치하지 않습니다. (소스 코드는 아래에서 COPY 예정)
# --no-dev: 서빙에 불필요한 개발용 의존성(예: pytest)은 기본적으로 제외합니다.
#           (이 단계에서는 serve 이미지의 기반이 되므로, 최소한의 의존성만 설치)
RUN poetry install --no-root --no-dev


# --- Stage 2: `train` ---
# 목적: 모델 학습에 필요한 모든 코드와 의존성을 포함하는 이미지.
FROM base as train

# `base` 단계에서 설치하지 않았던 개발용 의존성까지 모두 설치합니다.
# (학습 코드에는 테스트나 데이터 분석 관련 라이브러리가 필요할 수 있음)
RUN poetry install --no-root

# 프로젝트의 모든 소스 코드를 이미지의 작업 디렉토리(/app)로 복사합니다.
COPY . .

# 이 이미지를 `docker run`으로 실행할 때 기본적으로 실행될 명령어입니다.
# `main.py`의 `train` 커맨드를 실행하여 학습 파이프라인을 시작합니다.
CMD ["poetry", "run", "python", "main.py", "train"]


# --- Stage 3: `serve` ---
# 목적: API 서빙에 필요한 최소한의 파일과 의존성만 포함하는 경량 이미지.
#      `base` 이미지를 기반으로 하여, 불필요한 학습용 의존성이나 데이터 없이 가볍게 만듭니다.
FROM base as serve

# API 서빙에 필요한 파일 및 디렉토리만 선별하여 복사합니다.
# - `config/`: API 서버 실행에 필요한 설정 파일
# - `serving/`: FastAPI 앱(`api.py`, `schemas.py`)
# - `src/`: 모델 예측에 필요한 핵심 로직(Transformer, Model 등)
# - `main.py`: `serve-api` 커맨드를 실행하기 위한 진입점
COPY config ./config
COPY serving ./serving
COPY src ./src
COPY main.py ./

# 이 이미지를 `docker run`으로 실행할 때 기본적으로 실행될 명령어입니다.
# `main.py`의 `serve-api` 커맨드를 실행하여 FastAPI 서버를 시작합니다.
CMD ["poetry", "run", "python", "main.py", "serve-api"]