# -----------------------------------------------------------------------------
# Dockerfile for Modern ML Pipeline
# -----------------------------------------------------------------------------
# Multi-stage build 전략을 사용하여, 최종 목적(학습, 서빙)에 따라
# 최적화된 이미지를 생성합니다.
# -----------------------------------------------------------------------------

# --- Stage 1: `base` ---
# 목적: Python과 Hatch, 그리고 공통 의존성을 설치하는 기본 토대 이미지.
FROM python:3.10-slim as base

# --- 환경 변수 설정 ---
ENV PYTHONUNBUFFERED=1 
    PIP_NO_CACHE_DIR=off 
    HATCH_ENV=dev

# 시스템에 Hatch 설치
RUN pip install "hatch==1.11.1"

# 작업 디렉토리 설정
WORKDIR /app

# --- 의존성 설치 (Docker 캐시 최적화) ---
# 소스 코드를 복사하기 전에, 의존성 정의 파일만 먼저 복사합니다.
COPY pyproject.toml hatch.toml ./

# 기본 의존성 설치 (개발용 제외)
RUN hatch dep sync


# --- Stage 2: `train` ---
# 목적: 모델 학습에 필요한 모든 코드와 의존성을 포함하는 이미지.
FROM base as train

# 개발용 의존성까지 모두 설치
RUN hatch dep sync dev

# 프로젝트의 모든 소스 코드를 이미지의 작업 디렉토리(/app)로 복사
COPY . .

# 빌드 시 모델 이름을 인자로 받을 수 있도록 설정
ARG MODEL_NAME=xgboost_x_learner

# 이 이미지를 실행할 때 기본적으로 실행될 명령어
# `main.py`의 `train` 커맨드를 실행
CMD ["hatch", "run", "python", "main.py", "train", "--model-name", "${MODEL_NAME}"]


# --- Stage 3: `serve` ---
# 목적: API 서빙에 필요한 최소한의 파일과 의존성만 포함하는 경량 이미지.
FROM base as serve

# API 서빙에 필요한 파일 및 디렉토리만 선별하여 복사
COPY . .

# 빌드 시 모델 이름을 인자로 받을 수 있도록 설정
ARG MODEL_NAME=xgboost_x_learner

# 이 이미지를 실행할 때 기본적으로 실행될 명령어
# `main.py`의 `serve-api` 커맨드를 실행
CMD ["hatch", "run", "python", "main.py", "serve-api", "--model-name", "${MODEL_NAME}"]
