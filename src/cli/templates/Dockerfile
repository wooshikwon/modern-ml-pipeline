FROM python:3.11-slim as builder

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY pyproject.toml ./
RUN uv sync --frozen --no-cache --no-dev

FROM python:3.11-slim as runtime

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 가상환경 복사
COPY --from=builder /app/.venv /app/.venv

# PATH 설정
ENV PATH="/app/.venv/bin:$PATH"

# 작업 디렉토리 설정  
WORKDIR /app

# 애플리케이션 파일 복사
COPY . .

# 기본 포트 (환경변수로 오버라이드 가능)
EXPOSE 8000

# 헬스체크 (모든 환경 동일)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# 기본 실행 명령 (APP_ENV에 따라 자동 환경 전환)
CMD ["modern-ml-pipeline", "api-serving"]