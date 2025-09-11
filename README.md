# Modern ML Pipeline

```
███╗   ███╗███╗   ███╗████████╗ 
████╗ ████║████╗ ████║██╔════██╗
██╔████╔██║██╔████╔██║████████╔╝
██║╚██╔╝██║██║╚██╔╝██║██╔═════╝ 
██║ ╚═╝ ██║██║ ╚═╝ ██║██║     
╚═╝     ╚═╝╚═╝     ╚═╝╚═╝     
```

**머신러닝 프로젝트를 쉽고 빠르게 만들어주는 CLI 도구입니다.**

설정 파일로 모델을 학습하고, API로 서빙하고, 배치로 예측할 수 있습니다.

## 설치

### 방법 1: Git URL로 설치 (간단)
```bash
# 최신 버전 설치
pip install git+https://github.com/wooshikwon/modern-ml-pipeline.git

# 특정 버전 설치
pip install git+https://github.com/wooshikwon/modern-ml-pipeline.git@v0.5.0
```

### 방법 2: GitHub Packages로 설치
```bash
pip install modern-ml-pipeline --index-url https://pypi.pkg.github.com/wooshikwon/simple/
```

> **참고**: GitHub Packages 방식은 GitHub Personal Access Token이 필요합니다. (packages:read 권한)

## 사용법

### 1. 새 프로젝트 시작하기

```bash
# 프로젝트 폴더와 기본 파일들 생성
mmp init

# 데이터베이스, MLflow 등 연결 설정 만들기
mmp get-config

# 연결 상태 확인하기
mmp system-check --config-path configs/my-config.yaml
```

### 2. 모델 만들기

```bash
# 모델 설정 파일 만들기 (어떤 모델을 어떻게 학습할지)
mmp get-recipe

# 모델 학습하기
mmp train --config-path configs/my-config.yaml --recipe-path recipes/my-recipe.yaml --data-path data/train.csv
```

### 3. 모델 사용하기

```bash
# 한 번에 여러 데이터 예측하기
mmp batch-inference --config-path configs/my-config.yaml --run-id <run_id> --data-path data/test.csv

# API 서버로 실시간 예측 서비스 시작하기
mmp serve-api --host 0.0.0.0 --port 8000
```

## 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `mmp init` | 새 프로젝트 폴더 만들기 |
| `mmp get-config` | 환경 설정 파일 만들기 |
| `mmp get-recipe` | 모델 설정 파일 만들기 |
| `mmp system-check` | 연결 상태 확인하기 |
| `mmp train` | 모델 학습하기 |
| `mmp batch-inference` | 배치로 예측하기 |
| `mmp serve-api` | API 서버 시작하기 |
| `mmp list adapters` | 사용 가능한 데이터 연결 방식 보기 |
| `mmp list models` | 사용 가능한 모델 보기 |

## 특징

- **간단한 설정**: YAML 파일로 모든 것을 설정
- **자동 실험 추적**: MLflow로 모든 학습 기록을 자동 저장
- **즉시 API 서빙**: 학습한 모델을 바로 REST API로 서빙
- **다양한 모델 지원**: scikit-learn부터 딥러닝 모델까지
- **배치 처리**: CSV, SQL 등 다양한 데이터로 대량 예측

## 도움말

```bash
# 전체 명령어 보기
mmp --help

# 특정 명령어 도움말
mmp train --help
```

---

**Version**: 0.5.0 | **License**: MIT