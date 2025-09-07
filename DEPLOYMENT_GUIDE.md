# Modern ML Pipeline - Deployment Guide

이 가이드는 Modern ML Pipeline을 CLI 라이브러리로 배포하는 표준적인 방법들을 설명합니다.

## 배포 준비

### 1. pyproject.toml 수정

배포 전 프로젝트 메타데이터를 실제 정보로 업데이트하세요:

```toml
[project]
name = "modern-ml-pipeline"
version = "0.2.0"  # 버전 업데이트
authors = [
    { name = wooshikwon", email = "wonwooshik@gmail.com" }  # 실제 정보로 변경
]
```

### 2. README.md 작성

PyPI에서 프로젝트 설명으로 사용될 README.md를 작성하세요:
- 프로젝트 소개
- 설치 방법
- 기본 사용법
- 예제 코드

## PyPI 배포 (표준 방법)

### 1. PyPI 계정 준비

```bash
# PyPI 계정 생성 (https://pypi.org/account/register/)
# API 토큰 생성 (https://pypi.org/manage/account/token/)
```

### 2. 배포 도구 설치

```bash
pip install build twine
```

### 3. 패키지 빌드

```bash
# 프로젝트 루트에서 실행
python -m build
```

빌드 후 생성되는 파일들:
- `dist/modern_ml_pipeline-0.2.0-py3-none-any.whl`
- `dist/modern-ml-pipeline-0.2.0.tar.gz`

### 4. TestPyPI에서 테스트 (권장)

```bash
# TestPyPI에 업로드
twine upload --repository testpypi dist/*

# TestPyPI에서 설치 테스트
pip install --index-url https://test.pypi.org/simple/ modern-ml-pipeline
```

### 5. 본 PyPI에 배포

```bash
# 본 PyPI에 업로드
twine upload dist/*
```

### 6. 설치 확인

```bash
pip install modern-ml-pipeline

# CLI 명령어 테스트
ml-pipeline --help
mmp --help
modern-ml-pipeline --help
```

## GitHub Actions 자동화

### 1. GitHub Secrets 설정

Repository Settings > Secrets and variables > Actions에서 설정:
- `PYPI_API_TOKEN`: PyPI API 토큰

### 2. Workflow 파일 생성

`.github/workflows/release.yml` 파일 생성:

```yaml
name: Release to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

### 3. 릴리스 프로세스

1. 버전 업데이트 (`pyproject.toml`에서 version 수정)
2. GitHub에서 새 릴리스 생성
3. 자동으로 PyPI에 배포됨

## 배포 전 체크리스트

- [ ] `pyproject.toml`의 메타데이터 확인 (name, version, authors, description)
- [ ] `README.md` 작성 완료
- [ ] 모든 테스트 통과 확인 (`pytest`)
- [ ] 린트 체크 통과 (`ruff check`)
- [ ] 타입 체크 통과 (`mypy`)
- [ ] CLI 명령어 동작 확인
- [ ] TestPyPI에서 테스트 완료
- [ ] GitHub Secrets 설정 완료 (자동화 사용 시)

## 버전 관리

### Semantic Versioning 사용

- **MAJOR** (1.0.0): 호환성을 깨는 변경사항
- **MINOR** (0.1.0): 새로운 기능 추가 (하위 호환)
- **PATCH** (0.0.1): 버그 수정

### 버전 업데이트 방법

```bash
# pyproject.toml에서 version 수정
version = "0.3.0"

# 새로운 태그 생성
git tag v0.3.0
git push origin v0.3.0
```

## 배포 후 확인사항

1. **PyPI 페이지 확인**: https://pypi.org/project/modern-ml-pipeline/
2. **설치 테스트**:
   ```bash
   pip install modern-ml-pipeline
   ml-pipeline --version
   ```
3. **기본 기능 테스트**:
   ```bash
   ml-pipeline init --help
   ml-pipeline train --help
   ```

## 문제 해결

### 빌드 오류
- `pyproject.toml` 구문 확인
- 필수 파일 존재 확인 (`README.md`, `LICENSE`)
- Python 버전 호환성 확인

### 업로드 오류
- API 토큰 유효성 확인
- 패키지명 중복 확인
- 버전 중복 확인 (PyPI는 같은 버전 재업로드 불가)

### 설치 후 CLI 명령어 인식 안됨
- `[project.scripts]` 설정 확인
- 가상환경 PATH 확인
- pip install 시 `--force-reinstall` 옵션 사용