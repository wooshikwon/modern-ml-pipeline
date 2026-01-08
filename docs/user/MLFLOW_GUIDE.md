# MLflow 실험 추적 가이드

MMP는 MLflow와 연동하여 모든 실험 결과를 자동으로 기록합니다. 이 가이드에서는 MLflow 설정 방법과 UI 사용법을 설명합니다.

---

## 1. MLflow 개요

### 자동 기록 항목

학습 실행 시 다음 항목이 자동으로 MLflow에 기록됩니다:

| 항목 | 설명 |
|------|------|
| 파라미터 | 모델 하이퍼파라미터, 전처리 설정 |
| 메트릭 | 학습/검증 성능 지표 (accuracy, f1, rmse 등) |
| 아티팩트 | 학습된 모델, 전처리기, SHAP 분석 결과 |
| 태그 | Task 유형, 모델 클래스, 실행 환경 정보 |

---

## 2. Config 설정

### 2-1. 로컬 파일 저장 (기본)

별도 서버 없이 로컬 디렉토리에 실험 결과를 저장합니다.

```yaml
# configs/dev.yaml
mlflow:
  tracking_uri: ./mlruns           # 프로젝트 내 mlruns/ 디렉토리
  experiment_name: my-experiment
```

> `mmp get-config`로 Config 생성 시 기본값으로 설정됩니다.

---

### 2-2. 원격 MLflow 서버

팀에서 운영하는 MLflow 서버가 있다면 해당 주소를 지정합니다.

```yaml
# configs/prod.yaml
mlflow:
  tracking_uri: http://mlflow.company.com:5000
  experiment_name: production-models
```

**인증이 필요한 경우** `.env` 파일에 설정:

```bash
# .env
MLFLOW_TRACKING_USERNAME=your-username
MLFLOW_TRACKING_PASSWORD=your-password
```

---

## 3. MLflow UI 실행

실험 결과를 웹 브라우저에서 시각적으로 비교하고 분석할 수 있습니다.

### 3-1. 직접 실행 (권장)

MMP 설치 시 MLflow가 함께 설치됩니다. 프로젝트 루트에서 실행:

```bash
mlflow ui --port 5000

# 백그라운드 실행
mlflow ui --port 5000 &
```

브라우저에서 `http://localhost:5000` 접속

---

### 3-2. Docker Compose로 실행

격리된 환경에서 UI를 실행합니다.

```bash
# MLflow UI 시작
docker-compose --profile mlflow up mlflow

# 백그라운드 실행
docker-compose --profile mlflow up -d mlflow

# 종료
docker-compose --profile mlflow down
```

브라우저에서 `http://localhost:5000` 접속

---

### 3-3. 원격 서버 접속

팀 MLflow 서버가 있다면 해당 URL로 직접 접속:

```
http://mlflow.company.com:5000
```

---

## 4. UI 활용

### 실험 비교

1. MLflow UI 접속
2. 좌측 Experiments 목록에서 실험 선택
3. 비교할 run들을 체크박스로 선택
4. "Compare" 버튼 클릭

### 주요 기능

| 기능 | 설명 |
|------|------|
| **Runs 테이블** | 모든 실험 실행 목록, 메트릭/파라미터 정렬 |
| **Compare** | 선택한 run들의 메트릭 차트 비교 |
| **Artifacts** | 저장된 모델, 전처리기, 로그 파일 다운로드 |
| **Charts** | 메트릭 변화 시각화 |

---

## 5. 실험 워크플로우 예시

### Step 1: 여러 Recipe로 실험

```bash
# XGBoost 실험
mmp train -c configs/dev.yaml -r recipes/xgb-baseline.yaml -d data/train.csv

# LightGBM 실험
mmp train -c configs/dev.yaml -r recipes/lgbm-tuned.yaml -d data/train.csv

# RandomForest 실험
mmp train -c configs/dev.yaml -r recipes/rf-experiment.yaml -d data/train.csv
```

### Step 2: MLflow UI에서 비교

```bash
mlflow ui --port 5000
```

브라우저에서 실험 결과를 비교하고 최적 모델의 `run_id` 확인

### Step 3: 최적 모델로 서빙

```bash
mmp serve-api -c configs/dev.yaml --run-id <best_run_id>
```

---

## 6. 문제 해결

### mlruns 디렉토리가 없음

학습을 한 번도 실행하지 않으면 `mlruns/` 디렉토리가 생성되지 않습니다. 학습 실행 후 UI를 시작하세요.

### 원격 서버 연결 실패

```bash
# 연결 테스트
curl http://mlflow-server:5000/health

# 환경변수 확인
echo $MLFLOW_TRACKING_URI
```

### Docker에서 mlruns 마운트 문제

```bash
# mlruns 디렉토리 권한 확인
ls -la mlruns/

# 필요시 권한 수정
chmod -R 755 mlruns/
```

---

## 참고

- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [설정 스키마](./SETTINGS_SCHEMA.md): Config 파일 상세 옵션
- [CLI 레퍼런스](./CLI_REFERENCE.md): 명령어 옵션
