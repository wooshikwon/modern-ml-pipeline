# 배포 및 운영 가이드

MMP 프로젝트의 배포 흐름과 운영 방법을 안내합니다.

---

## 1. 배포 흐름 개요

```text
mmp init → 로컬 실험 → docker build → CI/CD로 GCR/ECR 푸시 → k8s에서 실행
```

| 단계 | 담당 | 도구 |
|------|------|------|
| 프로젝트 생성, 실험 | ML 엔지니어 | MMP CLI |
| 이미지 빌드/푸시 | ML 엔지니어 + CI/CD | Docker, GitHub Actions 등 |
| k8s 배포, 운영 | 플랫폼팀/SRE | k8s 매니페스트, ConfigMap |

> **MMP의 범위**: 프로젝트 생성 → 실험 → 이미지 빌드 → GCR/ECR 푸시
>
> **MMP 범위 외**: CI/CD 파이프라인, k8s 매니페스트, ConfigMap

---

## 2. Docker 이미지 빌드

### 기본 빌드

```bash
docker build -t my-model:latest .
```

### Extras 선택

Dockerfile의 `RUN pip install` 라인에서 필요한 extras를 조합합니다:

```dockerfile
# 클라우드 배포 + Gradient Boosting (권장)
RUN pip install ".[ml-extras,cloud-extras]"

# 클라우드 배포 + 딥러닝
RUN pip install ".[cloud-extras,torch-extras]"
```

| 시나리오 | extras |
|----------|--------|
| 클라우드 + XGBoost/LightGBM | `ml-extras,cloud-extras` |
| 클라우드 + 딥러닝 | `cloud-extras,torch-extras` |
| 전체 기능 | `all` |

---

## 3. CI/CD로 레지스트리 푸시

CI/CD 파이프라인(GitHub Actions, GitLab CI 등)은 각 조직에서 구성합니다. 아래는 참고 예시입니다.

### GitHub Actions 예시

```yaml
# .github/workflows/docker-publish.yml
name: Build and Push
on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Login to GCR
        run: echo "${{ secrets.GCP_SA_KEY }}" | docker login -u _json_key --password-stdin gcr.io

      - name: Build and Push
        run: |
          docker build -t gcr.io/${{ vars.PROJECT_ID }}/mmp:${{ github.ref_name }} .
          docker push gcr.io/${{ vars.PROJECT_ID }}/mmp:${{ github.ref_name }}
```

### 수동 푸시

```bash
# GCR
docker build -t gcr.io/my-project/mmp:v1 .
docker push gcr.io/my-project/mmp:v1

# ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t <account>.dkr.ecr.<region>.amazonaws.com/mmp:v1 .
docker push <account>.dkr.ecr.<region>.amazonaws.com/mmp:v1
```

---

## 4. 운영: 설정 변경

### 이미지 내 기본 설정 사용

이미지 빌드 시 `configs/`, `recipes/`가 포함됩니다. 변경 없이 그대로 사용 가능합니다.

```bash
# 이미지 내 설정으로 실행
mmp serve-api --run-id abc123 -c configs/prod.yaml
```

### ConfigMap으로 설정 덮어쓰기

이미지 재빌드 없이 설정만 변경하고 싶을 때, k8s ConfigMap을 마운트합니다.

```yaml
# k8s 매니페스트 (플랫폼팀 관리)
apiVersion: v1
kind: ConfigMap
metadata:
  name: mmp-config
data:
  prod.yaml: |
    environment:
      name: production
    mlflow:
      tracking_uri: http://mlflow:5000
---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: mmp
        image: gcr.io/my-project/mmp:v1
        command: ["mmp", "serve-api", "--run-id", "abc123", "-c", "/app/configs/prod.yaml"]
        volumeMounts:
        - name: config
          mountPath: /app/configs    # 이미지 내 configs/ 덮어씀
      volumes:
      - name: config
        configMap:
          name: mmp-config
```

### 환경변수로 동적 값 주입

`MODEL_RUN_ID` 등 자주 변경되는 값은 환경변수로 주입합니다.

```yaml
env:
- name: MODEL_RUN_ID
  valueFrom:
    configMapKeyRef:
      name: mmp-env
      key: run_id
```

---

## 5. 단일 이미지 실행 명령어

빌드된 이미지는 command만 다르게 지정하여 학습/추론/서빙 모두 가능합니다.

| 용도 | command |
|------|---------|
| API 서빙 | `mmp serve-api --run-id <run_id> -c /app/configs/prod.yaml` |
| 배치 추론 | `mmp batch-inference --run-id <run_id> -d gs://bucket/data.csv` |
| 학습 | `mmp train -r /app/recipes/model.yaml -d gs://bucket/train.csv` |

---

## 6. Health Check

API 서빙 시 k8s probe에 사용합니다.

| 엔드포인트 | 용도 | 응답 |
|------------|------|------|
| `GET /health` | Liveness | 항상 200 |
| `GET /ready` | Readiness | 모델 로드 완료 시 200 |

---

## 7. 환경변수 레퍼런스

| 환경변수 | 용도 | 기본값 |
|----------|------|--------|
| `MODEL_RUN_ID` | 서빙/추론할 모델 ID | - |
| `CONFIG_PATH` | Config 파일 경로 | `configs/production.yaml` |
| `RECIPE_PATH` | Recipe 파일 경로 | `recipes/model.yaml` |
| `MLFLOW_TRACKING_URI` | MLflow 서버 주소 | `./mlruns` |

---

## 참고

- [API 서빙 가이드](./API_SERVING_GUIDE.md): REST API 엔드포인트 상세
- [CLI 레퍼런스](./CLI_REFERENCE.md): 명령어 상세 옵션
- [환경 설정 가이드](./ENVIRONMENT_SETUP.md): Config 파일 작성법
