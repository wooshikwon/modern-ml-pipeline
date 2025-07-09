# README.md

ml-pipeline
├── recipe/                      # 여러 모델 설정을 관리하는 yaml 레시피 파일
│   ├── causal_forest.yaml       # 데이터 로더, 모델, 트레이너 등 핵심 로직 구현체
│   └── xgboost_x_learner.yaml   # 로거, GCP 클라이언트 등 공통 유틸리티
├── credentials/                 # GCP 인증 키 등 민감 정보를 저장  (local 개발 환경 용, Git ingored)
│   └── my-service-account.json  # 로거, GCP 클라이언트 등 공통 유틸리티
├── src/                         # 핵심 소스 코드
│   ├── core/                    # 데이터 로더, 모델, 트레이너 등 핵심 로직 구현체
│   │   ├── loader.py            # 데이터 로드. API serving 환경에선 외부 input으로 대체됨. 배치 추론/학습 환경에선 Data Lake에서 추론/학습하고자 하는 시점-대상 추출
│   │   ├── augmenter.py         # 로드된 데이터 혹은 Api input을 기반으로 매칭도는 feature sotre 데이터를 불러와 증강
│   │   ├── preprocessor.py      # 범주형 데이터 수치형 변환 및 standard scaler fit, transform, save, load 등
│   │   ├── factory.py           # 학습 시 trainer.py에서 호출하여 '모델명' 인자에 맞는 모델 인스턴스 생성. 학습 완료 후 model, augmenter, preprocessor를 pyfuncwapper로 래핑.
│   │   └── trainer.py           # 
│   ├── interface/               # 핵심 로직의 설계도 (추상 클래스)
│   ├── models/                  # sklearn, casualml 패키지가 지원하는 모델을 fit, transformer 함수 구현 후 래핑한 모델 클래스
│   ├── pipelines/               # 학습/추론 등 End-to-End 파이프라인 스크립트
│   ├── settings/                # 환경 변수와 config.yaml, recipe/ 파일을 받아서 dict 형식으로 객체 저장하는 get_settings 함수 구현
│   ├── sql/                     # 데이터 추출을 위한 SQL 쿼리
│   └── utils/                   # 로거, GCP 클라이언트 등 공통 유틸리티
├── serving/                     # 실시간 API 서빙 관련 코드 (FastAPI)
├── tests/                       # 단위/통합 테스트 코드
├── .env.example                 # 환경 변수 설정 예시 파일
├── Dockerfile                   # Docker 이미지 빌드를 위한 명세서
├── main.py                      # CLI 애플리케이션의 진입점
├── pyproject.toml               # 프로젝트 정의 및 의존성 관리 (Hatch)
└── README.md                    # 당신이 지금 보고 있는 파일