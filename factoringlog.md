# 📋 Recipe 구조 안정화 계획: Point-in-Time + 기존 구조 보존

## 🎯 **핵심 목표**

**Phase 1 Point-in-Time 정합성 추가 + 기존 시스템 완전 보존**
- `entity_schema`: Point-in-Time 전용 (Entity + Timestamp)
- `data_interface`: 기존 ML 설정 완전 보존 (treatment_column, class_weight, average 등)
- `hyperparameters`: Dictionary 형태 유지 (Optuna 호환)

## 🔍 **현재 상황 분석**

### **올바른 Recipe 구조 (목표)**
```yaml
# 완벽한 분리된 구조
model:
  # Phase 1 추가: Point-in-Time 정합성만
  loader:
    entity_schema:
      entity_columns: ["user_id", "product_id"]  # PK 정의
      timestamp_column: "event_timestamp"        # 시점 기준
      
  # 기존 보존: ML 작업별 세부 설정만  
  data_interface:
    task_type: "causal"
    target_column: "outcome"
    treatment_column: "treatment_group"  # Causal 필수
    class_weight: "balanced"             # Classification 필수
    average: "weighted"                  # 평가 필수
    
  # 기존 보존: Dictionary 형태 hyperparameters
  hyperparameters:
    C: {type: "float", low: 0.001, high: 100.0, log: true}
    penalty: {type: "categorical", choices: ["l1", "l2"]}
```

### **현재 문제점**
1. **YAML 구조 변질**: `yaml.dump()`로 인해 Dictionary → nested YAML로 변경됨
2. **ML 설정 누락**: Causal 모델의 `treatment_column` 등 핵심 설정 손실
3. **과도한 구조 변경**: 기존 호환성 파괴 위험

---

## 🛡️ **안정성 우선 개선 전략**

### **전략 1: 최소 침습적 변경 (Minimal Invasive Change)**

#### **1.1 Recipe 구조 복원 원칙**
```yaml
우선순위 1: 기존 data_interface 완전 보존
우선순위 2: entity_schema 추가 (별도 영역)
우선순위 3: hyperparameters Dictionary 형태 복원
우선순위 4: 코드 참조 경로 최소 변경
```

#### **1.2 단계별 안전 진행**
```yaml
Step 1: Recipe 백업 생성 (전체 롤백 가능)
Step 2: 개별 Recipe 구조 검증 (1개씩 테스트)
Step 3: Hyperparameters Dictionary 복원
Step 4: ML 설정 누락 부분 복원
Step 5: 단계별 검증 (validate → train → inference)
```

---

## 📊 **전체 코드 영향 범위 분석**

### **영향받는 컴포넌트 분석**

#### **1. Recipe 파일들 (27개)**
```yaml
상태: 구조 변경 필요
영향도: HIGH
위험도: MEDIUM (백업으로 롤백 가능)

복원 대상:
- Causal (4개): treatment_column 필수 복원
- Classification (8개): class_weight, average 복원  
- All (27개): hyperparameters Dictionary 복원
```

#### **2. Pydantic Models (src/settings/models.py)**
```yaml
상태: 부분 수정 필요
영향도: HIGH  
위험도: LOW (타입 안전성 보장)

수정 내용:
- ModelConfigurationSettings: data_interface 필수 필드로
- EntitySchema: Point-in-Time 전용으로 명확화
- MLTaskSettings: 기존 호환성 100% 유지
```

#### **3. Pipeline 코드 (6개 파일)**
```yaml
상태: 참조 경로 정리 필요
영향도: MEDIUM
위험도: LOW (컴파일 타임 체크 가능)

수정 파일:
- trainer.py: ML 설정은 model.data_interface에서
- factory.py: evaluator 생성 시 올바른 인터페이스 전달
- evaluator.py: getattr() 하드코딩 제거
```

#### **4. Test 파일들 (3개)**
```yaml
상태: 새 구조 반영 필요
영향도: LOW
위험도: LOW (기능 변경 없음)
```

### **위험도 평가**

#### **높은 위험 (즉시 대응 필요)**
```yaml
1. Causal 모델 학습 실패
   원인: treatment_column 누락
   대응: 즉시 복원 + 검증

2. Classification 불균형 데이터 처리 실패  
   원인: class_weight 누락
   대응: 즉시 복원 + 검증
```

#### **중간 위험 (주의 깊게 처리)**
```yaml
1. Hyperparameters 파싱 오류
   원인: Dictionary → nested YAML 변경
   대응: 안전한 복원 스크립트 작성

2. 참조 경로 불일치
   원인: loader.entity_schema vs model.data_interface 혼재
   대응: 명확한 분리 + 일관성 유지
```

#### **낮은 위험 (점진적 처리)**
```yaml
1. Test 케이스 실패
   원인: 새 구조 미반영
   대응: 새 구조에 맞춰 업데이트
```

---

## 🔧 **구체적 개선 방식**

### **방식 1: 점진적 복원 (권장)**

#### **1단계: 백업 및 안전성 확보 (5분)**
```bash
# 완전한 롤백 가능성 확보
git add . && git commit -m "구조 변경 전 백업"
cp -r recipes/ recipes_backup_safe/
```

#### **2단계: Recipe 구조 안전 복원 (15분)**
```python
# 정확한 Dictionary 형태 보존하는 스크립트
def restore_recipe_safely(file_path):
    # 1. 기존 hyperparameters 형태 보존
    # 2. entity_schema + data_interface 분리  
    # 3. ML 설정 누락 부분 복원
    # 4. YAML 구조 변경 없이 작업
```

#### **3단계: Pydantic 모델 정리 (10분)**
```python
# 명확한 책임 분리
class EntitySchema(BaseModel):
    """Point-in-Time 정합성만"""
    entity_columns: List[str]
    timestamp_column: str

class MLTaskSettings(BaseModel):  
    """ML 작업 설정만"""
    task_type: str
    target_column: str
    treatment_column: Optional[str] = None  # Causal용
    class_weight: Optional[str] = None      # Classification용
    average: Optional[str] = "weighted"     # 평가용
```

#### **4단계: 참조 경로 일관성 (10분)**
```python
# 명확한 분리된 접근
entity_info = settings.recipe.model.loader.entity_schema    # Point-in-Time
ml_config = settings.recipe.model.data_interface            # ML 설정
```

#### **5단계: 단계별 검증 (10분)**
```bash
# 각 단계마다 안전성 확인
APP_ENV=local uv run python main.py validate --recipe-file e2e_classification_test
APP_ENV=local uv run python main.py validate --recipe-file s_learner  # Causal 검증
```

### **방식 2: 전체 롤백 후 재작업 (대안)**

#### **완전 롤백 시나리오**
```bash
# 모든 변경사항 되돌리기
git reset --hard be0b35f
# 처음부터 올바른 방식으로 접근
```

---

## ⏰ **실행 계획 및 일정**

### **권장 접근: 점진적 복원 (총 50분)**

| 단계 | 작업 | 시간 | 위험도 | 검증 방법 |
|:-----|:-----|:-----|:-------|:----------|
| 1 | 백업 및 안전성 확보 | 5분 | 없음 | Git 상태 확인 |
| 2 | Recipe 구조 안전 복원 | 15분 | 중간 | 개별 Recipe 검증 |
| 3 | Pydantic 모델 정리 | 10분 | 낮음 | Type 체크 통과 |
| 4 | 참조 경로 일관성 | 10분 | 낮음 | 컴파일 성공 |
| 5 | 단계별 검증 | 10분 | 없음 | 모든 시나리오 통과 |

### **성공 기준**
```yaml
✅ Recipe 구조: entity_schema + data_interface 분리
✅ Hyperparameters: Dictionary 형태 복원  
✅ ML 설정: 누락 없이 완전 복원
✅ 파이프라인: E2E 정상 동작
✅ 하위 호환성: 기존 사용법 100% 지원
```

---

## 🎯 **최종 목표 달성 상태**

### **완벽한 Recipe 예시 (Causal)**
```yaml
name: "s_learner"
model:
  class_path: "causalml.inference.meta.SLearner"
  
  # Phase 1: Point-in-Time 정합성
  loader:
    entity_schema:
      entity_columns: ["user_id", "product_id"]
      timestamp_column: "event_timestamp"
      
  # 기존: ML 작업 설정 완전 보존
  data_interface:
    task_type: "causal"
    target_column: "outcome"
    treatment_column: "treatment_group"  # ✅ 복원
    treatment_value: "treatment"         # ✅ 복원
    
  # 기존: Dictionary 형태 유지
  hyperparameters:
    n_estimators: {type: "int", low: 50, high: 500}     # ✅ Dictionary
    max_depth: {type: "int", low: 3, high: 20}          # ✅ Dictionary
```

**이 계획을 통해 Phase 1의 혁신적 Point-in-Time 안전성을 추가하면서, 기존 시스템의 모든 가치를 완벽하게 보존합니다.** 🚀
