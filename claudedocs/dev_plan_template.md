# {{ doc_title | default("개발 실행 계획: 안정성 확보 및 실용적 커버리지 달성") }}

## 🎯 핵심 현황 및 목표

### 현재 상태 ({{ as_of_date | default("YYYY-MM-DD") }} 검증)
| 지표 | 현재 (실측) | 단기목표 | 최종목표 | 우선순위 |
|------|-------------|----------|----------|----------|
{% if metrics %}
{% for m in metrics %}
| **{{ m.name }}** | **{{ m.current | default("N/A") }}** | **{{ m.short_term | default("N/A") }}** | **{{ m.final | default("N/A") }}** | {{ m.priority | default("P?") }} |
{% endfor %}
{% else %}
| **예시 지표** | **N/A** | **N/A** | **N/A** | P? |
{% endif %}

### 완료된 기반 작업 (간략)
{% if groundwork and groundwork|length > 0 %}
{% for item in groundwork %}
- ✅ {{ item }}
{% endfor %}
{% else %}
- ✅ (예시) 테스트 인프라 기본 구성 완료
- ✅ (예시) 코어 모듈 스켈레톤 반영
{% endif %}

### 핵심 블로커
{% if blockers and blockers|length > 0 %}
{% for b in blockers %}
- 🚨 **{{ b.title }}**: {{ b.detail | default("세부 설명") }}
{% endfor %}
{% else %}
- 🚨 **(예시) 실행 환경 불안정**: 패키지/권한/병렬화 충돌
{% endif %}

---

## 🔥 Phase 개요
**목표**: 단계적 안정화 → 통합 기능 완성 → 실용적 커버리지/성능 확보  
**총 기간(예상)**: {{ total_duration | default("N주") }}

{% if phases and phases|length > 0 %}
{% for ph in phases %}
### {{ loop.index }}. {{ ph.title | default("Phase 제목") }} ({{ ph.duration | default("기간") }})
**목표**: {{ ph.objective | default("Phase 핵심 목표") }}

{% if ph.summary %}
- **요약**: {{ ph.summary }}
{% endif %}

{% if ph.streams and ph.streams|length > 0 %}
#### 작업 스트림
{% for s in ph.streams %}
- **{{ s.name }}**: {{ s.goal | default("목표 설명") }}
  {% if s.tasks and s.tasks|length > 0 %}
  {% for t in s.tasks %}
  - [{{ "x" if t.done else " " }}] {{ t.title }}{% if t.note %} — {{ t.note }}{% endif %}
  {% endfor %}
  {% endif %}
{% endfor %}
{% endif %}

{% if ph.coverage_targets and ph.coverage_targets|length > 0 %}
#### 커버리지/성능 타깃
| 대상 | 현재 | 목표 | 비고 |
|------|------|------|------|
{% for ct in ph.coverage_targets %}
| {{ ct.target }} | {{ ct.current | default("N/A") }} | {{ ct.goal | default("N/A") }} | {{ ct.note | default("") }} |
{% endfor %}
{% endif %}

{% if ph.acceptance and ph.acceptance|length > 0 %}
#### 완료 기준 (Acceptance Criteria)
{% for a in ph.acceptance %}
- [ ] {{ a }}
{% endfor %}
{% else %}
- [ ] (예시) 핵심 테스트 0 실패
- [ ] (예시) 지정 모듈 커버리지 기준 달성
{% endif %}

---
{% endfor %}
{% else %}
### 1. (예시) 기반 안정화 (2-3주)
- [ ] 테스트 실행 안정성 확보
- [ ] 핵심 CLI/엔드포인트 최소 커버리지 달성
---
### 2. (예시) 통합 기능 완성 (3-4주)
- [ ] API/파이프라인 통합 테스트 통과
- [ ] 커버리지/성능 목표 수립 및 달성
---
{% endif %}

## ⚡ 실행 우선순위 매트릭스

| 작업 | 현재 상태 | 목표 | 우선순위 | 예상 소요 | 핵심 방법 |
|------|----------|------|----------|----------|-----------|
{% if priority_matrix and priority_matrix|length > 0 %}
{% for row in priority_matrix %}
| **{{ row.name }}** | {{ row.current | default("N/A") }} | {{ row.goal | default("N/A") }} | {{ row.priority | default("P?") }} | {{ row.duration | default("-") }} | {{ row.method | default("-") }} |
{% endfor %}
{% else %}
| **(예시) 테스트 인프라 안정화** | 불안정 | 안정 | P0 | 3일 | 권한/병렬화/경로 정리 |
{% endif %}

---

## 📊 성공 지표 및 마일스톤

### 현실적 목표 설정
- **현재 (실측)**: {{ headline_status.current | default("N/A") }}
- **중간 목표**: {{ headline_status.intermediate | default("N/A") }}
- **최종 목표**: {{ headline_status.final | default("N/A") }}

### 단계별 검증 기준
{% if acceptance_global and acceptance_global|length > 0 %}
{% for ac in acceptance_global %}
- [ ] {{ ac }}
{% endfor %}
{% else %}
- [ ] (예시) 주요 워크플로우 E2E 테스트 통과
- [ ] (예시) 커버리지/성능 지표 달성
{% endif %}

---

## ⚠️ 주요 리스크 및 완화 방안

{% if risks and risks|length > 0 %}
{% for r in risks %}
1. **{{ r.title }}**  
   - 영향도: {{ r.impact | default("중") }} / 가능성: {{ r.likelihood | default("중") }}  
   - 완화: {{ r.mitigation | default("대응 전략") }}
{% endfor %}
{% else %}
1. **(예시) 측정 불일치**  
   - 영향도: 중 / 가능성: 중  
   - 완화: 측정 도구/옵션 표준화, 주기적 재측정
{% endif %}

---

## 가이드(철학/원칙)

### ✅ 권장 패턴
```python
{{ patterns.correct | default("# 예시: 컨텍스트/실컴포넌트/성능 측정 활용") }}
````

### ❌ 안티패턴 (피해야 할 것)

```python
{{ patterns.anti | default("# 예시: 과도한 내부 Mock, 시간/환경 의존 네이밍 등") }}
```

---

## ✅ 실행 체크리스트

### 즉시 실행 (단기)

{% if checklists.short and checklists.short|length > 0 %}
{% for c in checklists.short %}

* [ ] {{ c }}
  {% endfor %}
  {% else %}
* [ ] (예시) CI에서 테스트 병렬화 옵션 점검
  {% endif %}

### 단기(2-3주)

{% if checklists.mid and checklists.mid|length > 0 %}
{% for c in checklists.mid %}

* [ ] {{ c }}
  {% endfor %}
  {% else %}
* [ ] (예시) 핵심 경로 커버리지 보강
  {% endif %}

### 완료 기준(현실적)

{% if checklists.done and checklists.done|length > 0 %}
{% for c in checklists.done %}

* [ ] {{ c }}
  {% endfor %}
  {% else %}
* [ ] (예시) 0 실패 + 목표 커버리지 도달
  {% endif %}

---

## 참고 자료

{% if references and references|length > 0 %}
{% for ref in references %}

* {{ ref }}
  {% endfor %}
  {% else %}
* (예시) 프로젝트 내부 문서 / 공식 레퍼런스
  {% endif %}

---

## 📋 현재 상황 및 다음 단계

**현재 위치**: {{ now\_status | default("기반 완성 단계") }}
**다음 단계**: {{ next\_steps | default("핵심 기능 커버리지 확장 및 통합 검증") }}

---

*문서 버전: {{ doc\_version | default("0.1.0") }}*
*기준일: {{ as\_of\_date | default("YYYY-MM-DD") }}*
*작성 원칙: 현실 기반 계획, 실행 가능한 목표, 구체적 액션 아이템*