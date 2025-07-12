-- recipes/sql/loaders/user_features.sql
-- 특정 캠페인의 대상자(실험군/대조군)와 결과를 정의하는 로더 쿼리.
-- 이 쿼리는 모델 학습 및 배치 추론의 기본 데이터셋을 생성합니다.

SELECT
    -- 1. 기본 식별자
    t.member_id,

    -- 2. 처치(Treatment) 정보
    t.campaign_group AS grp, -- 'treatment' 또는 'control'

    -- 3. 결과(Outcome) 정보
    -- 캠페인 노출 후 14일 이내 첫 구매까지 걸린 시간 (일). 미구매 시 null.
    p.days_to_first_purchase AS outcome,

    -- 4. 조인 및 필터링을 위한 기준 시점
    t.exposure_date AS event_timestamp

FROM
    `{{ gcp_project_id }}.marketing_data.campaign_targets` AS t
LEFT JOIN
    `{{ gcp_project_id }}.marketing_data.first_purchase_after_campaign` AS p
    ON t.member_id = p.member_id AND t.campaign_id = p.campaign_id
WHERE
    -- Jinja 템플릿: 파이프라인 실행 시 `context_params`로 캠페인 ID를 주입받음
    t.campaign_id = '{{ campaign_id }}'
