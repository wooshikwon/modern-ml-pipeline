-- src/sql/user_features.sql
-- 모델 학습 및 추론에 사용할 기본 데이터를 생성하는 SQL 쿼리
-- Jinja 템플릿 구문을 사용하여 파라미터를 동적으로 주입할 수 있습니다.
-- 예: WHERE event_date BETWEEN '{{ start_date }}' AND '{{ end_date }}'

WITH rsvn_features AS (
    -- 최근 90일간의 예약 데이터를 기반으로 피처 생성
    SELECT
        member_id,
        COUNT(rsvn_id) AS rsvn_90_count,
        AVG(price) AS rsvn_90_avg_price,
        DATE_DIFF(CURRENT_DATE(), MAX(rsvn_date), DAY) AS last_rsvn_date_elapsed
    FROM
        `your-gcp-project-id.raw_data.reservations`
    WHERE
        rsvn_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    GROUP BY 1
),
campaign_data AS (
    -- 특정 캠페인 기간의 대상자 및 결과 데이터
    SELECT
        member_id,
        campaign_group AS grp, -- 'treatment' 또는 'control'
        CASE WHEN purchase_within_7_days = 1 THEN 1 ELSE 0 END AS outcome
    FROM
        `your-gcp-project-id.campaign_data.uplift_campaign_2024_q1`
)
-- 최종 학습 데이터셋 생성
SELECT
    c.member_id,
    m.member_gender,
    m.member_age,
    COALESCE(r.rsvn_90_count, 0) AS rsvn_30_count, -- 컬럼명 오타 ���정 제안 (rsvn_90_count -> rsvn_30_count)
    COALESCE(r.rsvn_90_avg_price, 0) AS rsvn_90_avg_price,
    r.last_rsvn_date_elapsed,
    c.grp,
    c.outcome
FROM
    campaign_data c
LEFT JOIN
    `your-gcp-project-id.user_data.member_master` m ON c.member_id = m.member_id
LEFT JOIN
    rsvn_features r ON c.member_id = r.member_id
WHERE
    m.is_active = 1 -- 활성 사용자만 대상으로 함
;