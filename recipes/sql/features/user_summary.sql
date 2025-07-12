-- recipes/sql/features/user_summary.sql
-- Loader가 생성한 대상 집단(임시 테이블)에 대해,
-- 각 사용자의 특정 시점(Point-in-Time) 피처를 증강하는 Augmenter용 쿼리 템플릿.

-- 1. 로더가 업로드한 추론 대상 임시 테이블을 CTE로 정의
WITH targets AS (
  SELECT
    member_id,
    event_timestamp
  FROM
    `{{ temp_target_table_id }}`
)
-- 2. 피처 스토어와 동적으로 조인하여 시점 일관성 보장
SELECT
  t.member_id,

  -- 인구통계학적 피처
  fs.gender,
  fs.age_group,

  -- 행동 피처 (캠페인 노출 시점 기준)
  fs.days_since_last_visit,
  fs.lifetime_purchase_count,
  fs.avg_purchase_amount_90d,
  fs.avg_session_duration_30d

FROM
  targets t
LEFT JOIN
  -- 피처 스토어 테이블 (날짜별로 파티셔닝되어 있다고 가정)
  `{{ gcp_project_id }}.feature_store.user_daily_summary` fs
  -- 각 사용자의 이벤트 타임스탬프보다 이전이면서 가장 가까운 날짜의 피처를 가져옴
  ON t.member_id = fs.member_id AND DATE(t.event_timestamp) >= fs.snapshot_date
-- 이 테크닉은 BigQuery에서 Point-in-Time Join을 수행하는 매우 효율적인 방법입니다.
QUALIFY ROW_NUMBER() OVER(PARTITION BY t.member_id ORDER BY fs.snapshot_date DESC) = 1
