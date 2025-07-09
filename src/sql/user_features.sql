-- 사용자별 피처 집계 쿼리 예시
-- 실제 프로덕션 환경에서는 더 복잡하고 최적화된 쿼리가 필요합니다.

SELECT
    member_id,
    -- 최근 90일간 예약 건수
    COUNT(CASE WHEN rsvn_created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY) THEN rsvn_id END) AS rsvn_90_day_count,
    -- 최근 90일간 평균 예약 금액
    AVG(CASE WHEN rsvn_created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY) THEN rsvn_amount END) AS avg_rsvn_amount_90_day,
    -- 총 누적 예약 건수
    COUNT(rsvn_id) AS total_rsvn_count,
    -- 첫 예약일로부터 현재까지 기간 (일)
    DATE_DIFF(CURRENT_DATE(), MIN(DATE(rsvn_created_at)), DAY) AS days_since_first_rsvn
FROM
    `your_project.your_dataset.reservation_logs`
GROUP BY
    1
