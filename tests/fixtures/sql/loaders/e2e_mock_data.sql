-- E2E 테스트용 Mock 데이터 로더
-- LIMIT 100이 포함되어 E2E Mock 시스템을 자동 활성화합니다

SELECT 
    user_id,
    product_id,
    event_timestamp,
    session_duration,
    page_views,
    outcome
FROM user_sessions_fact 
WHERE created_at >= '2024-01-01'
  AND event_timestamp >= '2024-01-01'
ORDER BY event_timestamp
LIMIT 100 