-- E2E 테스트용 세션 Spine 생성 SQL
-- 이 SQL은 ML 모델의 예측 대상 엔티티(user_id, product_id, session_id)와 
-- 타임스탬프를 정의하는 Spine 데이터를 생성합니다.

SELECT 
    session_id,
    user_id, 
    product_id,
    event_timestamp,
    outcome  -- 타겟 변수 (분류 문제: 0=구매안함, 1=구매함)
FROM sessions 
WHERE event_timestamp >= '2024-07-01'
    AND event_timestamp <= '2024-07-05'
ORDER BY event_timestamp DESC; 