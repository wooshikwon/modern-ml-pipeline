-- E2E 테스트용 세션 Spine 생성 SQL
-- 이 SQL은 ML 모델의 예측 대상 엔티티(user_id, product_id, session_id)와 
-- 타임스탬프, 그리고 기본 피처들을 포함하는 완전한 데이터를 생성합니다.

SELECT 
    s.session_id,
    s.user_id, 
    s.product_id,
    s.event_timestamp,
    -- 사용자 피처들 (users 테이블에서 조인)
    u.age as user_age,
    u.country_code as user_country,
    u.ltv as user_ltv,
    u.total_purchase_count as user_total_purchases,
    -- 상품 피처들 (products 테이블에서 조인)  
    p.price as product_price,
    p.category as product_category,
    p.brand as product_brand,
    -- 타겟 변수
    s.outcome  -- 분류 문제: 0=구매안함, 1=구매함
FROM sessions s
LEFT JOIN users u ON s.user_id = u.user_id
LEFT JOIN products p ON s.product_id = p.product_id
WHERE s.event_timestamp >= '2024-07-01'
    AND s.event_timestamp <= '2024-07-05'
ORDER BY s.event_timestamp DESC; 