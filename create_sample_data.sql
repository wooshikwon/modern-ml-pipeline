-- E2E 테스트용 샘플 데이터 생성 스크립트
-- PostgreSQL (Offline Store용 데이터)

-- 1. 사용자 정보 테이블
DROP TABLE IF EXISTS users CASCADE;
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    age INTEGER,
    country_code VARCHAR(2),
    signup_date DATE,
    ltv DECIMAL(10,2),
    total_purchase_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 상품 정보 테이블  
DROP TABLE IF EXISTS products CASCADE;
CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    category VARCHAR(50),
    price DECIMAL(10,2),
    brand VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. 세션 정보 테이블 (ML 학습/추론의 Spine)
DROP TABLE IF EXISTS sessions CASCADE;
CREATE TABLE sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    product_id VARCHAR(50) REFERENCES products(product_id),
    event_timestamp TIMESTAMP,
    time_on_page_seconds INTEGER,
    click_count INTEGER,
    outcome INTEGER, -- 타겟 변수 (0: 구매안함, 1: 구매함)
    treatment_group VARCHAR(20), -- A/B 테스트용 (control, treatment)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 샘플 사용자 데이터 삽입
INSERT INTO users (user_id, age, country_code, signup_date, ltv, total_purchase_count) VALUES
('user_001', 25, 'US', '2024-01-15', 1250.50, 8),
('user_002', 34, 'KR', '2024-02-20', 890.25, 5),
('user_003', 28, 'JP', '2024-03-10', 2100.75, 12),
('user_004', 42, 'US', '2024-01-05', 3200.00, 18),
('user_005', 31, 'DE', '2024-04-22', 1580.30, 9),
('user_006', 27, 'KR', '2024-02-14', 750.80, 4),
('user_007', 38, 'US', '2024-03-30', 2950.60, 15),
('user_008', 29, 'JP', '2024-01-18', 1320.45, 7),
('user_009', 45, 'DE', '2024-05-02', 4100.20, 22),
('user_010', 33, 'KR', '2024-04-08', 1890.90, 11);

-- 샘플 상품 데이터 삽입
INSERT INTO products (product_id, category, price, brand) VALUES
('prod_001', 'Electronics', 599.99, 'TechBrand'),
('prod_002', 'Clothing', 89.50, 'FashionCorp'),
('prod_003', 'Books', 24.95, 'ReadMore'),
('prod_004', 'Electronics', 1299.00, 'GadgetPro'),
('prod_005', 'Home', 159.75, 'HomeStyle'),
('prod_006', 'Sports', 79.99, 'ActiveLife'),
('prod_007', 'Electronics', 299.50, 'TechBrand'),
('prod_008', 'Clothing', 125.00, 'PremiumWear'),
('prod_009', 'Books', 19.99, 'ClassicReads'),
('prod_010', 'Home', 89.99, 'ModernLiving');

-- 샘플 세션 데이터 삽입 (Spine 데이터)
INSERT INTO sessions (session_id, user_id, product_id, event_timestamp, time_on_page_seconds, click_count, outcome, treatment_group) VALUES
('sess_001', 'user_001', 'prod_001', '2024-07-01 10:30:00', 180, 5, 1, 'treatment'),
('sess_002', 'user_002', 'prod_002', '2024-07-01 11:15:00', 120, 3, 0, 'control'),
('sess_003', 'user_003', 'prod_003', '2024-07-01 14:20:00', 300, 8, 1, 'treatment'),
('sess_004', 'user_004', 'prod_004', '2024-07-01 16:45:00', 90, 2, 0, 'control'),
('sess_005', 'user_005', 'prod_005', '2024-07-02 09:30:00', 210, 6, 1, 'treatment'),
('sess_006', 'user_006', 'prod_006', '2024-07-02 12:10:00', 75, 1, 0, 'control'),
('sess_007', 'user_007', 'prod_007', '2024-07-02 15:25:00', 450, 12, 1, 'treatment'),
('sess_008', 'user_008', 'prod_008', '2024-07-02 18:30:00', 150, 4, 0, 'control'),
('sess_009', 'user_009', 'prod_009', '2024-07-03 11:00:00', 240, 7, 1, 'treatment'),
('sess_010', 'user_010', 'prod_010', '2024-07-03 14:15:00', 330, 9, 1, 'treatment'),
('sess_011', 'user_001', 'prod_005', '2024-07-03 16:20:00', 105, 2, 0, 'control'),
('sess_012', 'user_003', 'prod_007', '2024-07-04 10:45:00', 270, 8, 1, 'treatment'),
('sess_013', 'user_005', 'prod_002', '2024-07-04 13:30:00', 195, 5, 0, 'control'),
('sess_014', 'user_007', 'prod_004', '2024-07-04 17:10:00', 380, 11, 1, 'treatment'),
('sess_015', 'user_002', 'prod_009', '2024-07-05 09:15:00', 85, 1, 0, 'control');

-- 데이터 검증 쿼리
SELECT 'Users count: ' || COUNT(*) FROM users;
SELECT 'Products count: ' || COUNT(*) FROM products;
SELECT 'Sessions count: ' || COUNT(*) FROM sessions;

-- 결과 분포 확인
SELECT 
    treatment_group,
    outcome,
    COUNT(*) as count
FROM sessions 
GROUP BY treatment_group, outcome 
ORDER BY treatment_group, outcome;

COMMIT; 