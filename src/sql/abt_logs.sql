DECLARE start_date STRING DEFAULT '{{ start_date }}';
DECLARE end_date STRING default FORMAT_DATE('%Y-%m-%d', DATE_ADD(DATE(start_date), INTERVAL 2 WEEK));

WITH get_markers AS (
  -- android get_markers 필요한 row
  SELECT
    DISTINCT
    'android' AS os,
    CASE
      WHEN EXTRACT(HOUR FROM t.timeMS AT TIME ZONE 'Asia/Seoul') < 2
        THEN DATE(DATETIME(t.timeMS, 'Asia/Seoul')) - INTERVAL 1 DAY
      ELSE DATE(DATETIME(t.timeMS, 'Asia/Seoul'))
    END AS TDDate,
    CASE
      WHEN MOD(
            ABS(FARM_FINGERPRINT(CONCAT(CAST(t.callerLog.memberId AS STRING), '250509_deliverable_marker'))),
            100
          ) BETWEEN 0 AND 49 THEN 'control'
      WHEN MOD(
            ABS(FARM_FINGERPRINT(CONCAT(CAST(t.callerLog.memberId AS STRING), '250509_deliverable_marker'))),
            100
          ) BETWEEN 50 AND 99 THEN 'target'
      ELSE NULL
    END AS grp,
    t.callerLog.memberId AS member_id,
    marker.zone.id AS zone_id,
    t.interval.startAt AS interval_startAt,
    t.interval.endAt AS interval_endAt,

    marker.zone.location.lng AS zone_lng,
    marker.zone.location.lat AS zone_lat,
    marker.zone.available AS zone_available,

    t.timeMS
  FROM
    socar-data.socar_server_3.GET_MARKERS_V2 AS t
  LEFT JOIN UNNEST(t.markersV2) AS marker
  WHERE
    t.timeMS
    BETWEEN TIMESTAMP(start_date) - INTERVAL 9 HOUR
      AND TIMESTAMP(end_date) + INTERVAL 1 DAY - INTERVAL 9 HOUR
    AND marker.zone.id IS NOT NULL

  UNION DISTINCT

  -- ios get_markers 필요한 row만
  SELECT
    DISTINCT
    'ios' AS os,
    CASE
      WHEN EXTRACT(HOUR FROM t.timeMS AT TIME ZONE 'Asia/Seoul') < 2
        THEN DATE(DATETIME(t.timeMS, 'Asia/Seoul')) - INTERVAL 1 DAY
      ELSE DATE(DATETIME(t.timeMS, 'Asia/Seoul'))
    END AS TDDate,
    CASE
      WHEN MOD(
            ABS(FARM_FINGERPRINT(CONCAT(CAST(t.callerLog.memberId AS STRING), '250509_deliverable_marker'))),
            100
          ) BETWEEN 0 AND 49 THEN 'control'
      WHEN MOD(
            ABS(FARM_FINGERPRINT(CONCAT(CAST(t.callerLog.memberId AS STRING), '250509_deliverable_marker'))),
            100
          ) BETWEEN 50 AND 99 THEN 'target'
      ELSE NULL
    END AS grp,
    t.callerLog.memberId AS member_id,
    marker.zone.id AS zone_id,
    t.interval.startAt AS interval_startAt,
    t.interval.endAt AS interval_endAt,

    marker.zone.location.lng AS zone_lng,
    marker.zone.location.lat AS zone_lat,
    marker.zone.available AS zone_available,

    t.timeMS
  FROM
    socar-data.socar_server_3.GET_MARKERS AS t
  LEFT JOIN UNNEST(t.markers) AS marker
  WHERE
    t.timeMS
    BETWEEN TIMESTAMP(start_date) - INTERVAL 9 HOUR
      AND TIMESTAMP(end_date) + INTERVAL 1 DAY - INTERVAL 9 HOUR
    AND marker.zone.id IS NOT NULL
)

, sessions AS (
  -- 2단계: 맨 첫값만 뽑아오고, 나머지 집계 수행
  SELECT
    gm.os,
    gm.TDDate,
    gm.grp,
    gm.member_id,
    gm.interval_startAt,
    gm.interval_endAt,
    gm.zone_id,

    -- 각 그룹에서 timeMS 기준 첫 번째
    ARRAY_AGG(gm.zone_lng            ORDER BY gm.timeMS ASC LIMIT 1)[OFFSET(0)] AS zone_lng,
    ARRAY_AGG(gm.zone_lat            ORDER BY gm.timeMS ASC LIMIT 1)[OFFSET(0)] AS zone_lat,
    ARRAY_AGG(gm.zone_available      ORDER BY gm.timeMS ASC LIMIT 1)[OFFSET(0)] AS zone_available,

  FROM
    get_markers AS gm

  GROUP BY
    gm.os,
    gm.TDDate,
    gm.grp,
    gm.member_id,
    gm.interval_startAt,
    gm.interval_endAt,
    gm.zone_id
)

, get_car_class_section AS (
  SELECT
    DISTINCT
    CASE
      WHEN EXTRACT(HOUR FROM t.timeMs AT TIME ZONE 'Asia/Seoul') < 2
        THEN DATE(DATETIME(t.timeMs, 'Asia/Seoul')) - INTERVAL 1 DAY
      ELSE DATE(DATETIME(t.timeMs, 'Asia/Seoul'))
    END AS TDDate,
    cl.memberId     AS member_id,
    t.rentalLocation.zoneId  AS zone_id,
    iv.startAt      AS interval_startAt,
    iv.endAt        AS interval_endAt,
  FROM
    socar-data.socar_server_3.GET_CAR_CLASS_SECTIONS AS t
  CROSS JOIN UNNEST(t.callerLog) AS cl
  CROSS JOIN UNNEST(t.interval) AS iv
  WHERE
    t.timeMs
    BETWEEN TIMESTAMP(start_date) - INTERVAL 9 HOUR
      AND TIMESTAMP(end_date) + INTERVAL 1 DAY - INTERVAL 9 HOUR
  GROUP BY ALL
)

, zone_click AS (
  SELECT
    s.TDDate,
    s.grp,
    s.member_id,
    s.interval_startAt,
    s.interval_endAt,
    s.zone_id,
    s.zone_lng,
    s.zone_lat,
    s.zone_available,
    CASE
      WHEN za.zone_id IS NOT NULL THEN 1
      ELSE 0
    END AS click_flag
  FROM
    sessions AS s
  LEFT JOIN
    get_car_class_section AS za
    USING (TDDate, interval_startAt, interval_endAt, member_id, zone_id)
)
,reservations AS (
  -- rental_option 없이 원래 로직을 그대로 사용합니다.
  SELECT
    zc.*,
    SUM(
      CASE
        WHEN zc.click_flag = 1 AND ri.reservation_count IS NOT NULL THEN 1
        ELSE 0
      END
    ) AS reservation_flag,
    SUM(
      CASE
        WHEN zc.click_flag = 1 AND ri.rental_option = 'DELIVERY' AND (zc.zone_id = ri.zone_id) THEN 1
        ELSE 0
      END
    ) AS zone_vroom_flag
  FROM zone_click zc
  LEFT JOIN (
    SELECT
      DISTINCT
      CASE
        WHEN EXTRACT(HOUR FROM timeMs AT TIME ZONE 'Asia/Seoul') < 2 THEN DATE(DATETIME(timeMs, 'Asia/Seoul')) - INTERVAL 1 DAY
        ELSE DATE(DATETIME(timeMs, 'Asia/Seoul'))
      END AS TDDate,
      t.callerLog.memberId AS member_id,
      t.itinerary.startPoint.zoneId AS zone_id,
      t.itinerary.interval.startAt AS interval_startAt,
      t.itinerary.interval.endAt AS interval_endAt,
      t.itinerary.rentalOption AS rental_option,
      t.itinerary.startPoint.deliveryLocation.lng AS delivery_start_lng,
      t.itinerary.startPoint.deliveryLocation.lat AS delivery_start_lat,
      COUNT(DISTINCT t.carRental.id) AS reservation_count
    FROM socar-data.socar_server_3.RESERVE_CAR_RENTAL t
    WHERE t.timeMs BETWEEN TIMESTAMP(start_date) - INTERVAL 9 HOUR AND TIMESTAMP(end_date) + INTERVAL 1 DAY - INTERVAL 9 HOUR
      AND t.carRental.id IS NOT NULL
    GROUP BY ALL
  ) ri
  ON zc.TDDate = ri.TDDate
    AND zc.member_id = ri.member_id
    AND zc.interval_startAt = ri.interval_startAt
    AND zc.interval_endAt = ri.interval_endAt
    AND CASE
          WHEN ri.rental_option = 'ZONE' THEN zc.zone_id = ri.zone_id
          WHEN ri.rental_option = 'DELIVERY' THEN (zc.zone_id = ri.zone_id) OR (zc.zone_lng = ri.delivery_start_lng AND zc.zone_lat = ri.delivery_start_lat)
        END
  GROUP BY ALL
),
result AS (
  SELECT
    member_id,
    grp,
    MAX(reservation_flag) AS outcome
  FROM reservations
  WHERE
    grp IS NOT NULL
  GROUP BY
    member_id,
    grp
)
, member_properties AS (
  SELECT
    member_id,
    COALESCE(member_gender, 'NaN') AS member_gender,
    COALESCE(member_age, -1) AS member_age,
    CASE WHEN member_passport_current_state IN ('SUBSCRIBED', 'RENEWAL_SUSPENDED')
      THEN 1
      ELSE 0
    END AS passport_usage,
    COALESCE(CAST(member_socar_club_level AS INT), 0) AS socar_club_level,
    member_credit_balance,
    car_sharing_use_count,
    COALESCE(DATE_DIFF(DATE(start_date), last_app_open_date, DAY), 0) AS app_open_interval
  FROM
    socar-data.socar_data_queries_bifrost_hist.bf_user_properties_fct_hist
  WHERE
    DATETIME(dbt_valid_from, 'Asia/Seoul') < DATETIME(start_date)
    AND (DATETIME(dbt_valid_to, 'Asia/Seoul') > DATETIME(start_date)
      OR DATETIME(dbt_valid_to, 'Asia/Seoul') IS NULL)
)
, reservation_info AS (
  SELECT
    member_id,
    COUNT(DISTINCT id) AS rsvn_90_count,
    COUNT(
      DISTINCT
      CASE WHEN
        DATETIME(created_at, 'Asia/Seoul') BETWEEN DATETIME(start_date) - INTERVAL 2 MONTH AND DATETIME(start_date) - INTERVAL 1 MONTH
          THEN id
        ELSE NULL
        END
      ) AS rsvn_60_count,
    COUNT(
      DISTINCT
      CASE WHEN
        DATETIME(created_at, 'Asia/Seoul') BETWEEN DATETIME(start_date) - INTERVAL 1 MONTH AND DATETIME(start_date)
          THEN id
        ELSE NULL
        END
      ) AS rsvn_30_count,
  FROM
    socar-data.tianjin_replica.reservation_info
  WHERE
    member_imaginary in (0, 9)
    AND DATETIME(created_at, 'Asia/Seoul') BETWEEN DATETIME(start_date) - INTERVAL 3 MONTH AND DATETIME(start_date)
    AND (
        cancel_at IS NULL
        OR DATETIME(cancel_at, 'Asia/Seoul') > DATETIME(start_date)
    )
  GROUP BY
    member_id
)
, member_result AS (
  SELECT
    mp.member_id,
    mp.member_gender,
    mp.member_age,
    mp.passport_usage,
    mp.socar_club_level,
    mp.member_credit_balance,
    COALESCE(ri.rsvn_90_count, 0) AS rsvn_90_count,
    COALESCE(ri.rsvn_60_count, 0) AS rsvn_60_count,
    COALESCE(ri.rsvn_30_count, 0) AS rsvn_30_count
  FROM
    member_properties mp
  LEFT JOIN
    reservation_info ri
    ON mp.member_id = ri.member_id
)

SELECT
  mr.*,
  r.grp,
  r.outcome
FROM
  result r
LEFT JOIN
  member_result mr
  ON r.member_id = mr.member_id
WHERE
  mr.member_id IS NOT NULL
