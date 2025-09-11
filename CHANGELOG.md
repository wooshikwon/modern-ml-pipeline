## [Unreleased]

### Added
- Docs: Timeseries 규약(레시피 `data_interface.timestamp_column` 필수), 카탈로그-데이터핸들러 매칭 규칙(LSTM=deeplearning), Feature Store 시 `fetcher.timestamp_column` 권장 문구를 `claudedocs/` 3곳에 추가.
- CLI: 레시피 빌더에서 Timeseries/Feature Store 선택 시 `timestamp_column` 공백 입력 불가 검증 추가.
- CI: PR 게이트에 Validator 크리티컬 체크(`test_recipe_validation_timeseries_requires_timestamp`) 추가.
- Tests: `SettingsBuilder.with_timestamp_column`, `with_treatment_column` 체이닝 헬퍼 추가.

### Changed
- 템플릿: `recipe.yaml.j2`에 Timeseries/Feature Store 관련 주석 강화(Validator 강제, PIT join 기준 안내).
- README: Timeseries 규약/카탈로그 매칭/Feature Store 가이드 섹션 추가.

### Housekeeping
- 향후 경고 정리(MLflow/pandas) 백로그 티켓화 항목 명시.

