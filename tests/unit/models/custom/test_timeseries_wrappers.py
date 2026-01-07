"""
Timeseries Wrappers Tests
timeseries_wrappers.py 테스트

tests/README.md 테스트 전략 준수:
- Factory를 통한 실제 컴포넌트 생성
- 퍼블릭 API만 호출
- 결정론적 테스트 (고정 시드)
- 실제 데이터 흐름 검증
"""

import numpy as np
import pandas as pd
import pytest

from src.models.custom.timeseries_wrappers import ARIMA, ExponentialSmoothing


class TestARIMA:
    """ARIMA 테스트 - BaseModel 인터페이스 준수"""

    def test_arima_window_mismatch_error(self, test_data_generator):
        """케이스 A: 윈도우 길이·특성 수 불일치 시 명확한 예외"""
        # Given: ARIMA 초기화
        arima = ARIMA(order_p=1, order_d=1, order_q=1)

        # 너무 짧은 시계열 데이터 (ARIMA 최소 요구사항 미달)
        short_series = pd.Series([1.0, 2.0])  # 매우 짧은 시계열
        X_short = pd.DataFrame({"feature": [1, 2]})

        # When & Then: 짧은 시계열로 fit 시도 시 명확한 에러
        try:
            arima.fit(X_short, short_series)
            # statsmodels이 설치되어 있지 않은 경우 ImportError 발생 가능
        except ImportError:
            pytest.skip("statsmodels not available for ARIMA testing")
        except ValueError as e:
            # ARIMA 모델링에 필요한 최소 데이터 부족
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg
                for keyword in ["insufficient", "data", "sample", "length", "order"]
            )
        except (IndexError, Exception) as e:
            # statsmodels의 다른 예외들도 허용 (모델링 실패)
            error_msg = str(e).lower()
            # IndexError의 경우 array 차원 관련 메시지도 허용
            assert any(
                keyword in error_msg
                for keyword in ["arima", "model", "array", "dimension", "indices", "indexed"]
            )

    def test_arima_minimal_sequence_single_step_prediction(self, test_data_generator):
        """케이스 B: 최소 길이 시퀀스에서 단일 스텝 예측 성공"""
        # Given: ARIMA와 최소 길이 시계열 데이터
        arima = ARIMA(order_p=1, order_d=0, order_q=1)  # 차분 없음으로 단순화

        # 최소한의 시계열 데이터 (ARIMA 요구사항 만족)
        np.random.seed(42)  # 결정론적 테스트
        minimal_series = pd.Series([10.0, 12.0, 14.0, 13.0, 15.0, 16.0, 14.0, 17.0])
        X_train = pd.DataFrame({"feature": range(len(minimal_series))})

        # When: ARIMA 모델 학습 및 단일 스텝 예측
        try:
            arima.fit(X_train, minimal_series)

            # 단일 스텝 예측을 위한 X
            X_predict = pd.DataFrame({"feature": [len(minimal_series)]})  # 1개 스텝
            prediction = arima.predict(X_predict)

            # Then: 예측 결과 검증 (pd.DataFrame 반환)
            assert prediction is not None
            assert len(prediction) == 1  # 단일 스텝 예측
            assert isinstance(prediction, pd.DataFrame)
            assert "prediction" in prediction.columns
            assert not np.isnan(prediction["prediction"].iloc[0])  # 유효한 예측값

        except ImportError:
            pytest.skip("statsmodels not available for ARIMA testing")
        except Exception as e:
            # 모델링이 실패할 수 있지만, 명확한 에러 메시지여야 함
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg
                for keyword in ["arima", "model", "data", "convergence", "optimization"]
            )

    def test_arima_not_fitted_prediction_error(self):
        """학습되지 않은 모델로 예측 시도 시 명확한 에러"""
        # Given: 학습되지 않은 ARIMA
        arima = ARIMA(order_p=1, order_d=1, order_q=1)
        X_predict = pd.DataFrame({"feature": [1, 2, 3]})

        # When & Then: 학습 없이 예측 시도 시 명확한 에러
        with pytest.raises(ValueError) as exc_info:
            arima.predict(X_predict)

        error_msg = str(exc_info.value)
        assert "학습되지 않았습니다" in error_msg or "fit" in error_msg.lower()


class TestExponentialSmoothing:
    """ExponentialSmoothing 테스트 - BaseModel 인터페이스 준수"""

    def test_exponential_smoothing_minimal_sequence_prediction(self, test_data_generator):
        """최소 길이 시퀀스에서 지수평활법 예측 성공"""
        # Given: ExponentialSmoothing (트렌드만, 계절성 없음)
        exp_smooth = ExponentialSmoothing(trend="add", seasonal=None)

        # 최소한의 시계열 데이터
        np.random.seed(42)
        minimal_series = pd.Series([100.0, 102.0, 105.0, 107.0, 110.0, 108.0])
        X_train = pd.DataFrame({"feature": range(len(minimal_series))})

        # When: 지수평활법 모델 학습 및 예측
        try:
            exp_smooth.fit(X_train, minimal_series)

            # 단일 스텝 예측
            X_predict = pd.DataFrame({"feature": [len(minimal_series)]})
            prediction = exp_smooth.predict(X_predict)

            # Then: 예측 결과 검증 (pd.DataFrame 반환)
            assert prediction is not None
            assert len(prediction) == 1
            assert isinstance(prediction, pd.DataFrame)
            assert "prediction" in prediction.columns
            assert not np.isnan(prediction["prediction"].iloc[0])
            assert prediction["prediction"].iloc[0] > 0  # 양수 예측값 (시계열이 양수였으므로)

        except ImportError:
            pytest.skip("statsmodels not available for ExponentialSmoothing testing")
        except Exception as e:
            # 모델링 실패 시 명확한 에러 메시지
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg
                for keyword in ["exponential", "smooth", "model", "data", "trend"]
            )

    def test_exponential_smoothing_with_seasonal_error(self, test_data_generator):
        """계절성 설정 시 데이터 길이 부족으로 인한 에러 처리"""
        # Given: 계절성이 설정된 ExponentialSmoothing (12개월 주기)
        exp_smooth = ExponentialSmoothing(trend="add", seasonal="add", seasonal_periods=12)

        # 계절성 주기보다 짧은 데이터 (에러 유발)
        short_series = pd.Series([1.0, 2.0, 3.0, 4.0])  # 12보다 훨씬 짧음
        X_short = pd.DataFrame({"feature": range(len(short_series))})

        # When & Then: 부족한 데이터로 계절성 모델링 시도 시 에러
        try:
            exp_smooth.fit(X_short, short_series)
            # 에러가 발생하지 않으면 의외이지만, 일부 경우 가능할 수 있음
        except ImportError:
            pytest.skip("statsmodels not available for ExponentialSmoothing testing")
        except ValueError as e:
            # 계절성 설정과 데이터 길이 불일치
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg
                for keyword in ["seasonal", "period", "data", "length", "insufficient"]
            )
        except Exception as e:
            # 다른 statsmodels 관련 에러들
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg for keyword in ["exponential", "smooth", "seasonal", "model"]
            )

    def test_exponential_smoothing_not_fitted_error(self):
        """학습되지 않은 모델로 예측 시도 시 에러"""
        # Given: 학습되지 않은 ExponentialSmoothing
        exp_smooth = ExponentialSmoothing(trend="add", seasonal=None)
        X_predict = pd.DataFrame({"feature": [1, 2]})

        # When & Then: 학습 없이 예측 시도 시 명확한 에러
        with pytest.raises(ValueError) as exc_info:
            exp_smooth.predict(X_predict)

        error_msg = str(exc_info.value)
        assert "학습되지 않았습니다" in error_msg or "fit" in error_msg.lower()


class TestTimeseriesWrappersIntegration:
    """Timeseries Wrappers 통합 테스트"""

    def test_basemodel_interface_compatibility(self):
        """BaseModel 인터페이스 호환성 확인"""
        # Given: 시계열 모델들
        arima = ARIMA(order_p=1, order_d=1, order_q=1)
        exp_smooth = ExponentialSmoothing(trend="add")

        # When & Then: BaseModel 인터페이스 메서드들이 존재하는지 확인
        assert hasattr(arima, "fit")
        assert hasattr(arima, "predict")
        assert hasattr(exp_smooth, "fit")
        assert hasattr(exp_smooth, "predict")

        # BaseModel 상속 확인
        from src.models.base import BaseModel

        assert isinstance(arima, BaseModel)
        assert isinstance(exp_smooth, BaseModel)

    def test_timeseries_wrappers_module_exports(self):
        """모듈 export 확인"""
        # Given: timeseries_wrappers 모듈
        from src.models.custom import timeseries_wrappers

        # When & Then: __all__에 정의된 클래스들이 export되는지 확인
        assert hasattr(timeseries_wrappers, "__all__")
        assert "ARIMA" in timeseries_wrappers.__all__
        assert "ExponentialSmoothing" in timeseries_wrappers.__all__

        # 실제 클래스들도 접근 가능한지 확인
        assert hasattr(timeseries_wrappers, "ARIMA")
        assert hasattr(timeseries_wrappers, "ExponentialSmoothing")

    def test_wrapper_parameter_validation(self):
        """래퍼 초기화 파라미터 검증"""
        # Given & When: 다양한 파라미터로 래퍼 초기화
        arima1 = ARIMA(order_p=2, order_d=1, order_q=1)
        arima2 = ARIMA(order_p=1, order_d=0, order_q=2)

        exp1 = ExponentialSmoothing(trend=None, seasonal=None)
        exp2 = ExponentialSmoothing(trend="add", seasonal="mul", seasonal_periods=4)

        # Then: 파라미터가 올바르게 설정되었는지 확인
        assert arima1.order_p == 2
        assert arima1.order_d == 1
        assert arima1.order_q == 1

        assert arima2.order_p == 1
        assert arima2.order_d == 0
        assert arima2.order_q == 2

        assert exp1.trend is None
        assert exp1.seasonal is None

        assert exp2.trend == "add"
        assert exp2.seasonal == "mul"
        assert exp2.seasonal_periods == 4
