# src/components/_evaluator/_regression.py
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.interface import BaseEvaluator
from src.settings import Settings
from src.utils.core.console import get_console

class RegressionEvaluator(BaseEvaluator):

    METRIC_KEYS = ["r2_score", "mean_squared_error"]

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.console = get_console()

    def evaluate(self, model, X, y, source_df=None):
        self.console.info(f"회귀 모델 평가를 시작합니다 - 테스트 데이터: {len(X)}개",
                         rich_message="📊 Starting regression model evaluation")

        # 타겟 변수 기본 통계
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_min, y_max = np.min(y), np.max(y)
        self.console.info(f"타겟 변수 분리: 평균={y_mean:.4f}, 표준편차={y_std:.4f}, 범위=[{y_min:.4f}, {y_max:.4f}]",
                         rich_message=f"   📊 Target statistics: mean={y_mean:.4f}, std={y_std:.4f}")

        self.console.info("모델 예측 수행 중...",
                         rich_message="   🔮 Making predictions...")
        predictions = model.predict(X)

        # 예측값 기본 통계
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        pred_min, pred_max = np.min(predictions), np.max(predictions)
        self.console.info(f"예측값 분리: 평균={pred_mean:.4f}, 표준편차={pred_std:.4f}, 범위=[{pred_min:.4f}, {pred_max:.4f}]",
                         rich_message=f"   📊 Prediction statistics: mean={pred_mean:.4f}, std={pred_std:.4f}")

        # 주요 메트릭 계산
        self.console.info("회귀 성능 메트릭을 계산합니다...",
                         rich_message="   📋 Computing regression metrics...")

        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)

        metrics = {
            "r2_score": r2,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "mean_absolute_error": mae
        }

        # 메트릭 해석 및 로깅
        self.console.info(f"R² 스코어: {r2:.4f} ({'저조' if r2 < 0.5 else '양호' if r2 > 0.8 else '보통'} 성능)",
                         rich_message=f"   ✅ R² Score: [{'red' if r2 < 0.5 else 'green' if r2 > 0.8 else 'yellow'}]{r2:.4f}[/{'red' if r2 < 0.5 else 'green' if r2 > 0.8 else 'yellow'}]")

        self.console.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f} (vs 타겟 표준편차: {y_std:.4f})",
                         rich_message=f"   📊 MSE: [cyan]{mse:.4f}[/cyan], RMSE: [cyan]{rmse:.4f}[/cyan]")

        self.console.info(f"MAE: {mae:.4f} (vs 타겟 표준편차: {y_std:.4f})",
                         rich_message=f"   📊 MAE: [cyan]{mae:.4f}[/cyan]")

        # 예측 성능 요약
        accuracy_pct = r2 * 100 if r2 > 0 else 0
        self.console.info(f"회귀 모델 평가 완료 - 예측 정확도: {accuracy_pct:.1f}%",
                         rich_message=f"🏁 Regression evaluation completed - [green]{accuracy_pct:.1f}%[/green] explained variance")

        return metrics

# Self-registration
from ..registry import EvaluatorRegistry
EvaluatorRegistry.register("regression", RegressionEvaluator)
