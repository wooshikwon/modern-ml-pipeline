# src/components/_evaluator/_classification.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.interface import BaseEvaluator
from src.settings import Settings
from src.utils.core.console import get_console

class ClassificationEvaluator(BaseEvaluator):
    METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.console = get_console()

    def evaluate(self, model, X, y, source_df=None):
        self.console.info(f"분류 모델 평가를 시작합니다 - 테스트 데이터: {len(X)}개",
                         rich_message="🎯 Starting classification model evaluation")
        self.console.info(f"평가 대상 클래스: {len(np.unique(y))}개, 분포: {dict(zip(*np.unique(y, return_counts=True)))}",
                         rich_message=f"   📊 Target classes: [cyan]{len(np.unique(y))}[/cyan], distribution analyzed")

        predictions = model.predict(X)
        metrics = {"accuracy": accuracy_score(y, predictions)}

        self.console.info(f"기본 정확도 계산 완료: {metrics['accuracy']:.4f}",
                         rich_message=f"   ✅ Accuracy calculated: [green]{metrics['accuracy']:.4f}[/green]")
        
        # 클래스별 메트릭 (average=None → 클래스별 배열 반환)
        self.console.info("클래스별 정밀도, 재현율, F1 스코어를 계산합니다",
                         rich_message="   🔍 Computing precision, recall, F1-score per class")
        precision_per_class = precision_score(y, predictions, average=None)
        recall_per_class = recall_score(y, predictions, average=None)
        f1_per_class = f1_score(y, predictions, average=None)
        
        # ROC AUC 계산 - 클래스 수에 따라 다르게 처리
        unique_classes, support_per_class = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)

        self.console.info(f"ROC AUC 스코어 계산 중 - {'이진' if n_classes == 2 else '다중'} 분류 모드",
                         rich_message=f"   📈 Computing ROC AUC for [yellow]{'binary' if n_classes == 2 else 'multi-class'}[/yellow] classification")

        if n_classes == 2:
            # Binary classification - predict_proba 필요
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]  # 클래스 1의 확률
                    metrics["roc_auc"] = roc_auc_score(y, y_proba)
                    self.console.info(f"이진 분류 ROC AUC 완료: {metrics['roc_auc']:.4f}",
                                    rich_message=f"   ✅ Binary ROC AUC: [green]{metrics['roc_auc']:.4f}[/green]")
                else:
                    metrics["roc_auc"] = None
                    self.console.warning("모델이 predict_proba를 지원하지 않아 ROC AUC 계산을 스킵합니다",
                                       rich_message="   ⚠️  ROC AUC skipped - model doesn't support predict_proba")
            except Exception as e:
                metrics["roc_auc"] = None
                self.console.warning(f"ROC AUC 계산 중 오류 발생: {str(e)}",
                                   rich_message="   ⚠️  ROC AUC calculation failed - see logs for details")
        else:
            # Multi-class classification - probabilities 필요
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    metrics["roc_auc"] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
                    self.console.info(f"다중 분류 ROC AUC 완료: {metrics['roc_auc']:.4f}",
                                    rich_message=f"   ✅ Multi-class ROC AUC: [green]{metrics['roc_auc']:.4f}[/green]")
                else:
                    # predict_proba가 없으면 roc_auc 계산 불가
                    metrics["roc_auc"] = None
                    self.console.warning("모델이 predict_proba를 지원하지 않아 ROC AUC 계산을 스킵합니다",
                                       rich_message="   ⚠️  ROC AUC skipped - model doesn't support predict_proba")
            except Exception as e:
                # 오류 발생 시 None 설정
                metrics["roc_auc"] = None
                self.console.warning(f"ROC AUC 계산 중 오류 발생: {str(e)}",
                                   rich_message="   ⚠️  ROC AUC calculation failed - see logs for details")
        
        # 클래스별 메트릭을 딕셔너리에 추가
        class_metrics_summary = []
        for i, class_label in enumerate(unique_classes):
            metrics[f"class_{class_label}_precision"] = precision_per_class[i]
            metrics[f"class_{class_label}_recall"] = recall_per_class[i]
            metrics[f"class_{class_label}_f1"] = f1_per_class[i]
            metrics[f"class_{class_label}_support"] = int(support_per_class[i])

            class_metrics_summary.append(f"Class {class_label}: P={precision_per_class[i]:.3f}, R={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}")

        self.console.info(f"클래스별 성능 지표 계산 완료 - 총 {len(unique_classes)}개 클래스",
                         rich_message=f"   📋 Per-class metrics completed for [green]{len(unique_classes)}[/green] classes")
        for summary in class_metrics_summary:
            self.console.info(f"  {summary}",
                            rich_message=f"   📊 [dim]{summary}[/dim]")

        # 통합된 메트릭 표시 시스템 적용
        self.console.log_pipeline_connection(
            "ClassificationEvaluator",
            "MetricsOutput",
            f"{len(metrics)} 지표 계산 완료"
        )

        # 성능 기반 가이던스 제공
        if 'accuracy' in metrics and metrics['accuracy'] is not None:
            if metrics['accuracy'] >= 0.9:
                guidance = "모델 성능이 우수합니다. 프로덕션 배포를 고려하세요."
            elif metrics['accuracy'] >= 0.8:
                guidance = "모델 성능이 좋습니다. 추가 하이퍼파라미터 튜닝으로 개선 가능합니다."
            elif metrics['accuracy'] >= 0.7:
                guidance = "모델 성능이 보통입니다. 특성 엔지니어링이나 모델 변경을 고려하세요."
            else:
                guidance = "모델 성능 개선이 필요합니다. 데이터 품질, 특성 선택, 모델 아키텍처를 재검토하세요."

            self.console.log_performance_guidance("Accuracy", metrics['accuracy'], guidance)

        # 통합된 메트릭 테이블 표시
        total_metrics = sum(1 for k in metrics.keys() if not k.startswith('class_'))
        performance_summary = f"분류 모델 평가 완료: {len(unique_classes)}개 클래스, {total_metrics}개 주요 지표 계산됨"
        self.console.display_unified_metrics_table(metrics, performance_summary)

        return metrics

# Self-registration
from ..registry import EvaluatorRegistry
EvaluatorRegistry.register("classification", ClassificationEvaluator)
