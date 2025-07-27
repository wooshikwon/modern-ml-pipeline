from typing import Dict, Any, Optional
import importlib
from src.utils.system.logger import logger


class TuningUtils:
    """하이퍼파라미터 튜닝 관련 유틸리티 클래스"""
    
    @staticmethod
    def create_model_with_params(class_path: str, params: Dict[str, Any]):
        """동적 하이퍼파라미터로 모델 생성"""
        try:
            # 모듈 경로와 클래스 이름 분리
            module_path, class_name = class_path.rsplit('.', 1)
            
            # 동적 모듈 로드
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # 하이퍼파라미터 적용하여 인스턴스 생성
            logger.info(f"모델 생성: {class_path} with params: {params}")
            return model_class(**params)
                
        except Exception as e:
            logger.error(f"모델 생성 실패: {class_path}, 파라미터: {params}, 오류: {e}")
            raise ValueError(f"모델을 생성할 수 없습니다: {class_path}") from e
    
    @staticmethod
    def extract_optimization_score(metrics: Dict[str, Any], metric_name: str) -> float:
        """메트릭에서 최적화용 점수 추출"""
        if metric_name in metrics:
            score = metrics[metric_name]
        else:
            # 기본값으로 첫 번째 메트릭 사용
            if metrics:
                score = list(metrics.values())[0]
                logger.warning(f"지정된 메트릭 '{metric_name}'을 찾을 수 없어 첫 번째 메트릭 사용: {score}")
            else:
                score = 0.0
                logger.warning("메트릭이 비어있어 0.0 반환")
        
        return float(score)
    
    @staticmethod
    def get_stratify_column(df, task_type: str, data_interface) -> Optional[str]:
        """데이터프레임에서 stratify할 컬럼 결정"""
        stratify_col = None
        
        if task_type == "causal" and data_interface.treatment_column:
            if data_interface.treatment_column in df.columns:
                stratify_col = data_interface.treatment_column
        elif task_type == "classification" and data_interface.target_column:
            if data_interface.target_column in df.columns:
                stratify_col = data_interface.target_column
        
        # stratify가 가능한지 확인 (unique 값이 2개 이상)
        if stratify_col and df[stratify_col].nunique() > 1:
            return stratify_col
        else:
            return None
    
    @staticmethod
    def validate_hyperparameter_config(hyperparams_config: Dict[str, Any]) -> bool:
        """하이퍼파라미터 설정의 유효성 검증"""
        for param_name, param_config in hyperparams_config.items():
            if isinstance(param_config, dict) and 'type' in param_config:
                param_type = param_config['type']
                
                if param_type == 'float':
                    if 'low' not in param_config or 'high' not in param_config:
                        logger.error(f"float 타입 파라미터 '{param_name}'에 low, high가 필요합니다.")
                        return False
                elif param_type == 'int':
                    if 'low' not in param_config or 'high' not in param_config:
                        logger.error(f"int 타입 파라미터 '{param_name}'에 low, high가 필요합니다.")
                        return False
                elif param_type == 'categorical':
                    if 'choices' not in param_config:
                        logger.error(f"categorical 타입 파라미터 '{param_name}'에 choices가 필요합니다.")
                        return False
                else:
                    logger.warning(f"지원하지 않는 파라미터 타입: {param_type} (파라미터: {param_name})")
        
        return True
    
    @staticmethod
    def create_optimization_metadata(study, start_time, end_time, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """최적화 결과 메타데이터 생성"""
        try:
            return {
                'enabled': True,
                'engine': 'optuna',
                'best_params': best_params,
                'best_score': study.best_value,
                'total_trials': len(study.trials),
                'pruned_trials': len([t for t in study.trials if hasattr(t.state, 'name') and t.state.name == 'PRUNED']),
                'optimization_time_seconds': (end_time - start_time).total_seconds(),
                'study_name': study.study_name
            }
        except Exception as e:
            logger.warning(f"최적화 메타데이터 생성 실패: {e}")
            return {
                'enabled': True,
                'engine': 'optuna',
                'best_params': best_params,
                'error': str(e)
            } 