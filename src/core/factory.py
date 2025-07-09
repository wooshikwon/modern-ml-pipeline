# logger import
import logging
logger = logging.getLogger(__name__)

# library import
import pandas as pd
import mlflow

# module import
from config.settings import Settings
from src.interface.base_model import BaseModel
from src.core.transformer import Transformer
from src.models import XGBoostXLearner, CausalForestModel
from src.core.augmenter import Augmenter
from src.core.preprocessor import Preprocessor

class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    학습 완료된 모델과 Transformer를 함께 감싸는 MLflow Pyfunc 모델 래퍼.
    """

    # 학습이 완료된 모델(model)과 데이터에 맞게 학습된 증강기(augmenter), 전처리기(preprocessor)를 인자로 받는다.
    def __init__(self, model: BaseModel, augmenter: Augmenter, preprocessor: Preprocessor):
        self.model = model # 학습된 모델 객체, predict 메서드에서 사용.
        self.augmenter = augmenter # 학습된 augmenter 객체, predict 메서드에서 사용.
        self.preprocessor = preprocessor # 학습된 preprocessor 객체, predict 메서드에서 사용.

    # Predict 메서드는 MLflow API 서빙 환경에서 필요함. 예측 요청이 들어올 때마다 호출된다.
    # model_input은 예측을 원하는 새로운 원본 데이터(DataFrame).
    def predict(self, context, model_input: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        입력 데이터를 받아 피처 증강, 피처 전처리, 예측을 순차적으로 수행.
        """
        if isinstance(model_input, dict):
            input_data = pd.DataFrame(model_input)
        else:
            input_data = model_input

        augmented_data = self.augmenter.augment(input_data) # 1. 저장된 증강기를 사용해 새로운 입력 데이터를 증강.
        preprocessed_data = self.preprocessor.process(augmented_data) # 2. 저장된 전처리기를 사용해 증강된 데이터를 전처리.
        predictions = self.model.predict(preprocessed_data) # 3. 변환된 데이터를 저장된 모델에 넣어 모델 예측
        return pd.DataFrame(predictions) # 4. 예측된 결과(Series)를 MLflow가 표준적으로 다루는 DataFrame 형태로 변환하여 반환.


class Factory:
    """
    models/ 폴더에 정의된 다수의 모델 레시피 파일을 분기하여 모델 객체를 생성한다.
        - 학습 하려는 모델 레시피 파일을 요청하면, 아래와 같은 흐름으로 모델 객체가 생성됨.
        - main.py(model name input) -> settings.py(load model yaml) -> ** factory.py ** -> trainer.py -> models/{model_name}.py
    """
    # 이 클래스는 trainer.py에서 파이프라인 시작 시 호출.
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Factory 초기화 완료.")

    # 요청된 모델 레시피 파일에 따라 모델 객체를 생성.
    def create_model(self) -> BaseModel:
        model_name = self.settings.model.name
        logger.info(f"'{model_name}' 모델 인스턴스 생성 시작")

        # 모델 이름에 따라 분기하여 적절한 모델 클래스를 인스턴스화.
        if model_name == "xgboost_x_learner":
            return XGBoostXLearner(settings=self.settings)
        elif model_name == "causal_forest":
            return CausalForestModel(settings=self.settings)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: '{model_name}'")

    # --- 학습 완료 후 모델 로깅 전 호출 ---
    # 이 메서드는 trainer.py의 train 메서드 안에서 모델 학습과 평가가 모두 끝난 후, 최종 모델을 MLflow에 로깅하기 직전에 호출.
    def create_pyfunc_wrapper(self, model: BaseModel, augmenter: Augmenter, preprocessor: Preprocessor) -> PyfuncWrapper:
        """MLflow Pyfunc 래퍼를 생성합니다."""
        logger.info("MLflow Pyfunc 래퍼 생성.")
        # 상기 정의된 PyfuncWrapper 클래스의 인스턴스를 생성하여 반환.
        # 학습 완료된 모델과 증강기, 전처리기를 인자로 전달해야 함.
        return PyfuncWrapper(model=model, augmenter=augmenter, preprocessor=preprocessor)