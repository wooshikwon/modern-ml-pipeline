# src/interface/base_factory.py

from abc import ABC, abstractmethod
import mlflow

# 필요한 타입 힌트를 위해 import
from src.settings.settings import Settings
# BaseModel import 제거: 외부 라이브러리 직접 사용으로 전환
from src.core.preprocessor import Preprocessor

"""
아래와 같은 PyfuncWrapper 클래스를 같은 파일 안에 병렬로 만들어 사용하는 것을 권장.

class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    '''
    데이터 전처리기와 모델을 함께 감싸는 범용 MLflow 래퍼.
    '''
    def __init__(self, model: BaseModel, preprocessor: Preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        preprocessed_data = self.preprocessor.transform(model_input)
        predictions = self.model.predict(preprocessed_data)
        return pd.DataFrame(predictions)
"""

class BaseFactory(ABC):
    """
    객체 생성을 책임지는 팩토리의 기본 인터페이스(추상 클래스)입니다.
    이 인터페이스를 상속받아 구체적인 팩토리 클래스를 구현해야 합니다.
    """

    @abstractmethod
    def create_model(self):
        """
        외부 라이브러리 기반 동적 모델 객체를 생성합니다.
        """
        raise NotImplementedError

    @abstractmethod
    def create_pyfunc_wrapper(self, model, preprocessor: Preprocessor) -> mlflow.pyfunc.PythonModel:
        """
        학습된 모델과 전처리기를 MLflow가 이해할 수 있는 형태로 포장합니다.
        """
        raise NotImplementedError