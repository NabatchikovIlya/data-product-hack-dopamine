import pickle

import pandas as pd

from data import PreprocessingPipe  # type: ignore
from features import FeatureBuilder
from pydantic_models import ServiceOutput, TestScores, Score, Level, PsychoLevel, PsychoScore


class DofamineService:
    def __init__(self, preprocessing_pipe: PreprocessingPipe, feature_builder: FeatureBuilder, model_path: str):
        self.preprocessing_pipe = preprocessing_pipe
        self.feature_builder = feature_builder
        self.model_path = model_path
        self.model = self.load_dopamine_model()

    def load_dopamine_model(self):
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def get_level(value: float, lower_level: int, upper_level: int) -> Level:
        if value <= lower_level:
            return Level.LOW
        elif value >= upper_level:
            return Level.HIGH
        else:
            return Level.MEDIUM

    @staticmethod
    def get_psycho_level(value: float, bound: int) -> PsychoLevel:
        if value <= bound:
            return PsychoLevel.NORMAL
        else:
            return PsychoLevel.ABNORMAL

    def get_test_scores(self, features: pd.DataFrame) -> TestScores:
        test_scores = TestScores(
            decentration=Score(
                value=float(features.fs), level=self.get_level(value=float(features.fs), lower_level=11, upper_level=22)
            ),
            empathy=Score(
                value=float(features.es), level=self.get_level(value=float(features.es), lower_level=14, upper_level=27)
            ),
            empathic_care=Score(
                value=float(features.pt), level=self.get_level(value=float(features.pt), lower_level=12, upper_level=23)
            ),
            empathic_distress=Score(
                value=float(features.pd), level=self.get_level(value=float(features.pd), lower_level=8, upper_level=19)
            ),
            psychological_indicator=PsychoScore(
                value=float(features.psycho), level=self.get_psycho_level(value=float(features.psycho), bound=30)
            ),
            total=Score(
                value=float(features.total),
                level=self.get_level(value=float(features.total), lower_level=45, upper_level=91)
            )
        )
        return test_scores

    def get_test_results(self, data: dict[str, str]) -> ServiceOutput:
        X = pd.DataFrame(data=[data])
        processed_data = self.preprocessing_pipe.get_data(X=X)
        features = self.feature_builder.get_features(X=processed_data)
        test_scores = self.get_test_scores(features=features)
        dopamine_value = data.get('dopamine') if data.get('dopamine') else self.model.predict(features)
        output = ServiceOutput(test_scores=test_scores, dopamine=dopamine_value)
        return output
