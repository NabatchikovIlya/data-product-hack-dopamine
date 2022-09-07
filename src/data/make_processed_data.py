import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline, FeatureUnion

from base import BaseTransformer, BasePipeline
from configs import (  # type: ignore
    score_cols,
    score_mapper,
    reversed_score_cols,
    reversed_score_mapper,
    psycho_score_cols,
    psycho_score_mapper,
)


class ScoreTransformer(BaseTransformer):
    def __init__(self, feature_names: list[str], mapper: dict[str, float]) -> None:
        super().__init__()
        self._mapper = mapper
        self._feature_names = feature_names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.loc[:, self._feature_names]
        return data.replace(self._mapper)


class PreprocessingPipeline(BasePipeline):
    def get_data(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        X_copy = X.copy()
        score_pipeline = Pipeline(
            steps=[
                ('score_transformer', ScoreTransformer(feature_names=score_cols, mapper=score_mapper))
            ]
        )
        reversed_score_pipeline = Pipeline(
            steps=[
                (
                    'reversed_score_transformer',
                    ScoreTransformer(feature_names=reversed_score_cols, mapper=reversed_score_mapper)
                )
            ]
        )
        psycho_score_pipeline = Pipeline(
            steps=[
                (
                    'psycho_score_transformer',
                    ScoreTransformer(feature_names=psycho_score_cols, mapper=psycho_score_mapper)
                )
            ]
        )

        data_transformation = FeatureUnion(
            transformer_list=[
                ('score_pipeline', score_pipeline),
                ('reversed_score_pipeline', reversed_score_pipeline),
                ('psycho_score_pipeline', psycho_score_pipeline),
            ]
        )
        data_transformation.fit(X=X_copy, y=y)
        data_values = data_transformation.transform(X_copy)

        self.feature_transformation = data_transformation

        columns = self.get_feature_names()

        data = pd.DataFrame(data=data_values, columns=columns)
        logger.info(f'Data transformation finished columns:{data.columns}')

        return data


if __name__ == "__main__":
    df = pd.read_excel('../../data/interim/data.xlsx')
    preprocessing_pipeline = PreprocessingPipeline()
    data = preprocessing_pipeline.get_data(X=df)
    data.to_excel('../../data/processed/data.xlsx', index=False)
