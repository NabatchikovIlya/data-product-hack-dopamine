import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline, FeatureUnion

from base import BaseTransformer, BasePipeline
from configs import fs_cols, es_cols, pt_cols, pd_cols, psycho_score_cols  # type: ignore


class SumTransformer(BaseTransformer):
    def __init__(self, feature_names: list[str], new_feature: str) -> None:
        super().__init__()
        self._feature_names = feature_names
        self._new_feature = new_feature

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.loc[:, self._feature_names]
        data.loc[:, self._new_feature] = data.sum(axis=1)
        self._feature_names = [self._new_feature]
        return data[[self._new_feature]]


class FeatureBuilder(BasePipeline):
    def get_data(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        X_copy = X.copy()
        fs_pipeline = Pipeline(
            steps=[
                ('fs_transformer', SumTransformer(feature_names=fs_cols, new_feature='fs'))
            ]
        )
        es_pipeline = Pipeline(
            steps=[
                ('es_transformer', SumTransformer(feature_names=es_cols, new_feature='es'))
            ]
        )
        pt_pipeline = Pipeline(
            steps=[
                ('pt_transformer', SumTransformer(feature_names=pt_cols, new_feature='pt'))
            ]
        )
        pd_pipeline = Pipeline(
            steps=[
                ('pd_transformer', SumTransformer(feature_names=pd_cols, new_feature='pd'))
            ]
        )
        psycho_pipeline = Pipeline(
            steps=[
                ('psycho_transformer', SumTransformer(feature_names=psycho_score_cols, new_feature='psycho'))
            ]
        )

        feature_transformation = FeatureUnion(
            transformer_list=[
                ('fs_pipeline', fs_pipeline),
                ('es_pipeline', es_pipeline),
                ('pt_pipeline', pt_pipeline),
                ('pd_pipeline', pd_pipeline),
                ('psycho_pipeline', psycho_pipeline),
            ]
        )
        feature_transformation.fit(X=X_copy, y=y)
        feature_values = feature_transformation.transform(X_copy)

        self.feature_transformation = feature_transformation

        columns = self.get_feature_names()

        features = pd.DataFrame(data=feature_values, columns=columns)
        logger.info(f'Data transformation finished features:{features.columns}')

        return features


if __name__ == "__main__":
    df = pd.read_excel('../../data/processed/data.xlsx')
    feature_pipeline = FeatureBuilder()
    features = feature_pipeline.get_data(X=df)
    print('meow')
