import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline, FeatureUnion

from base import BaseTransformer, BasePipeline
from configs import fs_cols, es_cols, pt_cols, pd_cols, psycho_score_cols


class SumTransformer(BaseTransformer):
    def __init__(self, selected_cols: list[str], new_feature: str, feature_names: list[str] | None = None) -> None:
        super().__init__()
        self._selected_cols = selected_cols
        self._feature_names = feature_names
        self._new_feature = new_feature

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.loc[:, self._selected_cols]
        data.loc[:, self._new_feature] = data.sum(axis=1)
        self._feature_names = [self._new_feature]
        return data[[self._new_feature]]


class FeatureBuilder(BasePipeline):
    def get_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        fs_pipeline = Pipeline(
            steps=[
                ('fs_transformer', SumTransformer(selected_cols=fs_cols, new_feature='fs'))
            ]
        )
        es_pipeline = Pipeline(
            steps=[
                ('es_transformer', SumTransformer(selected_cols=es_cols, new_feature='es'))
            ]
        )
        pt_pipeline = Pipeline(
            steps=[
                ('pt_transformer', SumTransformer(selected_cols=pt_cols, new_feature='pt'))
            ]
        )
        pd_pipeline = Pipeline(
            steps=[
                ('pd_transformer', SumTransformer(selected_cols=pd_cols, new_feature='pd'))
            ]
        )
        total_pipeline = Pipeline(
            steps=[
                (
                    'total_transformer',
                    SumTransformer(selected_cols=fs_cols + es_cols + pt_cols + pd_cols, new_feature='total')
                )
            ]
        )
        psycho_pipeline = Pipeline(
            steps=[
                ('psycho_transformer', SumTransformer(selected_cols=psycho_score_cols, new_feature='psycho'))
            ]
        )

        feature_transformation = FeatureUnion(
            transformer_list=[
                ('fs_pipeline', fs_pipeline),
                ('es_pipeline', es_pipeline),
                ('pt_pipeline', pt_pipeline),
                ('pd_pipeline', pd_pipeline),
                ('total_pipeline', total_pipeline),
                ('psycho_pipeline', psycho_pipeline),
            ]
        )
        feature_transformation.fit(X=X_copy)
        feature_values = feature_transformation.transform(X_copy)

        self.feature_transformation = feature_transformation

        columns = self.get_feature_names()

        features = pd.DataFrame(data=feature_values, columns=columns)
        logger.info(f'Data transformation finished features:{features.columns}')

        return features
