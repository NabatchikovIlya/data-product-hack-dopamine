from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._feature_names = None
        super().__init__()

    def fit(self, X=None, y=None):
        return self

    def get_feature_names_out(self) -> list[str]:
        return list(self._feature_names)


class BasePipeline:
    def __init__(self) -> None:
        self.feature_transformation = None

    def get_feature_names(self) -> list[str] | None:
        if self.feature_transformation:
            columns = []
            transformation_list = self.feature_transformation.transformer_list
            for i in range(len(transformation_list)):
                columns.extend(transformation_list[i][1].steps[-1][1].get_feature_names_out())
            return columns
        else:
            return None
