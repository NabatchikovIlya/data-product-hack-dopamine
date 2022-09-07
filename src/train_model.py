import pandas as pd

from data import PreprocessingPipeline, preparing_pipeline
from features.build_features import FeatureBuilder


if __name__ == "__main__":
    preprocessing_pipeline = PreprocessingPipeline()
    feature_pipeline = FeatureBuilder()

    raw_data = pd.read_excel('../data/raw/data.xlsx')
    interim_data = preparing_pipeline(raw_data)
    process_data = preprocessing_pipeline.get_data(X=interim_data)
    features = feature_pipeline.get_data(X=process_data)
    print('meow')
