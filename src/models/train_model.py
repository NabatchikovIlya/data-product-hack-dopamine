import json
import os
import pickle
import warnings
from functools import partial
from typing import Dict, Optional, Tuple, Any

import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor, early_stopping
from loguru import logger
from matplotlib import pyplot as plt
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from data import PreprocessingPipe, preparing_pipeline
from features import FeatureBuilder


def save_feature_importance(feature_importance: Dict[str, float]) -> None:
    with open('../artifacts/feature_importance.json', 'w') as file:
        json.dump(feature_importance, file, indent=4)


def save_plot_feature_importance(feature_importance: Dict[str, float], fig_size=(40, 20)) -> None:
    df_imp = pd.DataFrame([feature_importance]).T.reset_index().rename(columns={'index': 'feature', 0: 'importance'})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=3)
    sns.barplot(x="importance", y="feature", data=df_imp)
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('../artifacts/feature_importance.png')


class TrainPipe:
    def __init__(self, model_path: str):
        self.pipeline = None
        self.model_path = os.path.join(model_path, 'pipeline.pickle')

    @staticmethod
    def objective(trial, X: pd.DataFrame, y: np.ndarray, n_splits=5) -> np.ndarray:
        param_grid = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),  # Boosting learning rate
            'n_estimators': trial.suggest_int('n_estimators', 30, 5000),  # Number of boosted trees to fit
            'num_leaves': trial.suggest_int('num_leaves', 2, 512),  # Maximum tree leaves for base learners
            'max_depth': trial.suggest_int('max_depth', -1, 256),  # Max tree depth for base learners, <=0 means unlimit
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 256),  # Minimal number of data in one leaf
            'max_bin': trial.suggest_int('max_bin', 100, 1000),  # Max num of bins that feature values will be bucketed
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),  # Subsample ratio of the training instance
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),  # Frequency of subsample, <=0 means no enable
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            # Subsample ratio of columns when constructing each tree
            'min_child_weight': trial.suggest_float('min_child_weight', 0.01, 10.0),
            # Minimum sum of instance weight (hessian) needed in a child (leaf)
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),  # L2 regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),  # L1 regularization
        }

        cv = KFold(n_splits=n_splits)
        cv_scores = np.empty(n_splits)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            logger.info(f"For id: {idx} TRAIN: {train_idx}, TEST: {test_idx}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = LGBMRegressor(
                boosting_type='gbdt',
                metric='mape',
                n_jobs=1,
                verbose=-1,
                random_state=42,
                objective='poisson',
                **param_grid
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="mape",
                callbacks=[
                    LightGBMPruningCallback(trial, "mape"),
                    early_stopping(stopping_rounds=500)
                ],
            )
            y_pred = model.predict(X_test)
            cv_scores[idx] = mean_absolute_percentage_error(y_test, y_pred)

        return np.mean(cv_scores)

    def find_best_params(self, X: pd.DataFrame, y: np.ndarray, n_iter=10) -> Tuple[Dict[str, Any], Optional[float]]:
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize")
        partial_objective = partial(self.objective, X=X, y=y)
        study.optimize(partial_objective, n_trials=n_iter)  # type: ignore

        trial = study.best_trial
        best_params = trial.params
        logger.info(f"Best CV score: {trial.value}")
        logger.info(f'Best parameters: {best_params}')
        return best_params, trial.value

    def fit(self, X: pd.DataFrame, y: np.ndarray, best_params: Dict[str, float]):
        X_copy = X.copy()
        self.pipeline = Pipeline(steps=[
            ('model', LGBMRegressor(
                boosting_type='gbdt',
                metric='mape',
                n_jobs=1,
                verbose=-1,
                random_state=42,
                objective='poisson',
                **best_params))]
        )
        if self.pipeline:
            self.pipeline.fit(X_copy, y)
            self.save_pipeline()

        return self.pipeline

    def save_pipeline(self) -> None:
        with open(self.model_path, 'wb+') as f:
            pickle.dump(self.pipeline, f)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    preprocessing_pipe = PreprocessingPipe()
    feature_builder = FeatureBuilder()

    raw_data = pd.read_excel('../../data/raw/data.xlsx')
    y = np.array(raw_data['dopamine'].values.ravel())
    X = raw_data.drop(columns=['dopamine'])
    interim_data = preparing_pipeline(df=X)
    processed_data = preprocessing_pipe.get_data(X=interim_data)
    features = feature_builder.get_features(X=processed_data)

    train_pipe = TrainPipe(model_path='../../models')
    best_params, cv_score = train_pipe.find_best_params(X=features, y=y, n_iter=5)

    train_pipe.fit(X=features, y=y, best_params=best_params)

    with open('../../models/pipeline.pickle', 'rb') as f:
        model = pickle.load(f)

    print(model.predict(features))
