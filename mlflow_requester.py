import os
from typing import Dict, Optional

import mlflow
from lightgbm import LGBMRegressor
from mlflow.tracking import MlflowClient
from settings import Settings


class MLFlowRequester:
    def __init__(
            self,
            mlflow_client: MlflowClient,
            model_name: str,
            experiment_name: str,
            run_name: str,
            model_path: str,
            artifact_path: str
    ):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.client = mlflow_client
        self.run_name = run_name
        self.model_path = model_path
        self.artifact_path = artifact_path

    def logging(
            self,
            params: Optional[Dict[str, float]] = None,
            metrics: Optional[Dict[str, float]] = None,
            pipeline: Optional[LGBMRegressor] = None,
    ) -> None:
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            if pipeline:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path=os.path.join(self.model_path, 'pipeline.pickle'),
                    registered_model_name=self.model_name
                )
            if os.path.exists(os.path.join(self.artifact_path, 'feature_importance.json')):
                mlflow.log_artifact(os.path.join(self.artifact_path, 'feature_importance.json'))
            if os.path.exists(os.path.join(self.artifact_path, 'feature_importance.png')):
                mlflow.log_artifact(os.path.join(self.artifact_path, 'feature_importance.png'))


if __name__ == '__main__':
    client = MlflowClient()
    requester = MLFlowRequester(
        mlflow_client=client,
        model_name=Settings.MODEL_NAME,
        experiment_name='DPH-DOPAMINE',
        run_name='Baseline',
        model_path='../../models',
        artifact_path='../artifacts'
    )
    metrics = {
        'maape': 1,
        'mae': 2.2675131348511384,
        'mse': 69.8433199899925,
        'rmse': 8.357231598441706,
        'rmsle': 0.27379456644717326,
        'r2': 0.7875735193861948
    }
    requester.logging(metrics=metrics)
