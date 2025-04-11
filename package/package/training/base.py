import mlflow
from abc import ABC, abstractmethod
from typing import Optional


class CustomModel(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input):
        # Implement your prediction logic here using the loaded model and tokenizer
        pass

    def log_custom_model(self) -> str:
        """
        Logs the custom model to MLflow.
        """
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(
                artifact_path=self.__class__.__name__, python_model=self
            )

        return run.info.run_id
