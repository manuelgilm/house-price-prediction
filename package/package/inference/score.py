def load_model(run_id: str, artifact_path: str):
    """
    Load the model from MLflow using the run ID.

    :param run_id: The run ID of the MLflow run.
    :param artifact_path: The path in the MLflow artifact store where the model is saved.
    :return: The loaded model.
    """
    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/{artifact_path}")
    # Set the model's input shape if needed
    return model


def get_scoring_data():
    pass
