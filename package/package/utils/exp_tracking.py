import mlflow
from mlflow.entities import Experiment

from typing import Dict
from typing import Optional


def get_or_create_experiment(name: str, tags: Optional[Dict[str, str]]) -> Experiment:
    """
    Get or create an MLflow experiment with the specified name and tags.
    If the experiment already exists, it returns the existing experiment.

    :param name: Name of the experiment.
    :param tags: Optional dictionary of tags to associate with the experiment.
    :return: The experiment object
    """

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name, tags=tags)
        print("Experiment created with ID:", experiment_id)

    experiment = mlflow.set_experiment(name=name)
    return experiment
