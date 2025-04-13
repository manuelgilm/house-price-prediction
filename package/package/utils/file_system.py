from pathlib import Path
import pkgutil
import pandas as pd
from io import StringIO
import yaml


def get_root_project_path() -> Path:
    """
    Get the root path of the project.

    :return: Path object representing the root path of the project.
    """
    return Path(__file__).parent.parent.parent


def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    :param file_path: Path to the file to be read.
    :return: Contents of the file as a string.
    """
    data = pkgutil.get_data("package", file_path)
    if data is not None:
        return data.decode("utf-8")


def read_csv_as_dataframe(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return it as a pandas DataFrame.

    :param file_path: Path to the CSV file to be read.
    :return: pandas DataFrame containing the CSV data.
    """
    csv_content = read_file(file_path)
    return pd.read_csv(StringIO(csv_content))


def read_yaml(file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.

    :param file_path: Path to the YAML file to be read.
    :return: Dictionary containing the YAML data.
    """
    yaml_content = read_file(file_path)
    return yaml.safe_load(yaml_content) if yaml_content else {}
