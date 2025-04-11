from pathlib import Path
import pkgutil


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
