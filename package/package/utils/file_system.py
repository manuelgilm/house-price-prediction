from pathlib import Path


def get_root_project_path() -> Path:
    """
    Get the root path of the project.

    :return: Path object representing the root path of the project.
    """
    return Path(__file__).parent.parent.parent
