from package.utils.file_system import read_csv_as_dataframe
from package.utils.file_system import get_root_project_path
import numpy as np
from typing import Optional
from typing import Tuple
from typing import List
import pickle


class CustomDataset:
    def __init__(self):
        pass

    def load_feature_indexes(
        self, path: Optional[str] = "data/processed/feature_metadata.csv"
    ) -> np.ndarray:
        """
        Load the feature indexes from the given path.
        This method should be overridden by subclasses to load specific indexes.

        :param path: Path to the CSV file containing the feature indexes.
        :return: Numpy array of feature indexes.
        """
        df = read_csv_as_dataframe(path)
        indexes = df["id_"].values
        return indexes

    def get_train_test_val_indexes(
        self, test_size: float = 0.2, val_size: float = 0.1
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Get the train, validation, and test indexes from the given indexes.
        This function shuffles the indexes and splits them into train, validation, and test sets.
        """
        indexes = self.load_feature_indexes()
        np.random.shuffle(indexes)
        test_size = int(len(indexes) * test_size)
        val_size = int(len(indexes) * val_size)
        train_size = len(indexes) - test_size - val_size
        train_indexes = indexes[:train_size]
        val_indexes = indexes[train_size : train_size + val_size]
        test_indexes = indexes[train_size + val_size :]
        return train_indexes, val_indexes, test_indexes

    def save_indexes(
        self,
        path: Optional[str] = "package/data/processed/indexes.pkl",
        regenerate: Optional[bool] = False,
    ) -> dict:
        """
        Save the indexes to a file.
        This method should be overridden by subclasses to save specific indexes.
        """
        # Check if the file already exists and if we need to regenerate it
        root = get_root_project_path()
        path = root / path
        if path.exists() and not regenerate:
            print(f"File already exists at {path}. Use regenerate=True to overwrite.")
            return

        train_ind, val_ind, test_ind = self.get_train_test_val_indexes()
        indexes = {
            "train": train_ind,
            "val": val_ind,
            "test": test_ind,
        }
        root = get_root_project_path()
        path = root / path
        with open(path, "wb") as f:
            pickle.dump(indexes, f)

        return indexes

    def load_indexes(
        self, path: Optional[str] = "package/data/processed/indexes.pkl"
    ) -> dict:
        """
        Load the indexes from a file.
        This method should be overridden by subclasses to load specific indexes.

        :path: Path to the file containing the indexes.
        :return: Dictionary containing the train, validation, and test indexes.
        """

        root = get_root_project_path()
        path = root / path
        if not path.exists():
            raise FileNotFoundError(
                f"File not found at {path}. Please generate indexes first."
            )
        # Load the indexes from the file
        with open(path, "rb") as f:
            indexes = pickle.load(f)
        return indexes
