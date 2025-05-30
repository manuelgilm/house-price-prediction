from package.data.base import CustomDataset
from typing import Dict
from typing import Any
from typing import Tuple
from typing import Optional
from typing import List
from package.utils.file_system import read_csv_as_dataframe
from package.utils.image import read_image
from package.utils.image import stack_images
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class HousePriceDataset(CustomDataset):

    def __init__(self, regenerate_indexes: Optional[bool] = False) -> None:
        """
        Constructs all the necessary attributes for the HousePriceDataset object.
        The class generates dataset considering a specific house area as feature for the model.

        :param regenerate_index: A boolean flag to indicate whether to regenerate the indexes of the dataset.
        """
        self.scaler = None  # only used for combined dataset
        try:
            self.indexes = self.load_indexes()
        except FileNotFoundError:
            print("Indexes file not found. Generating new indexes.")
            self.indexes = self.save_indexes(regenerate=regenerate_indexes)

    def get_image_dataset(
        self,
        image_label: str,
        model_input_label: str,
        model_output_label: str,
        mode: Optional[str] = "train",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns the dataset for the specified image label.
        The dataset is a dictionary containing the image (arrays) and labels.

        :param image_label: The image label to consider for the dataset.["bedroom", "bathroom", "kitchen", "frontal"]
        :param mode: The mode to consider for the dataset.["train", "val", "test"]
        :return: A tuple containing the dataset and labels.
        """
        df = self._get_feature_dataframe(mode=mode)
        df = df[["id_", image_label, "price"]]
        df = df.map(lambda x: read_image(x) if type(x) is str else x)

        x = {model_input_label: np.array([image for image in df.get(image_label)])}
        y = {model_output_label: np.array([price for price in df.get("price")])}
        return x, y

    def get_train_test_val_data(
        self, dataset_type: str, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """

        :param dataset_type: The type of dataset to consider for the dataset.["single_image","multi_image", "combined"]
        :return: A tuple containing the dataset and labels.
        """
        model_output_label = kwargs.get("model_output_label", "price")
        image_label = kwargs.get("image_label", "kitchen")
        model_input_label = kwargs.get("model_input_label", "image_input")
        if dataset_type == "single_image":

            x_train, y_train = self.get_image_dataset(
                image_label=image_label,
                model_input_label=model_input_label,
                model_output_label=model_output_label,
                mode="train",
            )
            x_val, y_val = self.get_image_dataset(
                image_label=image_label,
                model_input_label=model_input_label,
                model_output_label=model_output_label,
                mode="val",
            )
            x_test, y_test = self.get_image_dataset(
                image_label=image_label,
                model_input_label=model_input_label,
                model_output_label=model_output_label,
                mode="test",
            )

        elif dataset_type == "multi_image":
            x_train, y_train = self.get_multi_image_dataset(
                mode="train",
                model_input_label=model_input_label,
                model_output_label="price",
            )
            x_val, y_val = self.get_multi_image_dataset(
                mode="val",
                model_input_label=model_input_label,
                model_output_label="price",
            )
            x_test, y_test = self.get_multi_image_dataset(
                mode="test",
                model_input_label=model_input_label,
                model_output_label="price",
            )
        elif dataset_type == "combined":
            x_train, y_train = self.get_combined_dataset(
                model_image_input_label=model_input_label,
                model_numerical_input_label="numerical_input",
                model_output_label="price",
                mode="train",
            )
            x_val, y_val = self.get_combined_dataset(
                model_image_input_label=model_input_label,
                model_numerical_input_label="numerical_input",
                model_output_label="price",
                mode="val",
            )
            x_test, y_test = self.get_combined_dataset(
                model_image_input_label=model_input_label,
                model_numerical_input_label="numerical_input",
                model_output_label="price",
                mode="test",
            )
        else:
            raise ValueError("Invalid dataset type.")

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _get_feature_dataframe(self, mode: Optional[str] = "train") -> pd.DataFrame:
        """
        Get the feature dataframe for the specified mode.
        The dataframe contains the feature metadata for the specified mode.

        :param mode: The mode to consider for the dataset.["train", "val", "test"]
        :return: A pandas DataFrame containing the feature metadata.
        """

        indexes = self.indexes[mode]
        df = read_csv_as_dataframe("data/processed/feature_metadata.csv")
        df_index = pd.DataFrame(indexes, columns=["id_"])
        df = pd.merge(df, df_index, on="id_")
        return df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from the DataFrame based on predefined thresholds.

        :param df: The DataFrame to process.
        :return: The DataFrame with outliers removed.
        """
        df_filtered = df[(df["price"]<= 700000) & (df["area"]<=8000) & (df["n_bedrooms"]<=7)]
        return df_filtered

    def get_multi_image_dataset(
        self,
        mode: Optional[str] = "train",
        model_input_label: Optional[str] = None,
        model_output_label: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get the multi-image dataset for the specified mode.

        The dataset is a dictionary containing the images (arrays) and labels.
        The images are numpy arrays of shape (n_samples, height, width, channels).

        :param mode: The mode to consider for the dataset.["train", "val", "test"]
        :param model_input_label: The label for the input images.
        :param model_output_label: The label for the output prices.
        :return: A tuple containing the dataset and labels.
        """
        df = self._get_feature_dataframe(mode=mode)
        df = df[["id_", "kitchen", "bathroom", "bedroom", "frontal", "price"]]
        df = df.map(
            lambda x: read_image(x, image_size=(256, 256)) if type(x) is str else x
        )
        # stack all images into a single image
        df["multi_image"] = df[["kitchen", "bathroom", "bedroom", "frontal"]].apply(
            lambda x: stack_images(x), axis=1
        )
        x = {model_input_label: np.array([image for image in df.get("multi_image")])}
        y = {model_output_label: np.array([price for price in df.get("price")])}
        return x, y

    def get_combined_dataset(
        self,
        model_image_input_label: str,
        model_numerical_input_label: str,
        model_output_label: str,
        mode: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        df = self._get_feature_dataframe(mode=mode)
        df = self.remove_outliers(df)
        df = df.map(
            lambda x: read_image(x, image_size=(512, 512)) if type(x) is str else x
        )

        # stack all images into a single image
        df["multi_image"] = df[["kitchen", "bathroom", "bedroom", "frontal"]].apply(
            lambda x: stack_images(x, final_size=(256, 256)), axis=1
        )
        x = {
            model_image_input_label: np.array(
                [image for image in df.get("multi_image")]
            ),
            model_numerical_input_label: np.array(
                [
                    [n_bathrooms, n_bedrooms, area]
                    for n_bathrooms, n_bedrooms, area in zip(
                        df["n_bathrooms"], df["n_bedrooms"], df["area"]
                    )
                ]
            ).astype(np.float32),
        }
        y = {model_output_label: np.array([price for price in df.get("price")])}

        # scale the numerical input data
        if mode == "train":
            self.scaler = MinMaxScaler()
            x[model_numerical_input_label] = self.scaler.fit_transform(
                x[model_numerical_input_label]
            )
        else:
            x[model_numerical_input_label] = self.scaler.transform(
                x[model_numerical_input_label]
            )

        return x, y

    def get_scaler(self) -> StandardScaler:
        """
        Returns the scaler used for the numerical input data.

        :return: A StandardScaler object.
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet.")
        return self.scaler
