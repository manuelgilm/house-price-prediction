from package.utils.file_system import get_root_project_path
from package.utils.file_system import read_file
import pandas as pd
import os
import numpy as np
import matplotlib.image as mpimg
from typing import Dict
from typing import Optional
from typing import Tuple


def create_feature_metadata_table() -> None:
    """
    Process the dataset to create a feature metadata table.
    This table contains the textual features and their corresponding images.
    """
    df_textual = process_textual_data()
    df_images = process_images()
    # Merge the two DataFrames on the 'id_' column
    df = pd.merge(df_textual, df_images, on="id_", how="inner")
    # Save the merged DataFrame to a CSV file
    root = get_root_project_path()
    df.to_csv(root / "package" / "data" / "feature_metadata.csv", index=False)


def process_textual_data() -> pd.DataFrame:
    """
    Process the textual data to create a feature metadata table.
    This table contains the textual features and their corresponding images.
    """
    textual_data = read_file("data/raw/HousesInfo.txt")
    lines = textual_data.split("\n")
    feature_metadata = []
    for n, line in enumerate(lines):
        if line.strip():  # Ignore empty lines
            n_bedroons, n_bathrooms, area, zipcode, price = line.split(" ")
            feature_metadata.append(
                {
                    "id_": n + 1,
                    "n_bedroons": float(n_bedroons),
                    "n_bathrooms": float(n_bathrooms),
                    "area": float(area),
                    "zipcode": float(zipcode),
                    "price": float(price),
                }
            )

    df = pd.DataFrame(feature_metadata)

    return df


def process_images() -> pd.DataFrame:
    """
    Process the images to create a feature metadata table.
    This table contains the image features and their corresponding textual data.
    :return: DataFrame containing the image metadata.
    """
    root = get_root_project_path()

    print(root.parent.parent / "HOUSES-DATASET" / "Houses Dataset")

    images_list = os.listdir(root.parent.parent / "HOUSES-DATASET" / "Houses Dataset")
    images_list = [image for image in images_list if image.endswith(".jpg")]
    image_folder = root.parent.parent / "HOUSES-DATASET" / "Houses Dataset"
    image_metadata = []
    for image in images_list:
        index = int(image.split("_")[0])
        image_metadata.append(
            {
                "id_": index,
                "bedroom": (image_folder / f"{index}_bedroom.jpg").as_posix(),
                "bathroom": (image_folder / f"{index}_bathroom.jpg").as_posix(),
                "kitchen": (image_folder / f"{index}_kitchen.jpg").as_posix(),
                "frontal": (image_folder / f"{index}_frontal.jpg").as_posix(),
            }
        )

    df = pd.DataFrame(image_metadata)
    df.drop_duplicates(subset=["id_"], inplace=True)

    return df


def get_train_test_val_indexes(indexes, test_size: float = 0.2, val_size: float = 0.1):
    """
    Get the train, validation, and test indexes from the given indexes.
    This function shuffles the indexes and splits them into train, validation, and test sets.

    :param indexes: List of indexes to be split.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :return: Tuple containing the train, validation, and test indexes.
    """
    np.random.shuffle(indexes)
    test_size = int(len(indexes) * test_size)
    val_size = int(len(indexes) * val_size)
    train_size = len(indexes) - test_size - val_size
    train_indexes = indexes[:train_size]
    val_indexes = indexes[train_size : train_size + val_size]
    test_indexes = indexes[train_size + val_size :]
    return train_indexes, val_indexes, test_indexes


def get_dataset(indexes: np.array, df: pd.DataFrame):
    """
    Get the dataset from the given indexes and DataFrame.
    This function processes the images and textual data, and returns them as numpy arrays.

    :param indexes: List of indexes to be used for splitting the dataset.
    :param df: DataFrame containing the feature metadata.
    :return: Tuple containing the image data, textual data, and target values.
    """

    df_index = pd.DataFrame(indexes, columns=["id_"])
    df = pd.merge(df, df_index, on="id_")
    return df


def get_model_data(df: pd.DataFrame) -> tuple:
    """
    Get the model data from the given DataFrame.
    This function processes the images and textual data, and returns them as numpy arrays.

    :param df: DataFrame containing the feature metadata.
    :return: Tuple containing the image data, textual data, and target values.
    """
    textual_data = df.get(["n_bedrooms", "n_bathrooms", "area"])
    target = df.get(["price"]) / max(df["price"])
    input_names_map = {
        "bedroom_image_input": "bedroom",
        "bathroom_image_input": "bathroom",
        "kitchen_image_input": "kitchen",
        "frontal_image_input": "frontal",
    }

    batch_size = len(df)
    x = {
        feature_label: np.array([image for image in df.get(input_key)])
        for feature_label, input_key in input_names_map.items()
    }

    x.update({"textual_data": textual_data.to_numpy().astype(np.float32)})
    y = np.resize(target.to_numpy().astype(np.float32), (batch_size,))

    return x, y
