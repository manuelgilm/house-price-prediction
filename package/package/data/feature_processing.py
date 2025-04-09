from package.utils.file_system import get_root_project_path
from package.utils.file_system import read_file
import pandas as pd
import os


def create_feature_metadata_table():
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
    print(df.head())


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

    # Convert the dictionary to a DataFrame or any other structure you prefer
    # For example, using pandas:
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
