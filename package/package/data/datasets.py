from package.data.base import CustomDataset
from typing import Dict
from typing import Any


class HouseImageDataset(CustomDataset):
    """
    A class to represent a dataset of house images.

    Attributes
    ----------
    images : list
        A list of image file paths.
    labels : list
        A list of labels corresponding to the images.

    Methods
    -------
    load_data():
        Loads the images and labels from the specified directory.
    preprocess_images():
        Preprocesses the images for model input.
    """

    def __init__(self, house_area: str) -> None:
        """
        Constructs all the necessary attributes for the HouseImageDataset object.
        The class generates dataset considering a specific house area as feature for the model.

        :param house_area: The house area to consider for the dataset.["bedroom", "bathroom", "kitchen", "frontal"]
        """
        self.house_area = house_area

    def get_dataset(model_inputs_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Returns the dataset for the specified house area.
        The dataset is a dictionary containing the image (arrays) and labels.
        The image is a numpy array of shape (n_samples, height, width, channels).

        :param model_inputs_map: A dictionary mapping the model input names to their corresponding image paths.
        :return: A dictionary containing the dataset.
        """

        dataset = {}
        pass


class HouseMultiImageDataset(CustomDataset):
    """
    A class to represent a dataset with multiple images and labels.

    Attributes
    ----------
    image_paths : list
        A list of lists of image file paths.
    labels : list
        A list of labels corresponding to the images.

    Methods
    -------
    load_data():
        Loads the images and labels from the specified directory.
    preprocess_images():
        Preprocesses the images for model input.
    """

    def __init__(self, image_paths, labels):
        """
        Constructs all the necessary attributes for the MultiImageDataset object.

        Parameters
        ----------
            image_paths : list of lists of str
                List of lists containing paths to multiple images for each sample.
            labels : list
                List of labels corresponding to the images.
        """
        self.image_paths = image_paths
        self.labels = labels


class HoulsMultiTypeDataset(CustomDataset):
    """
    A class to represent a dataset with multiple images and labels.

    Attributes
    ----------
    image_paths : list
        A list of lists of image file paths.
    labels : list
        A list of labels corresponding to the images.

    Methods
    -------
    load_data():
        Loads the images and labels from the specified directory.
    preprocess_images():
        Preprocesses the images for model input.
    """

    def __init__(self, image_paths, labels):
        """
        Constructs all the necessary attributes for the MultiImageDataset object.

        Parameters
        ----------
            image_paths : list of lists of str
                List of lists containing paths to multiple images for each sample.
            labels : list
                List of labels corresponding to the images.
        """
        self.image_paths = image_paths
        self.labels = labels
