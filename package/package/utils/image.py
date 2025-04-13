import numpy as np
from typing import Optional
from typing import Tuple
import matplotlib.image as mpimg
from pathlib import Path
import matplotlib.pyplot as plt
import cv2


def read_image(
    image_path: str, image_size: Optional[Tuple[int, int, int]] = (128, 128, 3)
):
    """
    Read an image from the given path.

    :param image_path: Path to the image file.
    :return: Loaded image.
    """
    # Read the image using matplotlib
    # image = mpimg.imread(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    # Resize the image to (128, 128, 3)
    # image = np.reshape(image, image_size).astype(np.float32)

    return image
