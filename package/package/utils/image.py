import numpy as np
from typing import Optional
from typing import Tuple
import matplotlib.image as mpimg
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from typing import List


def read_image(
    image_path: str, image_size: Optional[Tuple[int, int, int]] = (128, 128)
):
    """
    Read an image from the given path.

    :param image_path: Path to the image file.
    :return: Loaded image.
    """
    # Read the image using matplotlib
    # image = mpimg.imread(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    return image


def stack_images(ims: List, final_size: Optional[Tuple[int, int]] = (128, 128)):
    """
    Stack images into a single image.
    :param ims: List of images to stack.
    :return: Stacked image.
    """
    im1 = np.hstack((ims[0], ims[1]))
    im2 = np.hstack((ims[2], ims[3]))
    im = np.vstack((im1, im2))
    im = cv2.resize(im, final_size)
    return im
