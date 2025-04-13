from mlflow.models import infer_signature
from mlflow.types.schema import TensorSpec
from mlflow.types.schema import Schema
from mlflow.models import ModelSignature
import mlflow
from typing import Tuple
from typing import Optional
from typing import List
import numpy as np


def get_model_signature(
    image_input_names: Optional[List[str]] = None,
    image_input_shape: Optional[Tuple] = None,
    numerical_input_shape: Optional[Tuple] = None,
) -> ModelSignature:
    """
    Get the model signature for the model.

    :param image_input_names: List of image input names.
    :param image_input_shape: Shape of the image input.
    :param numerical_input_shape: Shape of the numerical input.
    :return: A ModelSignature object representing the model signature.
    """
    image_input_spec = [
        get_image_signature(
            image_shape=image_input_shape, image_input_name=image_input_name
        )
        for image_input_name in image_input_names
    ]

    if numerical_input_shape is not None:
        numerical_input_spec = TensorSpec(
            shape=(-1, *numerical_input_shape),
            type=np.dtype(np.float32),
            name="numerical_input",
        )
        input_specification = image_input_spec + [numerical_input_spec]

    input_specification = image_input_spec

    input_schema = Schema(inputs=input_specification)
    output_specification = TensorSpec(
        shape=(-1, 1), type=np.dtype(np.float32), name="output"
    )
    output_schema = Schema(inputs=[output_specification])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    return signature


def get_image_signature(
    image_shape: Tuple, image_input_name: Optional[str] = None
) -> TensorSpec:
    """
    Get the image signature for the model.

    :param image_shape: Shape of the image input.
    :return: A TensorSpec object representing the image input signature.
    """
    if image_input_name is None:
        image_input_name = "image_input"
    return TensorSpec(
        shape=(-1, *image_shape), type=np.dtype(np.uint8), name=image_input_name
    )
