import keras
from package.training.base import CustomModel
from package.training.utils import get_model_signature
from typing import Optional
import mlflow


class CNNPriceRegressor(CustomModel):

    def __init__(self, image_input_shape):
        self.image_input_shape = image_input_shape

    def train(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs: int = 10,
        batch_size: int = 8,
        log_history: bool = True,
    ):
        """
        Train the model using the provided training and validation data.

        :param x_train: Training data.
        :param y_train: Training labels.
        :param x_val: Validation data.
        :param y_val: Validation labels.
        :param epochs: Number of epochs to train. Default is 10.
        :param batch_size: Batch size for training. Default is 32.
        """

        model = self.build_model()
        optimizer = keras.optimizers.Adamax(learning_rate=0.001, decay=1e-3 / 200)

        model.compile(optimizer=optimizer, loss="mean_absolute_percentage_error")
        callbacks = [mlflow.keras.MlflowCallback()] if log_history else None

        model_signature = get_model_signature(
            image_input_names=["image_input"], image_input_shape=self.image_input_shape
        )
        print("model signature")
        print(model_signature)
        with mlflow.start_run() as run:
            model.fit(
                x=x_train,
                y=y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
            )

        self.log_keras_model(model=model, artifact_path="model", run_id=run.info.run_id)

        return run.info.run_id

    def log_keras_model(self, model, artifact_path: str, run_id: str):
        """
        Logs the Keras model to MLflow.

        :param model: The Keras model to log.
        :param artifact_path: The path in the MLflow artifact store where the model will be saved.
        :param run_id: The run ID of the MLflow run.
        """
        with mlflow.start_run(run_id=run_id):
            mlflow.keras.log_model(model, artifact_path=artifact_path, signature=None)

    def get_image_processor(self, output_dim: int, prefix: Optional[str]):
        """
        Create a Keras model for processing images.

        :param output_dim: Shape of the output data.
        :param prefix: Prefix for the input layer name. Default is None.
        :return:
        """

        input_name = f"{prefix}_image_input" if prefix else "image_input"

        x_im_i = keras.Input(shape=self.image_input_shape, name=input_name)
        for n, f in enumerate([16, 32, 64]):
            if n == 0:
                x_im = x_im_i

            x_im = keras.layers.Conv2D(f, (3, 3), activation="relu")(x_im)
            x_im = keras.layers.MaxPool2D((2, 2))(x_im)
            x_im = keras.layers.BatchNormalization()(x_im)

        x_im = keras.layers.Flatten()(x_im)
        x_im = keras.layers.Dense(16, activation="relu")(x_im)
        x_im = keras.layers.BatchNormalization()(x_im)
        x_im = keras.layers.Dense(output_dim, activation="relu")(x_im)

        return x_im_i, x_im

    def build_model(self, output_dim: int = 4, prefix: Optional[str] = None):
        """
        Build the final model by processing images.

        :param output_dim: Shape of the output data. Default is 4.
        :param prefix: Prefix for the input layer name. Default is None.
        :return: A Keras model that processes images.
        """

        x_im_i, x_im = self.get_image_processor(output_dim, prefix)

        # dense layers
        x = keras.layers.Dense(4, activation="relu")(x_im)
        x = keras.layers.Dense(1, activation="linear")(x)

        # creating the model

        model = keras.Model(inputs=x_im_i, outputs=x)

        return model
