from package.data.feature_processing import create_feature_metadata_table
from package.data.feature_processing import get_train_test_val_indexes
from package.data.feature_processing import get_model_data
from package.data.feature_processing import get_dataset
from package.utils.image import read_image
from package.utils.file_system import read_csv_as_dataframe

from package.training.models import CNNPriceRegressor
import pandas as pd
import numpy as np
import mlflow


def test():
    create_feature_metadata_table()


def train():
    df = read_csv_as_dataframe("data/feature_metadata.csv")
    indexes = df["id_"].values
    train_indexes, val_indexes, test_indexes = get_train_test_val_indexes(
        indexes, test_size=0.2, val_size=0.1
    )

    df_train = get_dataset(train_indexes, df)
    df_val = get_dataset(val_indexes, df)
    df_test = get_dataset(test_indexes, df)

    train_data = df_train.map(lambda x: read_image(x) if type(x) is str else x)
    val_data = df_val.map(lambda x: read_image(x) if type(x) is str else x)
    test_data = df_test.map(lambda x: read_image(x) if type(x) is str else x)

    x_train, y_train = get_model_data(train_data)
    x_val, y_val = get_model_data(val_data)
    x_test, y_test = get_model_data(test_data)
    x_train_ = {}
    x_train_["image_input"] = x_train["bedroom_image_input"]

    x_val_ = {}
    x_val_["image_input"] = x_val["bedroom_image_input"]

    x_test_ = {}
    x_test_["image_input"] = x_test["bedroom_image_input"]

    print(x_val_.keys())
    print(x_train_.keys())
    print(x_train_["image_input"].shape)
    print(x_val_["image_input"].shape)
    print(y_train.shape)
    print(y_val.shape)
    cvnn_regressor = CNNPriceRegressor(image_input_shape=(128, 128, 3))
    run_id = cvnn_regressor.train(
        x_train_, y_train, x_val_, y_val, epochs=10, batch_size=32
    )

    print(f"Model trained and logged with run ID: {run_id}")
    model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/model")
    predictions = model.predict(x_test_)
    print(predictions)
