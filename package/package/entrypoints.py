from package.data.feature_processing import create_feature_metadata_table
from package.data.feature_processing import get_train_test_val_indexes
from package.data.feature_processing import get_model_data
from package.data.feature_processing import get_dataset
from package.utils.image import read_image
from package.utils.file_system import read_csv_as_dataframe

from package.training.models import CNNPriceRegressor
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from package.data.base import CustomDataset
from package.data.datasets import HousePriceDataset

# measure regression performance
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd


def test():
    dataset = HousePriceDataset()
    image_labels = [
        ("kitchen", "single_image"),
        ("bathroom", "single_image"),
        ("frontal", "single_image"),
        ("bedroom", "single_image"),
        (None, "multi_image"),
        (None, "combined"),
    ]

    for image_label, dataset_type in image_labels:
        print("Training model for image label:", image_label)
        print("Dataset type:", dataset_type)
        # Load the dataset
        x_train, y_train, x_val, y_val, x_test, y_test = (
            dataset.get_train_test_val_data(
                dataset_type=dataset_type, image_label=image_label
            )
        )

        max_price = y_train["price"].max()
        y_train["price"] = y_train["price"] / max_price

        if dataset_type == "combined":
            registered_model_name = "combined_model"
            cvnn_regressor = CNNPriceRegressor(
                image_input_shape=(128, 128, 3), numerical_input_shape=(3,)
            )
        elif dataset_type == "single_image":
            cvnn_regressor = CNNPriceRegressor(image_input_shape=(128, 128, 3))
            registered_model_name = f"single_image_{image_label}_model"
        else:
            cvnn_regressor = CNNPriceRegressor(image_input_shape=(128, 128, 3))
            registered_model_name = "multi_image_model"

        print(x_train.keys())
        run_id = cvnn_regressor.train(
            x_train,
            y_train["price"],
            x_val,
            y_val["price"],
            epochs=200,
            batch_size=8,
            registered_model_name=registered_model_name,
            dataset_type=dataset_type,
        )

        print(f"Model trained and logged with run ID: {run_id}")
        model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/model")
        print(x_test.keys())
        predictions = model.predict(x_test)

        eval_data = pd.DataFrame(y_test)
        eval_data["predictions"] = max_price * predictions

        with mlflow.start_run(run_id=run_id) as run:
            mlflow.evaluate(
                model_type="regressor",
                data=eval_data,
                targets="price",
                predictions="predictions",
            )


def train():
    df = read_csv_as_dataframe("data/processed/feature_metadata.csv")
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

    max_price = y_train.max()
    # rescaling the price to be between 0 and 1
    y_train = y_train / max_price
    x_val, y_val = get_model_data(val_data)
    # rescaling the price to be between 0 and 1
    y_val = y_val / max_price
    x_test, y_test = get_model_data(test_data)
    # rescaling the price to be between 0 and 1
    y_test = y_test / max_price
    x_train_ = {}
    x_train_["image_input"] = x_train["kitchen_image_input"]

    x_val_ = {}
    x_val_["image_input"] = x_val["kitchen_image_input"]

    x_test_ = {}
    x_test_["image_input"] = x_test["kitchen_image_input"]

    print(x_val_.keys())
    print(x_train_.keys())
    print(x_train_["image_input"].shape)
    print(x_val_["image_input"].shape)
    print(y_train.shape)
    print(y_val.shape)
    cvnn_regressor = CNNPriceRegressor(image_input_shape=(128, 128, 3))
    run_id = cvnn_regressor.train(
        x_train_, y_train, x_val_, y_val, epochs=200, batch_size=8
    )

    print(f"Model trained and logged with run ID: {run_id}")
    model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/model")
    predictions = model.predict(x_test_)

    # measure regression performance
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_test, predictions)
    print(f"R2: {r2}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")


def train_multi_image():
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

    max_price = y_train.max()
    # rescaling the price to be between 0 and 1
    y_train = y_train / max_price
    x_val, y_val = get_model_data(val_data)
    # rescaling the price to be between 0 and 1
    y_val = y_val / max_price
    x_test, y_test = get_model_data(test_data)
    # rescaling the price to be between 0 and 1
    y_test = y_test / max_price
    x_train_ = {}
    x_train_["image_input"] = x_train["kitchen_image_input"]

    x_val_ = {}
    x_val_["image_input"] = x_val["kitchen_image_input"]

    x_test_ = {}
    x_test_["image_input"] = x_test["kitchen_image_input"]

    print(x_val_.keys())
    print(x_train_.keys())
    print(x_train_["image_input"].shape)
    print(x_val_["image_input"].shape)
    print(y_train.shape)
    print(y_val.shape)
    cvnn_regressor = CNNPriceRegressor(image_input_shape=(128, 128, 3))
    run_id = cvnn_regressor.train(
        x_train_, y_train, x_val_, y_val, epochs=200, batch_size=8
    )

    print(f"Model trained and logged with run ID: {run_id}")
    model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/model")
    predictions = model.predict(x_test_)

    # measure regression performance
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_test, predictions)
    print(f"R2: {r2}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
