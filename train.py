import pandas as pd
import numpy as np
import rfpimp
import seaborn as sns
import scipy
import json
from args import get_args
import joblib
from matplotlib import pyplot
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from feature_selection import Preprocessing

args = get_args()


class Trainer:
    """
    select and load model, train, metrics, test
    """

    def __init__(self, args, features, scaler=None, preproc=None):
        self.args = args
        self.features = features
        self.registry = {
            "linear_regression": linear_model.LinearRegression,
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
            "decision_tree": DecisionTreeRegressor,
        }
        if preproc is None:
            preproc = Preprocessing(args)
        if scaler is None:
            scaler = StandardScaler()
        self.preproc = preproc
        self.scaler = scaler

    def train(self, trainset, testset):
        try:
            with open(
                "models/{}/config.json".format(self.args.model_name)
            ) as json_file:
                self.model_params = json.load(json_file)
            # self.model_params = vars(getattr(self.args, self.args.model_name))
            self.model = self.registry[self.args.model_name](**self.model_params)
        except:
            self.model_params = None
            self.model = self.registry[self.args.model_name]()
        self.model.fit(trainset[0], trainset[1])

    def predict(self, testset, inference=False):
        if inference:
            return self.model.predict(testset)
            # return self.model.predict(np.expand_dims(testset[0].iloc[0], axis=0))
        return self.model.predict(testset[0])

    def metrics(self, testset, y_pred):
        if self.args.model_name == "linear_regression":
            importance = self.model.coef_
        elif (
            self.args.model_name == "decision_tree"
            or self.args.model_name == "random_forest"
        ):
            importance = self.model.feature_importances_
            # summarize feature importance
        for i, v in enumerate(importance):
            print("Feature: %0d, Score: %.5f" % (i, v))
        # plot feature importance
        pyplot.bar(self.features, importance)
        pyplot.show()
        # The mean squared error
        mse = mean_squared_error(testset[1], y_pred)
        # The mean absolute error
        mae = mean_absolute_error(testset[1], y_pred)
        # The coefficient of determination: 1 is perfect prediction
        r2score = r2_score(testset[1], y_pred)
        return mse, mae, r2score

    def save(self):
        state = {"model": self.model, "preproc": self.preproc, "scaler": self.scaler}
        joblib.dump(
            state, "{}/{}/model".format(self.args.model_path, self.args.model_name)
        )
        dict_args = vars(self.args)
        json_txt = json.dumps(dict_args, indent=4)
        with open(
            "{}/{}/args".format(self.args.model_path, self.args.model_name), "w+"
        ) as file:
            file.write(json_txt)

    def load(self):
        state = joblib.load(
            "{}/{}/model".format(self.args.model_path, self.args.model_name)
        )
        self.model = state["model"]
        self.preproc = state["preproc"]
        self.scaler = state["scaler"]
        with open(
            "{}/{}/args".format(self.args.model_path, self.args.model_name), "r"
        ) as file:
            self.args = json.load(file)
        try:
            with open(
                "{}/{}/config.json".format(self.args.model_path, self.args.model_name)
            ) as json_file:
                self.model_params = json.load(json_file)
            # self.model_params = vars(getattr(self.args, self.args.model_name))
        except:
            self.model_params = None


def get_scaler(dataframe, columns, scaler):
    scaled_features = dataframe.copy()
    features = scaled_features[columns]
    if scaler is None:
        scaler = StandardScaler().fit(features.values)
    else:
        scaler = scaler.fit(features.values)
    del scaled_features

    return scaler


def transform_features(dataframe, columns, scaler):
    # scaled_features = dataframe.copy()
    features = dataframe[columns]
    features = scaler.transform(features.values)
    dataframe[columns] = features

    return dataframe


def invert_features(dataframe, columns, scaler):
    inverted_features = scaler.inverse_transform(dataframe[columns].values)
    dataframe[columns] = inverted_features

    return dataframe


def load_dataset():
    # preproc.clean_dataset()
    df = pd.read_csv(args.post_preproc_data_path)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    features = list(df.columns)[1:]
    train, test = train_test_split(df, test_size=0.2)
    y_train = train["value"]
    x_train = train.drop(axis=1, columns=["value"])
    col_names = ["Age", "Heightcm", "Weightkg", "BMI"]
    scaler = get_scaler(x_train, col_names, StandardScaler())
    x_train_scaled = transform_features(x_train, col_names, scaler)
    # x_train = invert_features(x_train_scaled, col_names, scaler)

    y_test = test["value"]
    x_test = test.drop(axis=1, columns=["value"])
    x_test_scaled = transform_features(x_test, col_names, scaler)

    trainset = (x_train_scaled, y_train)
    testset = (x_test_scaled, y_test)
    return features, trainset, testset, scaler


if __name__ == "__main__":
    preproc = Preprocessing(args)
    preproc.clean_dataset()
    features, trainset, testset, scaler = load_dataset()
    trainer = Trainer(args, features, scaler, preproc)
    trainer.train(trainset, testset)
    y_pred = trainer.predict(testset)
    print(y_pred)
    # mse, mae, r2score = trainer.metrics(testset, y_pred)
    # # The mean squared error
    # print("Mean squared error: %.2f" % mse)
    # # The mean absolute error
    # print("Mean absolute error: %.2f" % mae)
    # # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f" % r2score)
    trainer.save()

# print(trainer.predict(testset, inference=True))
