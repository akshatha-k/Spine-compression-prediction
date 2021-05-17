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
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from feature_selection.Preprocessing import Preprocessing

args = get_args()

class Trainer:
    """
    select and load model, train, metrics, test
    """

    def __init__(self, args, features):
        self.args = args
        self.features = features
        self.registry = {
            "linear_regression": linear_model.LinearRegression,
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
            "decision_tree": DecisionTreeRegressor,
        }
        self.preproc = Preprocessing()

    def train(self, trainset, testset):
        try:
            with open("models/{}/config.json".format(self.args.model_name)) as json_file:
                self.model_params = json.load(json_file) 
            #self.model_params = vars(getattr(self.args, self.args.model_name))
            self.model = self.registry[self.args.model_name](**self.model_params)
        except:
            self.model_params = None
            self.model = self.registry[self.args.model_name]()
        self.model.fit(trainset[0], trainset[1])

    def predict(self, testset, inference=False):
        if inference:
            
            return self.model.predict(np.expand_dims(testset[0].iloc[0], axis=0))
        y_pred = self.model.predict(testset[0])
        return y_pred

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
        state = {'model':self.model, 'preproc': self.preproc}
        joblib.dump(state, "models/{}/model".format(self.args.model_name))
        dict_args = vars(self.args)
        json_txt = json.dumps(dict_args, indent=4)
        with open("models/{}/args".format(self.args.model_name), "w+") as file:
            file.write(json_txt)

    def load(self):
        state= joblib.load("models/{}/model".format(self.args.model_name))
        self.model = state['model']
        self.preproc = state['preproc']
        with open("models/{}/args".format(self.args.model_name), "r") as file:
            self.args = json.load(file)
        try:
            with open("models/{}/config.json".format(self.args.model_name)) as json_file:
                self.model_params = json.load(json_file) 
            #self.model_params = vars(getattr(self.args, self.args.model_name))
        except:
            self.model_params = None       

def load_dataset():
    self.preproc.clean_dataset()
    df = pd.read_csv(self.args.post_preproc_data_path)
    df.drop(["Unnamed: 0", "Heightin", "Weightlbs", "patient_id"], axis=1, inplace=True)
    features = list(df.columns)[1:]
    train, test = train_test_split(df, test_size=0.2)
    y_train = train["value"]
    x_train = train.drop(axis=1, columns=["value"])
    y_test = test["value"]
    x_test = test.drop(axis=1, columns=["value"])
    trainset = (x_train, y_train)
    testset = (x_test, y_test)
    return features, trainset, testset


if __name__ == "__main__":
    features, trainset, testset = load_dataset()
    trainer = Trainer(args, features)
    print(trainset[0].type.unique())
    exit(0)
    # trainer.train(trainset, testset)
    # y_pred = trainer.predict(testset)
    # mse, mae, r2score = trainer.metrics(testset, y_pred)
    # # The mean squared error
    # print("Mean squared error: %.2f" % mse)
    # # The mean absolute error
    # print("Mean absolute error: %.2f" % mae)
    # # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f" % r2score)
    # trainer.save()
    trainer.load()
    print(trainer.predict(testset, inference=True))