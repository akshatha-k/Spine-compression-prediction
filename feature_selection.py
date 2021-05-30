import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from args import get_args
import statsmodels.api as sm

np.random.seed(123)


class Preprocessing:
    def __init__(self, args):
        self.label_encoder = {}
        self.args = args

    def get_unique_values(self):
        df = pd.read_csv(self.args.dataset_path)
        types = df.type.unique().tolist()
        intervention = df.intervention.unique().tolist()
        experiment = df.experiment.unique().tolist()
        return [types, intervention, experiment]

    def clean_dataset(self):
        df = pd.read_csv(self.args.dataset_path)
        df["value"].replace("", np.nan, inplace=True)
        df = df[pd.notnull(df["value"])]
        df = df.reset_index(drop=True)
        df.drop(
            axis=1,
            columns=[
                "Race",
                "patient_id",
                "primary cause of death",
                "secondary cause of death",
                "Height (in)",
                "Weight (lbs)",
                "further diagnosis",
                "Group",
                "date of death",
            ],
            inplace=True,
        )
        df = df.drop(columns="smoking")
        df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
        for x in ["level", "type", "experiment", "intervention", "Gender"]:
            df[x] = df[x].str.strip().str.lower()
        for x in ["level", "type", "experiment", "intervention", "Gender"]:
            self.label_encoder[x] = LabelEncoder()
            self.label_encoder[x] = self.label_encoder[x].fit(df[x])
            df[x] = self.label_encoder[x].transform(df[x])
        df.to_csv(self.args.post_preproc_data_path)

    def col_correlation(self, df):
        corr = df.corr()
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
        selected_columns = df.columns[columns]
        df = df[selected_columns]
        return selected_columns

    def backwardElimination(self, x, Y, sl, columns):
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(Y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if regressor_OLS.pvalues[j].astype(float) == maxVar:
                        x = np.delete(x, j, 1)
                        columns = np.delete(columns, j)

        regressor_OLS.summary()
        return x, columns

    def col_pvalue(self, df, selected_columns):
        SL = 0.05  # 0.6 < p value for intervention < 0.7
        selected_columns = selected_columns[1:].values
        data_modeled, selected_columns = backwardElimination(
            df.iloc[:, 1:].values, df.iloc[:, 0].values, SL, selected_columns
        )
        result = pd.DataFrame()
        result["value"] = df.iloc[:, 0]
        data = pd.DataFrame(data=data_modeled, columns=selected_columns)
        return data


if __name__ == "__main__":
    preproc = Preprocessing()
    preproc.clean_dataset()
    df = pd.read_csv(preproc.args.post_preproc_data_path)
    selected_column = preproc.col_correlation(df)
    print(df[selected_column].columns)
    data = preproc.col_pvalue(df, selected_column)
    print(data.columns)
