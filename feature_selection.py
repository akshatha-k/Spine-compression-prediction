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
from args import get_args
import statsmodels.api as sm

np.random.seed(123)

class Preprocessing:
    def __init__():
        self.label_encoder = LabelEncoder()
        self.args = get_args()

    def clean_dataset():
        df = pd.read_csv(self.args.dataset_path)
        df['value'].replace('', np.nan, inplace=True)
        df = df[pd.notnull(df['value'])]
        df = df.reset_index(drop=True)
        df.drop(axis=1, columns=['Race', 'primary cause of death', 'secondary cause of death', 'Height (in)', 'Weight (lbs)', 'further diagnosis', 'Group', 'date of death'], inplace=True)
        df = df.drop(columns="Smoking")
        df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
        df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == "object" else x)
        new_df = df.loc[:, df.dtypes == object].apply(LabelEncoder().fit_transform)
        df = df.drop(
            columns=["patient_id", "level", "type", "experiment", "intervention", "Gender"]
        )
        result = pd.concat(
            [df, new_df],
            axis=1,
            join="outer",
            ignore_index=False,
            keys=None,
            levels=None,
            names=None,
            verify_integrity=False,
            copy=True,
        )
        result.to_csv(self.args.post_preproc_data_path)


    def col_correlation(df):
        corr = df.corr()
        # print(sns.heatmap(corr))
        # plt.show()
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
        selected_columns = df.columns[columns]
        df = df[selected_columns]
        return selected_columns


    def backwardElimination(x, Y, sl, columns):
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


    def col_pvalue(df, selected_columns):
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
    df = pd.read_csv(preproc.args.post_preproc_data_path)
    df.drop(["Unnamed: 0", "Heightin", "Weightlbs", "patient_id"], axis=1, inplace=True)
    selected_column = preproc.col_correlation(df)
    print(df[selected_column].columns)
    data = preproc.col_pvalue(df, selected_column)
    print(data.columns)