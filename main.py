import os
import sys
from train import Trainer
import itertools
import re
from feature_selection import Preprocessing
import pandas as pd 
from train import Trainer
from args import get_args
from sklearn.model_selection import train_test_split


'''

'''
def get_values():
    values = {'Age':66, 'Heightcm':175.26, 'Weightkg':81.65, 'BMI':26.58,'level':'L1L2','Gender':'m'}
    return pd.DataFrame(values, index=[0])

def create_dataframe(preproc, scaler):
    unique_vals = preproc.get_unique_values()
    combinations = list(itertools.product(*unique_vals))
    df2 = pd.DataFrame(combinations, columns =['type', 'intervention', 'experiment'])
    df1 = get_values()
    df2['tmp']=1
    df1['tmp']=1
    df = pd.merge(df1, df2, on=['tmp'])
    df = df.drop('tmp', axis=1)
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    for x in ["level", "type", "experiment", "intervention", "Gender"]:
        df[x]=df[x].str.strip().str.lower()
        df[x]= preproc.label_encoder[x].transform(df[x])
    df = scaler.transform(df)
    # print(df)
    # exit(0)
    return df 

def load_dataset():
    #preproc.clean_dataset()
    df = pd.read_csv(args.post_preproc_data_path)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    features = list(df.columns)[1:]
    train, test = train_test_split(df, test_size=0.2)
    y_train = train["value"]
    x_train = train.drop(axis=1, columns=["value"])
    y_test = test["value"]
    x_test = test.drop(axis=1, columns=["value"])
    trainset = (x_train, y_train)
    testset = (x_test, y_test)
    return features, trainset, testset

if __name__=="__main__":
    args = get_args()
    # f = open("myfile.txt", "a")
    # f.write(sys.argv)
    # f.close()
    #sys.stdout.write("Hola")
    # strlist = {"abc":5, "pqr":10}
    print(args)
    '''
    ##Get input from Unity
    features, trainset, testset = load_dataset()
    trainer = Trainer(args, features)
    trainer.load()
    df=create_dataframe(trainer.preproc, trainer.scaler)
    y_pred = trainer.predict(df, inference= True)
    print(y_pred)
    exit(0)
    for x in ["level", "type", "experiment", "intervention", "Gender"]:
        df[x]=trainer.preproc.label_encoder[x].inverse_transform(df[x])
    # df = trainer.preproc.label_encoder.inverse_transform(df)
    df['values']=y_pred 
    print(df)
    exit(0)
    df = df.groupby(["Age","Heightcm","Weightkg","BMI","level", "type", "intervention", "Gender"], sort= True).mean()
    #df.drop("experiment", axis=1, inplace=True)
    print(df)
    '''