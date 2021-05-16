import pandas as pd
import numpy as np

df = pd.read_csv("data/final_data.csv")
df['value'].replace('', np.nan, inplace=True)
df = df[pd.notnull(df['value'])]
df = df.reset_index(drop=True)
df.drop(axis=1, columns=['Race', 'primary cause of death', 'secondary cause of death', 'Height (in)', 'Weight (lbs)', 'further diagnosis', 'Group', 'date of death'], inplace=True)
print(df.shape)
df.to_csv("final_processed_data.csv", index=False)