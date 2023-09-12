# %% Imports
import numpy as np
import pandas as pd
import os 

# %% Loading data
# Load the csv data using the Pandas library
filename = 'Life Expectancy Data.csv'
df = pd.read_csv(filename)

# %%
attributeNames = np.asarray(df.columns)

for attribute in attributeNames:
    nan = df[attribute].isnull().values.any()
    if nan == True:
        amount = df[attribute].isnull().sum()
        print(attribute + " has " + str(amount) + " nan's" )

N, M = df.shape

# %%

new_df = df.dropna().reset_index(drop=True)

N, M = new_df.shape

#print(N,M)

#print(new_df)
#new_df.to_csv("Life Expectancy Data With Dropped Rows.csv")

# %%
mean = df["Measles "].mean()
sd = df["Measles "].std()
print("Mean = " + str(mean) +"\n"+ "SD = " + str(sd))
