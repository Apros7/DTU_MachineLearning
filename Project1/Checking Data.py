# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import os 

# %% Loading data
# Load the csv data using the Pandas library
filename = 'LifeExpectancyData.csv'
df = pd.read_csv(filename)

# %%
attributeNames = np.asarray(df.columns)

for attribute in attributeNames:
    nan = df[attribute].isnull().values.any() #what does this line of code do
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

# %% Mean and Standard Deviation
        
mean = df["Measles "].mean()
sd = df["Measles "].std()
print("Mean = " + str(mean) +"\n"+ "SD = " + str(sd))

#%%
filename = 'LifeExpectancyData.csv'
df = pd.read_csv(filename)
df.fillna(df.mean(), inplace=True) # Fills the columns with nan values with the mean of that column

#One problem with this method, is can be very inaccurate for some countries. 
#For example it says Antigua and Barbuda has a popluation of 10 million when
#it is actually 90,000

attributeNames = np.asarray(df.columns)


country = df.iloc[:, 0].tolist()
year = df.iloc[:,1].tolist()
develop = df.iloc[:,2].tolist()

countryNames = sorted(set(country)) # Creates a list of all the variable types from classLabels
countryDict = dict(zip(countryNames,range(193))) # turns that list into a dictionary

yearNames = sorted(set(year))
yearDict = dict(zip(yearNames,range(16))) 

developNames = sorted(set(develop))
developDict = dict(zip(developNames,range(2))) #0 for developed; 1 for developing

country = np.asarray([countryDict[value] for value in country])
year = np.asarray([yearDict[value] for value in year])
develop = np.asarray([developDict[value] for value in develop])

C = len(country)
Ye = len(year)
D = len(develop) 

data = df.iloc[:, 3:23].values
X = np.empty((2938, 19))
X[:, :data.shape[1]] = data


#%% Mean 

M = X - np.ones((C,1))*X.mean(axis=0)
U,S,Vt = svd(M,full_matrices=False)
V = Vt.T

# Project the centered data onto principal component space
Z = M @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data

for c in range(C):
    # select indices belonging to class c:
    class_mask = country==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
#plt.legend(countryNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()

#maybe split for developed and developing countries and even split by continent?
