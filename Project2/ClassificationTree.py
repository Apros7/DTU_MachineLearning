import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib.image import imread

file = 'StandardizedDataFrameWithNansFilled.csv'
data = pd.read_csv(file)

data['Status'] = data['Status'].map({'Developed': 1, 'Developing': 0})
y = data['Status'].values
y=y.T
X = data.loc[:, 'Adult Mortality':'Schooling'].values
attributeNames = data.columns[1:-1].tolist()

# classTree_to_test = ["gini", "entropy", "log_loss"]
criterion='entropy'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=100)
dtc = dtc.fit(X,y)

fname='Classification tree: ' + criterion + '.png'
fig = plt.figure(figsize=(12,12),dpi=300) 
_ = tree.plot_tree(dtc, filled=False,feature_names=attributeNames)
plt.savefig(fname)
plt.close() 

fig = plt.figure()
plt.imshow(imread(fname))
plt.axis('off')
plt.box('off')
plt.show()