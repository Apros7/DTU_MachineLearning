# exercise 6.3.2



from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread

file = 'StandardizedDataFrameWithNansFilled.csv'
data = pd.read_csv(file)

data['Status'] = data['Status'].map({'Developed': 1, 'Developing': 0})
y = data['Status'].values
y=y.T
X = data.loc[:, 'Adult Mortality':'Schooling'].values
N, M = X.shape


# Maximum number of neighbors
L=40
Folds = 500
CV = model_selection.KFold(Folds)

errors = np.zeros((N,L))
i=0
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,Folds))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])

    i+=1
    
# Plot the classification error rate
plt.figure()
plt.plot(100*sum(errors,0)/N)
plt.xlabel('Number of neighbors')
plt.ylabel('Classification error rate (%)')
plt.show()

print('Ran Exercise 6.3.2')