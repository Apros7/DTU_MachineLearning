import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

file = 'StandardizedDataFrameWithNansFilled.csv'
data = pd.read_csv(file)

data['Status'] = data['Status'].map({'Developed': 1, 'Developing': 0})
y = data['Status'].values
y=y.T
X = data.loc[:, 'Adult Mortality':'Schooling'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

N, M = X.shape
classNames = ['Developed', 'Developing']
C = len(classNames)


#Plot the training data points (color-coded) and test data points.
plt.figure(1)
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plt.plot(X_train[class_mask,2], X_train[class_mask,3], styles[c])


# K-nearest neighbors
K=5

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=1
metric = 'minkowski'
metric_params = {} # no parameters needed for minkowski



# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plt.plot(X_test[class_mask,2], X_test[class_mask,3], styles[c], markersize=8)
    plt.plot(X_test[class_mask,2], X_test[class_mask,3], 'kx', markersize=7)
plt.title('Data classification - KNN');

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
plt.figure(2)
plt.imshow(cm, cmap='binary', interpolation='None');
plt.colorbar()
plt.xticks(range(C)); plt.yticks(range(C));
plt.xlabel('Predicted class'); plt.ylabel('Actual class');
plt.title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

plt.show()

print(f"f1 Score: {f1_score(y_test, y_est)}")


