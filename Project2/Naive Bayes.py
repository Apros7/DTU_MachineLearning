# exercise 7.4.4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

file = 'StandardizedDataFrameWithNansFilled.csv'
data = pd.read_csv(file)

data['Status'] = data['Status'].map({'Developed': 1, 'Developing': 0})
y = data['Status'].values
y=y.T
X = data.loc[:, 'Adult Mortality':'Schooling'].values

# Naive Bayes classifier parameters
alpha = 1.0 # pseudo-count, additive parameter (Laplace correction if 1.0 or Lidtstone smoothing otherwise)
fit_prior = True   # uniform prior (change to True to estimate prior from data)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

errors = []
k=0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
nb_classifier = MultinomialNB(alpha=alpha,
                                  fit_prior=fit_prior)
nb_classifier.fit(X_train, y_train)
y_est_prob = nb_classifier.predict_proba(X_test)
y_est = np.argmax(y_est_prob,1)

errors = np.sum(y_est!=y_test,dtype=float)/y_test.shape[0]

# Plot the classification error rate
print('Error rate: {0}%'.format(100*np.mean(errors)))

print('Ran Exercise 7.2.4')