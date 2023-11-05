import numpy as np
from automation import Tester

def linear_regression(X_train, X_test, y_train, y_test, lamb):
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the value of lambda, on entire training set
    N, M = X_train.shape
    lambdaI = lamb * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    W = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with lambda
    Error_test = ((y_test-X_test @ W)**2).sum(axis=0)/y_test.shape[0]
    # Whatever you want
    MSE = Error_test
    return MSE.tolist()

lambda_to_test = np.power(10.,range(-100,100))
lambda_to_test = [float(x) for x in lambda_to_test]

path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"

tester = Tester("LifeExpectancyRegression", path_to_data, function_to_test = linear_regression, final_test = False, k = 10, vars_to_test=lambda_to_test)

