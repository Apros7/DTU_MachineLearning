import numpy as np
from automation import Tester
import matplotlib.pyplot as plt

def linear(X_train, X_test, y_train, y_test, lamb):
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

def linear_regression_with_W(X_train, X_test, y_train, y_test, lamb):
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
    return MSE.tolist(), W

if __name__ == "__main__":

    lambda_to_test = np.power(10.,range(-100,100))
    lambda_to_test = [float(x) for x in lambda_to_test]

    lambda_to_test = 1

    path_to_data = "/Users/william/Documents/University/civil engineering year 4/Semester 1 DTU/Introduction to machine learning/DTU_MachineLearning/Project2/StandardizedDataFrameWithNansFilled.csv"

    tester = Tester("LifeExpectancyRegression", path_to_data, function_to_test = linear_regression_with_W, final_test = False, k = 10, vars_to_test=lambda_to_test)
    W = list(tester.results.values())[0][0][1]
    x_cols = [tester.columns[3]] + tester.columns[5:]

    weights_values = [(w, x_col) for w, x_col in zip(W, x_cols)]

    categories = [item[1] for item in weights_values]
    values = [item[0] for item in weights_values]

    plt.figure(figsize=(15, 9))
    plt.subplots_adjust(bottom=0.35)
    plt.bar(categories, values)
    plt.xticks(rotation='vertical')
    plt.ylabel("Weight", fontsize=20)
    plt.title("Linear Regression Coefficients", fontsize=20)
    plt.savefig("Linear Coefficients.png", bbox_inches = "tight")
    plt.show()


  
