from automation import Tester
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def LogReg(x_train, x_test, y_train, y_test, func_var):

    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)

    logistic_regression_model = LogisticRegression(C=1/func_var, max_iter=1000)
    logistic_regression_model.fit(x_train, y_train)
    y_test_est = logistic_regression_model.predict(x_test).T

    f1score = f1_score(y_test, y_test_est)
    return f1score, logistic_regression_model.coef_
  

if __name__ == "__main__":
    path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
    lambda_to_test = [np.power(10., 2)]
    lambda_to_test = [float(x) for x in lambda_to_test]
    # h_to_test = 8
    print(lambda_to_test)
    tester = Tester("StatusClassification", path_to_data, function_to_test = LogReg, final_test = False, k = 10, vars_to_test=lambda_to_test)
    W = list(tester.results.values())[0][0][1][0]
    print(W)
    x_cols = tester.columns[4:]

    weights_values = [(w, x_col) for w, x_col in zip(W, x_cols)]

    categories = [item[1] for item in weights_values]
    values = [item[0] for item in weights_values]

    plt.figure(figsize=(15, 9))
    plt.subplots_adjust(bottom=0.35)
    # plt.ylim(-1000, 1000)
    plt.bar(categories, values)
    plt.xticks(rotation='vertical')
    plt.ylabel("Weight", fontsize=20)
    plt.title("Logistic Regression Coefficients", fontsize=20)
    plt.savefig("Linear Coefficients.png", bbox_inches = "tight")
    plt.show()