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

    K=func_var

    f1score = f1_score(y_test, y_test)
    return f1score
  

if __name__ == "__main__":
    path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
    lambda_to_test = [3, 4, 5, 6, 7, 8, 9, 10]
    # h_to_test = 8
    print(lambda_to_test)
    tester = Tester("StatusClassification", path_to_data, function_to_test = LogReg, final_test = False, k = 10, vars_to_test=lambda_to_test)