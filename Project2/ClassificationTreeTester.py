from automation import Tester

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib.image import imread
from sklearn.metrics import f1_score

def ClassTree(x_train, x_test, y_train, y_test, func_var):

    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)

    criterion=func_var
    dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=100)
    dtc = dtc.fit(x_train,y_train)
    y_pred = dtc.predict(x_test)

    # Calculate F1-score
    f1 = f1_score(y_test, y_pred)
    return f1


if __name__ == "__main__":
    path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
    h_to_test = ["gini", "entropy", "log_loss"]
    # h_to_test = 8
    print(h_to_test)
    tester = Tester("StatusClassification", path_to_data, function_to_test = ClassTree, final_test = False, k = 10, vars_to_test=h_to_test)