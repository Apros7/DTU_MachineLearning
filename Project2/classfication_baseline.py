# https://towardsdatascience.com/calculating-a-baseline-accuracy-for-a-classification-model-a4b342ceb88f

from automation import Tester
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from collections import Counter
import torch

def baseline(x_train, x_test, y_train, y_test, func_var):
    most_common_class = max(set(tuple(row) for row in y_train), key=y_train.tolist().count)
    y_pred = [most_common_class] * len(y_test)
    f1 = f1_score(y_test, y_pred, average=func_var)
    return f1
  

if __name__ == "__main__":
    path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
    averages_to_test = ['micro', 'macro', 'weighted', 'samples']
    tester = Tester("StatusClassification", path_to_data, function_to_test = baseline, final_test = False, k = 10, vars_to_test=averages_to_test)