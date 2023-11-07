# https://towardsdatascience.com/calculating-a-baseline-accuracy-for-a-classification-model-a4b342ceb88f

from automation import Tester
import numpy as np
from sklearn.metrics import f1_score
from collections import Counter
import torch

def baseline(x_train, x_test, y_train, y_test, func_var):

    y_test = np.argmax(y_test, axis=1).tolist()
    y_train = np.argmax(y_train, axis=1).tolist()

    most_frequent_class = Counter(y_train).most_common(1)[0][0]
    y_pred = [most_frequent_class] * len(y_test)

    f1score = f1_score(np.array(y_test), np.array(y_pred))
    print(f1score)

    return f1score
  

if __name__ == "__main__":
    path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
    tester = Tester("StatusClassification", path_to_data, function_to_test = baseline, final_test = False, k = 10)