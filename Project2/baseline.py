from automation import Tester
import numpy as np

def baseline_model(x_train, x_test, y_train, y_test, func_var):
    y_train, y_test = np.array(y_train), np.array(y_test)
    estimate = np.mean(y_train)
    errors = (y_test - estimate)**2
    mse = sum(errors) / len(errors)
    return mse.tolist()

path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
tester = Tester("LifeExpectancyRegression", path_to_data, function_to_test = baseline_model, final_test = False, k = 10)