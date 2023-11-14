from ClassificationTreeTester import ClassTree

from automation import Tester
import numpy as np

nn_vars_to_test = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
ks_to_test = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
lambda_to_test = np.power(10.,range(-30,30))
lambda_to_test = [float(x) for x in lambda_to_test]
classTree_to_test = ["gini", "entropy", "log_loss"]
baseline_to_test = ['micro', 'macro', 'weighted', 'samples']

all_vars_to_test1 = [nn_vars_to_test, ks_to_test, baseline_to_test]
all_vars_to_test2 = [lambda_to_test, classTree_to_test, baseline_to_test]

path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
tester = Tester("StatusClassification", path_to_data, function_to_test = ClassTree, final_test = False, 
                k = 10, cross_validation_level = 1, vars_to_test=classTree_to_test)
print(tester.results)