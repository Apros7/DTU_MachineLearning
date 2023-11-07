from ann_classification import ann
from KNN_tester import KNN
from classfication_baseline import baseline

from LogisticRegressionTester import LogReg
from ClassificationTreeTester import ClassTree

from automation import Tester

functions_to_compare = [ann, KNN, baseline]
functions_to_compare = [LogReg, ClassTree, baseline]

nn_vars_to_test = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
linear_regression_vars_to_test = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# nn_vars_to_test = [1, 2]
# linear_regression_vars_to_test = [0.1, 1]

all_vars_to_test = [nn_vars_to_test, linear_regression_vars_to_test, [0]]

path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
tester = Tester("LifeExpectancyRegression", path_to_data, function_to_test = functions_to_compare, final_test = False, 
                k = 10, cross_validation_level = 2, vars_to_test=all_vars_to_test)