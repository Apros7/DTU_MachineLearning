
"""
## How to import:
```
from automation import regression_tester, classification_tester
```
if you get a path error use:
```
import sys
sys.path.append()
from automation import regression_tester, classification_tester
```
where the path should be your path, this is mine :-)

## How to use:
You need to give it the following parameters:
- your function (explained below)
- final_test (when you are doing the final test, set this to True to use validation, else set to False)

your function has to take in: (X_train, X_test, Y_train, Y_test)

Optional parameters:
- k (default: 10)

"""

# Automatically split data, reserve for for validation, run the function on all the folds, return minimum accuracy
# generalization error and best model object

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

class Tester():

    def __init__(self, problem_type : str, path_to_data : str, function_to_test : function, final_test : bool = False, k : int = 10):
        avaliable_problem_types = ["LifeExpectancyRegression", "StatusClassification"]
        if problem_type not in avaliable_problem_types: raise ValueError(f"Problem type not in {avaliable_problem_types}")
        self.problem_function = {"LifeExpectancyRegression": self._set_life_expectancy, "StatusClassification": self._set_status_classification}
        self.problem_function[problem_type]()
        self.results = []
        self.func_to_test, self.path_to_data, self.final_test, self.k = function_to_test, path_to_data, final_test, k

    def _set_data(self): self.data = pd.read_csv(self.path_to_data, sep=";"); self.columns = list(self.data.columns)
    def _fix_data(self): self.data["Status"] = [1 if stat == "Developed" else 0 for stat in self.data["Status"]]
    def _set_data_x(self, x_columns): self.data_x = torch.tensor([[x[index] for x in x_columns] for index in range(len(x_columns[0]))])
    def _set_data_y(self, y_column): self.data_y = torch.tensor(self.data[y_column].to_list())
    def _set_data_props(self, x_cols, y_col): self._set_data(); self._fix_data(); self._set_data_x(x_cols); self._set_data_y(y_col)
    def _set_data_folds(self): sublist_size = len(self.data) // self.k; self.data_folds = [range(len(self.data))[i:i + sublist_size] for i in range(0, self.data, sublist_size)]
    def _get_fold_combs_with_final_test(self): self.fold_combs = [(self.data_folds[:i] + self.data_folds[i+1:], self.data_folds[i]) for i in range(self.k)] # (train, test)
    def _get_fold_combs(self): self.fold_combs = [(self.data_folds[:i] + self.data_folds[i+2:], self.data_folds[i+1 if i+1 < self.k else 0], self.data_folds[i]) for i in range(self.k)] # (train, val, test)
    def _set_folds(self): self._set_data_folds(); self._get_fold_combs_with_final_test() if self.final_test else self._get_fold_combs()
    def _indexes_to_values(self, lst, indexes): return [lst[i] for i in indexes]
    def _get_generalization_error(self): print(f"Generalization error is: {sum(self.results) / len(self.results)}")

    def _test_all_folds(self): 
        self.results = [self.func_to_test(
            self._indexes_to_values(self.data_x, fold[0]), self._indexes_to_values(self.data_y, fold[0]),
            self._indexes_to_values(self.data_x, fold[1]), self._indexes_to_values(self.data_y, fold[1])) 
        for fold in tqdm(self.fold_combs)]

    def _set_life_expectancy(self): 
        x_cols = [self.data[column].to_list() for column in [self.columns[3]] + self.columns[5:]]
        y_col = "Life expectancy "
        self._set_data_props(x_cols, y_col)
        self._set_folds()
        self._test_all_folds()
        self._get_generalization_error()

    def _set_status_classification(self): # not done
        x_cols = []
        y_col = ""



if __name__ == "__main__":
    path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
    tester = Tester("regression", path_to_data, final_test = False, k = 10)