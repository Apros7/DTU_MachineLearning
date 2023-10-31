
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

    def __init__(self, problem_type : str, path_to_data : str, function_to_test, final_test : bool = False, k : int = 10, function_variable = None):
        avaliable_problem_types = ["LifeExpectancyRegression", "StatusClassification"]
        if problem_type not in avaliable_problem_types: raise ValueError(f"Problem type not in {avaliable_problem_types}")
        self.problem_function = {"LifeExpectancyRegression": self._set_life_expectancy, "StatusClassification": self._set_status_classification}
        self.func_to_test, self.path_to_data, self.final_test, self.k, self.func_var = function_to_test, path_to_data, final_test, k, function_variable
        self.results = []
        self.problem_function[problem_type]()

    def _load_data(self): self.data = pd.read_csv(self.path_to_data); self.columns = list(self.data.columns); self._fix_data()
    def _fix_data(self): self.data["Status"] = [1 if stat == "Developed" else 0 for stat in self.data["Status"]]
    def _set_data_x(self, x_columns): self.data_x = torch.tensor([[x[index] for x in x_columns] for index in range(len(x_columns[0]))])
    def _set_data_y(self, y_column): self.data_y = torch.tensor(self.data[y_column].to_list())
    def _set_data_props(self, x_cols, y_col): self._set_data_x(x_cols); self._set_data_y(y_col)
    def _set_data_folds(self): sublist_size = len(self.data) // self.k; self.data_folds = [list(range(len(self.data)))[i:i + sublist_size] for i in range(0, len(self.data), sublist_size)]
    def _unnest_lst(self, lst): return [item for sublist in lst for item in sublist]
    def _get_fold_combs_with_final_test(self): self.fold_combs = [(self._unnest_lst(self.data_folds[:i] + self.data_folds[i+1:]), self.data_folds[i]) for i in range(self.k)] # (train, test)
    def _get_fold_combs(self): self.fold_combs = [(self._unnest_lst(self.data_folds[:i] + self.data_folds[i+2:]), self.data_folds[i+1 if i+1 < self.k else 0], self.data_folds[i]) for i in range(self.k)] # (train, val, test)
    def _set_folds(self): self._set_data_folds(); self._get_fold_combs_with_final_test() if self.final_test else self._get_fold_combs()
    def _get_generalization_error(self): print(f"Generalization (MSError) error is: {sum(self.results) / len(self.results)}")
    def _test_all_folds(self): self.results = [self.func_to_test(self.data_x[fold[0]], self.data_x[fold[1]], self.data_y[fold[0]], self.data_y[fold[1]], self.func_var) for fold in tqdm(self.fold_combs, desc="Training and testing...")]

    def _set_life_expectancy(self): 
        self._load_data()
        x_cols = [self.data[column].to_list() for column in [self.columns[3]] + self.columns[5:]]
        y_col = "Life expectancy "
        self._set_data_props(x_cols, y_col)
        self._set_folds()
        self._test_all_folds()
        self._get_generalization_error()

    def _set_status_classification(self): # not done
        x_cols = []
        y_col = ""