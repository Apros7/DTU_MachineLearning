
"""
Your function should have the following attributes for regression:
- Input should be: x_train, x_test, y_train, y_test, func_var
- Output 1 element, which is the MSE. 
    If multiple, return as tuple. First element should be MSE. The other elements can be found using tester.results
"""

"""
To do:
- [x] Be able to do single level cross validation
- [x] Be able to take in parameter input as list
- [ ] Be able to do 2 level cross validation
- [ ] When doing 2 level cross validation produce a table with outputs
- [ ] Make it work for classification provlem
"""

# Automatically split data, reserve for validation, run the function on all the folds, return minimum accuracy
# generalization error and best model object

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

class Tester():

    def __init__(
        self, problem_type : str, 
        path_to_data : str, 
        function_to_test, 
        final_test : bool = False, 
        k : int = 10, 
        vars_to_test = None,
        cross_validation_level : 1 or 2 = 1,
        ):
        avaliable_problem_types = ["LifeExpectancyRegression", "StatusClassification"]
        if problem_type not in avaliable_problem_types: raise ValueError(f"Problem type not in {avaliable_problem_types}")
        self.problem_function = {"LifeExpectancyRegression": self._set_life_expectancy, "StatusClassification": self._set_status_classification}
        self.func_to_test, self.path_to_data, self.final_test, self.k, self.cv_lvl = function_to_test, path_to_data, final_test, k, cross_validation_level
        self.func_vars = [vars_to_test] if type(vars_to_test) is not list else vars_to_test
        self.results = []
        self.problem_function[problem_type]()

    def _load_data(self): self.data = pd.read_csv(self.path_to_data); self.columns = list(self.data.columns); self._fix_data()
    def _fix_data(self): self.data["Status"] = [1 if stat == "Developed" else 0 for stat in self.data["Status"]]
    def _set_data_x(self, x_columns): self.data_x = torch.tensor([[x[index] for x in x_columns] for index in range(len(x_columns[0]))])
    def _set_data_y(self, y_column): self.data_y = torch.tensor(self.data[y_column].to_list())
    def _set_data_props(self, x_cols, y_col): self._set_data_x(x_cols); self._set_data_y(y_col)
    def _set_init_data_folds(self): sublist_size = len(self.data) // self.k; self.data_folds = [list(range(len(self.data)))[i:i + sublist_size] for i in range(0, len(self.data), sublist_size)]
    def _unnest_lst(self, lst): return [item for sublist in lst for item in sublist]
    def _get_fold_combs_without_val(self): self.fold_combs = [(self._unnest_lst(self.data_folds[:i] + self.data_folds[i+1:]), self.data_folds[i]) for i in range(self.k)] # (train, test)
    def _get_fold_combs_with_val(self): self.fold_combs = [(self._unnest_lst(self.data_folds[:i] + self.data_folds[i+2:]), self.data_folds[i+1 if i+1 < self.k else 0], self.data_folds[i]) for i in range(self.k)] # (train, val, test)
    def _set_folds(self): self._set_init_data_folds(); self._get_fold_combs_without_val() if self.final_test or self.cv_lvl == 2 else self._get_fold_combs_with_val()
    def _print_best_performer(self): print("".join(["-"]*40), f"\nBest performer is: {min(self.error.keys(), key=lambda x: self.error[x])}\n", sep="\n")
    def _print_generalization_table(self): print("Func var: |   Generalization error", "".join(["-"]*40), *[f"{k}   \t|    {v}" for k, v in self.error.items()], sep="\n"); self._print_best_performer()
    def _display(self): self._print_generalization_table() if len(self.error) > 1 else print("Generalization error is: ", list(self.error.values())[0])
    def _get_generalization_error(self): self.error = {k: sum(v) / len(v) for k, v in self.accuracies.items()}
    def _fix_results(self): self.accuracies = {k: self._check_results(v) for k, v in self.results.items()}
    def _check_results(self, results): return [float(r[0]) if type(r) in [list, tuple] and len(r) > 1 else r for r in results]
    def _test_folds_and_save_error(self): self._test_all_folds(); self._fix_results(); self._get_generalization_error(); self._display()
    def _test_all_folds(self): 
        self.results = {}
        for func_var in tqdm(self.func_vars, desc=f"Training and testing all function variables for {self.k} folds..."):
            self.results[func_var] = [self.func_to_test(self.data_x[fold[0]], self.data_x[fold[1]], self.data_y[fold[0]], self.data_y[fold[1]], func_var) for fold in self.fold_combs]
        return self.results

    def _set_life_expectancy(self): 
        # Still need to implement printing table and 2-level cross validation
        self._load_data()
        x_cols = [self.data[column].to_list() for column in [self.columns[3]] + self.columns[5:]]
        y_col = "Life expectancy "
        self._set_data_props(x_cols, y_col)
        self._set_folds()
        if self.cv_lvl == 1: self._test_folds_and_save_error()
        else: pass

    def _set_status_classification(self): # not done
        x_cols = []
        y_col = ""