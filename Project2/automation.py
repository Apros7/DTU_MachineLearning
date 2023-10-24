
# Automatically split data, reserve for for validation, run the function on all the folds, return minimum accuracy
# generalization error and best model object

def regression_tester(function_to_test, final_test, k = 10):
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
    pass