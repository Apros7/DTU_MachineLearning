# Projects for our Machine Learning & Data Mining Course
[Data](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who?resource=download) from Kaggle

1. Preliminary analysis of data
2. Machine Learning on Data

## Performance logging for Project 2

### Regression problem: Predicting Life Expectancy
The error returned from your function to Tester should be the MSE

| Type | Validation (final_test = False) | Test (final_test = True) |
|------|---------------------------------|---------------------------|
| NN (h=10) | 0.3065 | ---- |
| NN (h=50) | 0.1551 | ---- |
| NN (h=200) | 0.0992 | ---- |
| NN (h=500) | 0.0888 | 0.0914 |
| Linear Regression | 0.2251 | 0.2006 |

### Classification problem: Predicting Status
The error returned from your function to Tester should be the F1 (from sklean)

| Type | Validation (final_test = False) | Test (final_test = True) |
|------|---------------------------------|---------------------------|
| x | x | x |