import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


data_logistic_regression = [0.57 ,0.44 ,0.87 ,0.55 ,0.88 ,0.97 ,0.96 ,0.53 ,1 ,0.34]
data_ann = [0.76,0.42,0.88,0.57,0.79,0.96,0.98,0.56,0.99,0.81]
data_class_tree = [0.58,0.49,0.69,0.58,0.75,0.94,0.96,0.58,0.98,0.58]
data_knn = [0.73,0.45,0.89,0.57,0.92,0.78,0.95 ,0.60 ,0.95,0.84 ]
data_baseline = [0 ,0 ,0 ,0 ,0 ,0 ,0.34 ,0.37 ,0.36 ,0.20 ]


def compute_confidence_interval(lst):
    sample_mean = np.mean(lst)
    sample_std = np.std(lst, ddof=1)
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * (sample_std / np.sqrt(len(lst)))
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    return sample_mean, margin_of_error, confidence_interval

def plot(lst, string, x):
    sample_mean, margin_of_error, confidence_interval = compute_confidence_interval(lst)
    plt.errorbar(x=x, y=sample_mean, yerr=margin_of_error, fmt='o', capsize=5, label=string)

plot(data_ann, "ANN", 0)
plot(data_logistic_regression, "Logistic regression", 1)
plot(data_baseline, "Baseline", 2)
plot(data_knn, "KNN", 3)
plot(data_class_tree, "Classification tree", 4)


plt.xticks([])  # Hide x-axis label
plt.ylabel('Sample Mean')
plt.title('95% Confidence Interval with Error Bars')
plt.legend()
plt.show()
