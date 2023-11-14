import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data_ann = [0.18,0.26 ,0.12 ,0.20 ,0.15 ,0.17 ,0.16 ,0.29 ,0.17 ,0.18 ]
data_linear_regression = [0.18 ,0.27 ,0.14 ,0.21 ,0.21 ,0.19 ,0.24 ,0.31 ,0.20 ,0.18 ]
data_baseline = [0.86 ,1.57 ,0.82 ,0.99 ,0.75 ,1.08 ,0.99 ,0.88 ,1.21 ,0.78 ]


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
plot(data_linear_regression, "Linear Regression", 1)
plot(data_baseline, "Baseline", 2)


plt.xticks([])  # Hide x-axis label
plt.ylabel('Sample Mean')
plt.title('95% Confidence Interval with Error Bars')
plt.legend()
plt.show()
