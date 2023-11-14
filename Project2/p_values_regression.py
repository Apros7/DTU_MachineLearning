import numpy as np
import scipy.stats as st

def correlated_ttest(r, rho, alpha=0.05):
    rhat = np.mean(r)
    shat = np.std(r)
    J = len(r)
    sigmatilde = shat * np.sqrt(1 / J + rho / (1 - rho))

    CI = st.t.interval(1 - alpha, df=J - 1, loc=rhat, scale=sigmatilde)  # Confidence interval
    p = 2*st.t.cdf(-np.abs(rhat) / sigmatilde, df=J - 1)  # p-value
    return p, CI

data_ann = np.array([0.18,0.26 ,0.12 ,0.20 ,0.15 ,0.17 ,0.16 ,0.29 ,0.17 ,0.18 ])
data_linear_regression = np.array([0.18 ,0.27 ,0.14 ,0.21 ,0.21 ,0.19 ,0.24 ,0.31 ,0.20 ,0.18 ])
data_baseline = np.array([0.86 ,1.57 ,0.82 ,0.99 ,0.75 ,1.08 ,0.99 ,0.88 ,1.21 ,0.78 ])

strings = ["ANN vs LR", "LR vs Baseline", "ANN vs Baseline"]
rs = [data_ann - data_linear_regression, data_linear_regression - data_baseline, data_ann - data_baseline]


K = 10
alpha = 0.05
rho = 1/K

for i in range(len(strings)):
    r = rs[i]
    string = strings[i]
    p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)
    print(f"P for {string}")
    print(p_setupII)
    print(CI_setupII)
