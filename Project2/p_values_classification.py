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

data_logistic_regression = [0.57 ,0.44 ,0.87 ,0.55 ,0.88 ,0.97 ,0.96 ,0.53 ,1 ,0.34]
data_ann = [0.76,0.42,0.88,0.57,0.79,0.96,0.98,0.56,0.99,0.81]
data_class_tree = [0.58,0.49,0.69,0.58,0.75,0.94,0.96,0.58,0.98,0.58]
data_knn = [0.73,0.45,0.89,0.57,0.92,0.78,0.95 ,0.60 ,0.95,0.84 ]
data_baseline = [0 ,0 ,0 ,0 ,0 ,0 ,0.34 ,0.37 ,0.36 ,0.20 ]

datas = [data_logistic_regression, data_ann, data_class_tree, data_knn, data_baseline]
strings = ["LR", "Ann", "class tree", "knn", "baseline"]


K = 10
alpha = 0.05
rho = 1/K

for i in range(len(datas)):
    for j in range(len(datas)):
        if i == j: continue
        r = np.array(datas[i]) - np.array(datas[j])
        print(r)
        vs_string = strings[i] + " vs " + strings[j]
        p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)
        print(f"\n\nP for {vs_string}")
        print(p_setupII)
        print(CI_setupII)
