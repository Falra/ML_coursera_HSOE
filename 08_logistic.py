import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
k = 0.1
C = 10
tol = 1e-6
df_data = pd.read_csv('data-logistic.csv', header=None)
Y, X = df_data.iloc[:, 0], df_data.iloc[:, 1:]
size = Y.shape[0]
w1_old, w2_old = 0, 0
X1 = X.iloc[:, 0]
X2 = X.iloc[:, 1]
for i in range(0, 10000):
    w1_new = w1_old+k/size*np.sum(Y*X1*(1 - 1/(1 + np.exp(-Y*(w1_old*X1+w2_old*X2)))))-k*C*w1_old
    w2_new = w2_old+k/size*np.sum(Y*X2*(1 - 1/(1 + np.exp(-Y*(w1_old*X1+w2_old*X2)))))-k*C*w2_old
    dist = np.power(np.power(w1_old - w1_new, 2) + np.power(w2_old - w2_new, 2), 0.5)
    w1_old, w2_old = w1_new, w2_new
    if dist < tol:
        break
pred = 1 / (1 + np.exp(-w1_old*X1 - w2_old * X2))
acc = roc_auc_score(Y, pred)
print("Accuracy {:.3f}".format(acc))