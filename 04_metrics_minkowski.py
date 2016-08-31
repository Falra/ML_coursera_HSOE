from __future__ import division
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
housing = load_boston()
rows = housing.data.shape[0]
housing.data = scale(housing.data)
p_vars = np.linspace(1, 10, 200)
kf = cross_validation.KFold(rows, n_folds=5, shuffle=True, random_state=42)
max_acc, opt_p = float("-inf"), 0
for p in p_vars:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    cv_res = cross_validation.cross_val_score(neigh, housing.data, housing.target, cv=kf, scoring='mean_squared_error')
    cur_acc = np.max(cv_res)
    if cur_acc > max_acc:
        max_acc, opt_p = cur_acc, p
print('p = {:.2f}'.format(opt_p))
print('Max acc.: {:.2f}'.format(max_acc))
