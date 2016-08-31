from __future__ import division
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.preprocessing import scale
wine = pd.read_csv('wine.data.txt', header=None)
rows = wine.shape[0]
wine_target = wine.iloc[:, 0]
wine_data = wine.iloc[:, 1:]
wine_data = scale(wine_data)
kf = cross_validation.KFold(rows, n_folds=5, shuffle=True, random_state=42)
for k in xrange(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    print('K = {}'.format(k))
    cv_res = cross_validation.cross_val_score(neigh, wine_data, wine_target, cv=kf, scoring='accuracy')
    mean_acc = cv_res.mean()
    print('Mean acc.: {:.2f}'.format(mean_acc))
