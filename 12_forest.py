import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

df = pd.read_csv('abalone.csv')
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
df_x, df_y = df.iloc[:, :-1], df.iloc[:, -1]
found = 0
cv = KFold(df_x.shape[0], n_folds=5, shuffle=True, random_state=1)
N = range(1, 51)
Acc = []
for n in N:
    clf = RandomForestRegressor(n_estimators=n, random_state=1)
    cv_res = cross_val_score(clf, df_x, df_y, cv=cv, scoring='r2')
    mean_acc = cv_res.mean()
    Acc.append(mean_acc)
    print('N = {} R2 = {:.3f}'.format(n, mean_acc))
    if mean_acc >= 0.52 and not found:
        print('Acc.: {:.3f}'.format(mean_acc))
        print('Trees: {:}'.format(n))
        found = 1
plt.plot(N, Acc)
plt.show()
