import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

df = pd.read_csv('gbm-data.csv')
data = df.values
X, y = data[:, 1:], data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
for l_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    break
    if l_rate != 0.2:
        continue
    L_test, L_train = [], []
    min_n, min_loss = 0, float('inf')
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=l_rate)
    clf.fit(X_train, y_train)
    sdfs_train = clf.staged_decision_function(X_train)
    sdfs_test = clf.staged_decision_function(X_test)
    for sdf_train, sdf_test in zip(sdfs_train, sdfs_test):
        converted_train = 1 / (1 + np.exp(sdf_train*(-1)))
        converted_test = 1 / (1 + np.exp(sdf_test*(-1)))
        test_loss = log_loss(y_test, converted_test)
        train_loss = log_loss(y_train, converted_train)
        L_test.append(test_loss)
        L_train.append(train_loss)
    print('TEST Loss = {:.2f} N = {}'.format(np.min(L_test), np.argmin(L_test)+1))
    print('TRAIN Loss = {:.2f} N = {}'.format(np.min(L_train), np.argmin(L_train)+1))
    plt.figure()
    plt.plot(L_test, 'r', linewidth=2)
    plt.plot(L_train, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

clf = RandomForestClassifier(n_estimators=37, random_state=241)
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_test)[:, 1]
test_loss = log_loss(y_test, pred)
print('Loss: {:.2f}'.format(test_loss))
