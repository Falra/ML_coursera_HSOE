import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
test_data = pd.read_csv('perceptron-test.csv', header=None)
train_data = pd.read_csv('perceptron-train.csv', header=None)
test_y, test_x = test_data.iloc[:, 0], test_data.iloc[:, 1:]
train_y, train_x = train_data.iloc[:, 0], train_data.iloc[:, 1:]
clf = Perceptron(random_state=241)
clf.fit(train_x, train_y)
predictions = clf.predict(test_x)
acc_before = accuracy_score(test_y, predictions)
print('Acc. before: {:.3f}'.format(acc_before))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)
clf.fit(X_train_scaled, train_y)
predictions = clf.predict(X_test_scaled)
acc_after = accuracy_score(test_y, predictions)
print('Acc. after: {:.3f}'.format(acc_after))
print('Acc. diff: {:.3f}'.format(acc_after-acc_before))
