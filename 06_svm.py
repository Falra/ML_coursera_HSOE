import pandas as pd
from sklearn.svm import SVC
svm_data = pd.read_csv('svm-data.csv', header=None)
labels, features = svm_data.iloc[:, 0], svm_data.iloc[:, 1:]
clf = SVC(C=100000, random_state=241)
clf.fit(features, labels)
print(clf.support_+1)


