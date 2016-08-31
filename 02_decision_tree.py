from __future__ import division
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv', index_col='PassengerId')
data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
data = data.dropna()
data.loc[data['Sex'] == 'male', 'Sex'] = 1.0
data.loc[data['Sex'] == 'female', 'Sex'] = 0.0
clf = DecisionTreeClassifier(random_state=241, min_samples_leaf=5)
features = data[['Pclass', 'Fare', 'Age', 'Sex']]
features.dropna()
print features.shape[0]
print features.head()
objects = data['Survived']
clf.fit(features, objects)
importance = clf.feature_importances_
print importance

from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=['Pclass', 'Fare', 'Age', 'Sex'],
                     class_names=['Not Survived', 'Survived'],
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("titanic.pdf")
