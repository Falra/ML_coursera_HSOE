from __future__ import division
import pandas as pd
data = pd.read_csv('titanic.csv', index_col='PassengerId')
print data.head()
rows = data.shape[0]
male = data[data['Sex'] == 'male'].shape[0]
female = data[data['Sex'] == 'female'].shape[0]
survived = data['Survived'].sum() * 100 / rows
pclass1 = data[data['Pclass'] == 1].shape[0] * 100 / rows
mean_age = data['Age'].mean()
med_age = data['Age'].median()
corr = data['SibSp'].corr(data['Parch'], method='pearson')


def get_name(row):
    if '(' in row:
        parts = row.split('(')[-1].replace(')', '').strip()
    else:
        parts = row.split('.')[-1].replace('(', '').replace(')', '').strip()
    parts = parts.split(' ')
    return parts[0]

female_names = data[data['Sex'] == 'female']['Name']
female_names = female_names.apply(get_name)
print "Male: {}".format(male)
print "Female: {}".format(female)
print "Survived: {:.2f}%".format(survived)
print "Pclass 1: {:.2f}%".format(pclass1)
print "Mean age: {:.2f}".format(mean_age)
print "Median age: {}".format(med_age)
print "Correlation : {:.2f}".format(corr)
print female_names.value_counts()
