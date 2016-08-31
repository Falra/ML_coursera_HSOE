import pandas as pd
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import time
import datetime


data_train = pd.read_csv('features.csv', index_col = 'match_id')
data_train = data_train.drop(['start_time', 'duration','tower_status_radiant','tower_status_dire','barracks_status_radiant','barracks_status_dire'],1)
test = pd.read_csv('features_test.csv', index_col = 'match_id')
target = data_train.radiant_win
train = data_train.drop('radiant_win',1)
test = test.drop('start_time',1)
print train.shape
print test.shape

cols = train.columns.values.tolist()
#print cols

def fillNA(data, val):
    df = data.fillna(val)
    return df
    
def getBest(train, target):
    parameters = {'C':[0.001, 0.01, 0.1], 
                  'random_state':[2,10,42,100,241],
                  }
    clf = LogisticRegression()
    X = StandardScaler().fit_transform(train)
    cv = KFold(len(train), n_folds=5, shuffle=True, random_state=241)
    gs_clf = GridSearchCV(clf, parameters, cv=cv, scoring='roc_auc', verbose=True, n_jobs=4)
    gs_clf.fit(X, target)
    print "BEST ESTIMATOR:", gs_clf.best_estimator_
    print 'Best score: %0.6f' % gs_clf.best_score_
    print 'Best parameters set:'
    best_parameters = gs_clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
        
    return gs_clf.best_estimator_
    
     
def GBC(train, target):
    start_time = datetime.datetime.now()
    
    for i in [10, 20, 30, 40]:
        c_start_time = datetime.datetime.now()
        clf = GradientBoostingClassifier(random_state=241, n_estimators=i)
        cv = KFold(len(train), n_folds=5, shuffle=True, random_state=241)
        scores = cross_validation.cross_val_score(clf, train, target, cv=cv, scoring = 'roc_auc')
        print("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), "GradientBoostingClassifier"))
        print 'Time with '+str(i)+' estimators:', datetime.datetime.now() - c_start_time
        
    print 'Time elapsed:', datetime.datetime.now() - start_time
    
    
def logReg(train, target, clf):
    start_time = datetime.datetime.now()    
    X = StandardScaler().fit_transform(train)
    cv = KFold(len(train), n_folds=5, shuffle=True, random_state=241)
    scores = cross_validation.cross_val_score(clf, X, target, cv=cv, scoring = 'roc_auc')
    
    print("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'LogisticRegression'))    
    print 'Time elapsed:', datetime.datetime.now() - start_time
    
def dropCat(data):   
    data = data.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], 1)    
    return data

# for LogisticRegression 
train_filled_0 = fillNA(train, 0)
test_filled_0 = fillNA(test, 0)
# for GradientBoostingClassifier
train_filled_1 = fillNA(train, 1000000000)
test_filled_1 = fillNA(test, 1000000000)
GBC(train_filled_1,  target)
#clf = LogisticRegression(C=0.01, random_state=241)

clf = getBest(train_filled_0,  target)

print 'Filled Nan, with heroes'
logReg(train_filled_0, target, clf)

train_drop_category = dropCat(train_filled_0)
print 'Filled Nan,dropped heroes'
logReg(train_drop_category,  target, clf)

X_pick = np.zeros((train_filled_0.shape[0], 113))

for i, match_id in enumerate(train_filled_0.index):
    for p in xrange(5):
        X_pick[i, train_filled_0.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, train_filled_0.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
     
codes = pd.DataFrame(X_pick, columns=range(1,114))

bigdata = pd.concat([train_filled_0, codes],  axis=1, join_axes=[train_filled_0.index])
bigdata = fillNA(bigdata, 0)
for col in bigdata.columns.values.tolist():
    if bigdata[col].sum() == 0.0:
        bigdata.drop(col, 1)
        
print 'Filled Nan, with heroes, with codes'
clf = getBest(bigdata,  target)
logReg(bigdata,  target, clf)

bigdata2 = pd.concat([train_drop_category, codes],  axis=1, join_axes=[train_drop_category.index])
bigdata2 = fillNA(bigdata2, 0)
for col in bigdata2.columns.values.tolist():
    if bigdata2[col].sum() == 0.0:
        bigdata2.drop(col, 1)
print 'Filled Nan, with codes, without heroes'
clf = getBest(bigdata2,  target)
logReg(bigdata2,  target, clf)
clf.fit(bigdata2, target)

X_pick = np.zeros((test_filled_0.shape[0], 113))

for i, match_id in enumerate(test_filled_0.index):
    for p in xrange(5):
        X_pick[i, test_filled_0.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, test_filled_0.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
     
codes = pd.DataFrame(X_pick, columns=range(1,114))

testdata = pd.concat([test_filled_0, codes],  axis=1, join_axes=[test_filled_0.index])
testdata = fillNA(testdata, 0)
for col in testdata.columns.values.tolist():
    if testdata[col].sum() == 0.0:
        testdata.drop(col, 1)
testdata = dropCat(testdata)        
pred = clf.predict_proba(testdata)
print pred[0].min(), pred[0].max()


     
        
        
        
        

