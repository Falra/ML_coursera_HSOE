import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
labels = newsgroups.target
features = newsgroups.data
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(features)
words = tfidf_vectorizer.get_feature_names()
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(labels.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(tfidf, labels)
max_score, params = 0, None
for a in gs.grid_scores_:
    if a.mean_validation_score > max_score:
        max_score = a.mean_validation_score
        params = a.parameters
C = params['C']
clf = SVC(kernel='linear', random_state=241, C=C)
clf.fit(tfidf, labels)
matx = np.squeeze(np.asarray(clf.coef_.todense()))
matx = np.abs(matx)
ind = np.argsort(matx)
ind = ind[-10:]
print(ind)
my_words = [words[i] for i in ind]
my_words.sort()
print(my_words)
