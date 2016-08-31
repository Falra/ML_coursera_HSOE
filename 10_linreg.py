import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


def proc_df(df):
    df['FullDescription'] = df['FullDescription'].str.lower()
    df['FullDescription'] = df['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    df['LocationNormalized'].fillna('nan', inplace=True)
    df['ContractTime'].fillna('nan', inplace=True)

tfidf_vectorizer = TfidfVectorizer(min_df=5)
enc = DictVectorizer()

train_df = pd.read_csv('salary-train.csv')
test_df = pd.read_csv('salary-test-mini.csv')
proc_df(train_df)
proc_df(test_df)
train_fidf = tfidf_vectorizer.fit_transform(train_df['FullDescription'])
test_tfidf = tfidf_vectorizer.transform(test_df['FullDescription'])
cat_train_df = enc.fit_transform(train_df[['LocationNormalized', 'ContractTime']].to_dict('records'))
cat_test_df = enc.transform(test_df[['LocationNormalized', 'ContractTime']].to_dict('records'))
train_X = hstack([train_fidf, cat_train_df])
test_X = hstack([test_tfidf, cat_test_df])
clf = Ridge(alpha=1)
clf = clf.fit(train_X, train_df['SalaryNormalized'])
pred = clf.predict(test_X)
for salary in pred:
    print("{:.2f}".format(salary))
