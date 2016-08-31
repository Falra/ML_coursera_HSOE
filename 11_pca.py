from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

x_df = pd.read_csv('close_prices.csv', index_col=0)
y_df = pd.read_csv('djia_index.csv', index_col=0)

clf = PCA(n_components=10)
clf.fit(x_df)
ind_val = clf.components_[0].argmax()
print(np.cumsum(clf.explained_variance_ratio_))
print(x_df.columns[ind_val])
print(x_df.columns)
x_df = clf.transform(x_df)
first_col = x_df[:, 0]
cor_coef = np.corrcoef(first_col, y_df['^DJI'])
print("Correlation: {:.2f}".format(cor_coef[0, 1]))
